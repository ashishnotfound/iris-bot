"""

lib/hermes_runner.py â Hermes/Iris Agent Core Runner (Phase 2)

Full multi-modal agent turn loop:

  â¢ Multi-provider LLM (Gemini â OpenRouter â NVIDIA NIM) with key-ring failover

  â¢ Voice message speech-to-text via Groq Whisper (free)

  â¢ Photo/image understanding via vision-capable LLM

  â¢ Image generation via Pollinations.ai (Flux, zero-key, free)

  â¢ Persistent memory (MEMORY.md + USER.md) via Supabase

  â¢ Composio v3 tool integration (1000+ app actions)

  â¢ Autonomous cron job management with retry + idempotency

  â¢ Natural-language automation creation

  â¢ Web/internet search (DuckDuckGo, no key)

  â¢ Amazon/Flipkart business intelligence snapshot

  â¢ Job status and cancellation

  â¢ Full Telegram command center

"""

import base64

import json

import logging

import os

import re

from typing import Any, Dict, List, Optional

from lib.auth import is_allowed

from lib.business_snapshot import BusinessSnapshotManager

from lib.composio_client import ComposioClient

from lib.image_gen_client import ImageGenClient

from lib.llm_provider import ProviderRegistry

from lib.memory_manager import MemoryManager

from lib.stt_client import GroqSTT

from lib.task_router import MODEL_CATALOG, ModelTier, TaskRouter

from lib.telegram_client import TelegramClient

from lib.web_search import WebSearchClient

logger = logging.getLogger(__name__)

# Telemetry store per chat

_last_routing_telemetry: Dict[int, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------

# Keyword patterns for automatic context injection

# ---------------------------------------------------------------------------

# Patterns that suggest a web search would enrich the answer

_WEB_SEARCH_PATTERNS = re.compile(

    r"\b(news|latest|current|today|recent|2024|2025|2026|update|announcement|"

    r"trending|weather|price|rate|score|result|election|stock|crypto|"

    r"what happened|who won|what is the)\b",

    re.IGNORECASE,

)

# Patterns that suggest business context should be injected

_BUSINESS_PATTERNS = re.compile(

    r"\b(business|amazon|flipkart|order|orders|sale|sales|revenue|inventory|"

    r"stock|products?|seller|pending|shipped|returns?|cancel|summary|report|today.*biz)",

    re.IGNORECASE,

)

# ---------------------------------------------------------------------------

# Supabase helpers for conversation history, sessions, & model mode

# ---------------------------------------------------------------------------

def _supabase_headers() -> Dict[str, str]:

    key = (

        os.environ.get("SUPABASE_SERVICE_KEY", "")

        or os.environ.get("SUPABASE_KEY", "")

    ).strip()

    return {

        "apikey": key,

        "Authorization": f"Bearer {key}",

        "Content-Type": "application/json",

    }

def _supabase_url() -> str:

    return os.environ.get("SUPABASE_URL", "").rstrip("/")

def _ensure_session(chat_id: int, model: str = "auto") -> str:

    """Get or create a session_id for a chat. Returns session_id."""

    import requests

    base = _supabase_url()

    if not base:

        return "local"

    try:

        r = requests.get(

            f"{base}/rest/v1/sessions",

            headers=_supabase_headers(),

            params={"chat_id": f"eq.{chat_id}", "select": "session_id"},

            timeout=6,

        )

        if r.status_code == 200 and r.json():

            return r.json()[0]["session_id"]

        # Create new session

        import uuid

        sid = str(uuid.uuid4())

        requests.post(

            f"{base}/rest/v1/sessions",

            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates"},

            json={"chat_id": chat_id, "session_id": sid, "model": model, "platform": "telegram"},

            timeout=6,

        )

        return sid

    except Exception as e:

        logger.warning("Session lookup failed: %s", e)

        return "local"

def _get_session_model(chat_id: int) -> str:

    """Fetch session model override (default 'auto')."""

    import requests

    base = _supabase_url()

    if not base:

        return "auto"

    try:

        r = requests.get(

            f"{base}/rest/v1/sessions",

            headers=_supabase_headers(),

            params={"chat_id": f"eq.{chat_id}", "select": "model"},

            timeout=6,

        )

        if r.status_code == 200 and r.json():

            return r.json()[0].get("model") or "auto"

    except Exception as e:

        logger.warning("Session model lookup failed: %s", e)

    return "auto"

def _set_session_model(chat_id: int, model: str) -> bool:

    """Update session model override in Supabase."""

    import requests

    base = _supabase_url()

    if not base:

        return False

    try:

        r = requests.patch(

            f"{base}/rest/v1/sessions",

            headers=_supabase_headers(),

            params={"chat_id": f"eq.{chat_id}"},

            json={"model": model},

            timeout=6,

        )

        return r.status_code in (200, 204)

    except Exception as e:

        logger.error("Set session model failed: %s", e)

        return False

def _load_messages(chat_id: int, session_id: str, limit: int = 40) -> List[Dict[str, Any]]:

    """Load recent conversation history from Supabase."""

    import requests

    base = _supabase_url()

    if not base or session_id == "local":

        return []

    try:

        r = requests.get(

            f"{base}/rest/v1/messages",

            headers=_supabase_headers(),

            params={

                "chat_id": f"eq.{chat_id}",

                "session_id": f"eq.{session_id}",

                "select": "role,content,metadata",

                "order": "created_at.asc",

                "limit": limit,

            },

            timeout=6,

        )

        if r.status_code == 200:

            rows = r.json()

            msgs = []

            for row in rows:

                content = row["content"]

                # content is stored as JSONB â could be a string or a list

                if isinstance(content, str):

                    msgs.append({"role": row["role"], "content": content})

                else:

                    msgs.append({"role": row["role"], "content": content})

            return msgs

    except Exception as e:

        logger.warning("Failed to load messages: %s", e)

    return []

def _save_message(chat_id: int, session_id: str, role: str, content: Any) -> None:

    """Persist a single message to Supabase."""

    import requests

    base = _supabase_url()

    if not base or session_id == "local":

        return

    try:

        requests.post(

            f"{base}/rest/v1/messages",

            headers=_supabase_headers(),

            json={

                "chat_id": chat_id,

                "session_id": session_id,

                "role": role,

                "content": content,

            },

            timeout=6,

        )

    except Exception as e:

        logger.warning("Failed to save message: %s", e)

# ---------------------------------------------------------------------------

# Image / voice download helpers

# ---------------------------------------------------------------------------

def _download_photo(tg: TelegramClient, photo_list: List[Dict[str, Any]]) -> Optional[bytes]:

    """Download the highest-resolution photo from a Telegram message."""

    if not photo_list:

        return None

    # Telegram sends photo sizes sorted ascending â last is largest

    best = photo_list[-1]

    file_id = best.get("file_id")

    if not file_id:

        return None

    info = tg.get_file(file_id)

    if not info.get("ok"):

        return None

    file_path = info["result"]["file_path"]

    return tg.download_file(file_path)

def _download_voice(tg: TelegramClient, voice: Dict[str, Any]) -> Optional[bytes]:

    """Download a Telegram voice message and return raw bytes."""

    file_id = voice.get("file_id")

    if not file_id:

        return None

    info = tg.get_file(file_id)

    if not info.get("ok"):

        return None

    file_path = info["result"]["file_path"]

    return tg.download_file(file_path)

def _photo_to_content_part(photo_bytes: bytes) -> Dict[str, Any]:

    """Convert raw JPEG bytes to an OpenAI vision content part."""

    b64 = base64.b64encode(photo_bytes).decode("utf-8")

    return {

        "type": "image_url",

        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},

    }

# ---------------------------------------------------------------------------

# Cron helper

# ---------------------------------------------------------------------------

def _list_cron_jobs(chat_id: int) -> List[Dict[str, Any]]:

    import requests

    base = _supabase_url()

    if not base:

        return []

    try:

        r = requests.get(

            f"{base}/rest/v1/cron_jobs",

            headers=_supabase_headers(),

            params={

                "chat_id": f"eq.{chat_id}",

                "select": "job_id,cron_expression,task_description,enabled,next_run_at",

                "order": "created_at.asc",

            },

            timeout=6,

        )

        if r.status_code == 200:

            return r.json()

    except Exception:

        pass

    return []

def _add_cron_job(chat_id: int, cron_expression: str, task: str) -> bool:

    import requests

    base = _supabase_url()

    if not base:

        return False

    try:

        from lib.cron_manager import next_run_for

        next_run = next_run_for(cron_expression)

        r = requests.post(

            f"{base}/rest/v1/cron_jobs",

            headers=_supabase_headers(),

            json={

                "chat_id": chat_id,

                "cron_expression": cron_expression,

                "task_description": task,

                "next_run_at": next_run,

            },

            timeout=6,

        )

        return r.status_code in (200, 201)

    except Exception as e:

        logger.error("add_cron_job error: %s", e)

        return False

def _delete_cron_job(job_id: str) -> bool:

    import requests

    base = _supabase_url()

    if not base:

        return False

    try:

        r = requests.delete(

            f"{base}/rest/v1/cron_jobs",

            headers=_supabase_headers(),

            params={"job_id": f"eq.{job_id}"},

            timeout=6,

        )

        return r.status_code in (200, 204)

    except Exception as e:

        logger.error("delete_cron_job error: %s", e)

        return False

# ---------------------------------------------------------------------------

# Main agent turn with Dynamic Task Routing

# ---------------------------------------------------------------------------

_registry = ProviderRegistry()

_memory = MemoryManager()

_stt = GroqSTT()

_router = TaskRouter()

_biz = BusinessSnapshotManager()

_web = WebSearchClient()

def _is_update_processed(update_id: int) -> bool:

    """Check if a Telegram update_id was already processed in Supabase."""

    import requests

    base = _supabase_url()

    if not base or not update_id:

        return False

    try:

        r = requests.get(

            f"{base}/rest/v1/processed_updates",

            headers=_supabase_headers(),

            params={"update_id": f"eq.{update_id}", "select": "update_id"},

            timeout=4,

        )

        return r.status_code == 200 and bool(r.json())

    except Exception:

        return False

def _mark_update_processed(update_id: int, chat_id: int) -> bool:

    """Mark a Telegram update_id as processed in Supabase."""

    import requests

    base = _supabase_url()

    if not base or not update_id:

        return False

    try:

        r = requests.post(

            f"{base}/rest/v1/processed_updates",

            headers=_supabase_headers(),

            json={"update_id": update_id, "chat_id": chat_id},

            timeout=4,

        )

        return r.status_code in (200, 201)

    except Exception:

        return False

def execute_agent_turn(

    chat_id: int,

    user_message: str,

    *,

    photo: Optional[List[Dict[str, Any]]] = None,

    voice: Optional[Dict[str, Any]] = None,

    supabase_client=None,

    telegram_client: Optional[TelegramClient] = None,

) -> Dict[str, Any]:

    """Execute a single turn of Iris Agent for a given Telegram chat_id."""

    if not telegram_client:

        telegram_client = TelegramClient()

    # 1. Authorization

    if not is_allowed(chat_id):

        logger.warning("Unauthorized chat_id: %s", chat_id)

        telegram_client.send_message(chat_id, "â ï¸ Access denied. Your Telegram ID is not authorized.")

        return {"status": "unauthorized"}

    # 2. Route to command handler first

    clean = user_message.strip()

    cmd_result = _handle_command(chat_id, clean, telegram_client)

    if cmd_result is not None:

        return cmd_result

    # 3. Voice message â STT â text

    voice_transcript: Optional[str] = None

    if voice:

        telegram_client.send_chat_action(chat_id, "typing")

        audio_bytes = _download_voice(telegram_client, voice)

        if audio_bytes:

            stt_result = _stt.transcribe(audio_bytes)

            if stt_result.get("success"):

                voice_transcript = stt_result["transcript"]

                telegram_client.send_message(

                    chat_id,

                    f"ð¤ *Transcribed:* _{voice_transcript}_",

                    parse_mode="Markdown",

                )

                clean = voice_transcript

            else:

                err = stt_result.get("error", "Unknown STT error")

                telegram_client.send_message(chat_id, f"ð Voice transcription failed: {err}")

                return {"status": "error", "error": err}

        else:

            telegram_client.send_message(chat_id, "â ï¸ Failed to download voice message.")

            return {"status": "error", "error": "voice download failed"}

    # 4. Image generation intent detection

    msg_lower = clean.lower()

    is_image_cmd = (

        msg_lower.startswith("/image")

        or msg_lower.startswith("/generate_image")

        or "generate an image" in msg_lower

        or "create an image" in msg_lower

        or "draw an image" in msg_lower

        or "generate image of" in msg_lower

        or "draw me" in msg_lower

    )

    if is_image_cmd:

        return _handle_image_generation(chat_id, clean, msg_lower, telegram_client)

    # 5. Load session + conversation history

    session_id = _ensure_session(chat_id)

    history = _load_messages(chat_id, session_id)

    # 6. Load persistent memory â build system prompt

    mem = _memory.load(chat_id)

    system_prompt = _memory.build_system_prompt(mem["memory_md"], mem["user_md"])

    # 7. Build message list for LLM

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    messages.extend(history)

    # 8. Build user content (text and/or vision)

    is_vision = False

    if photo:

        telegram_client.send_chat_action(chat_id, "typing")

        photo_bytes = _download_photo(telegram_client, photo)

        if photo_bytes:

            is_vision = True

            user_content: Any = [

                {"type": "text", "text": clean or "Describe this image in detail."},

                _photo_to_content_part(photo_bytes),

            ]

        else:

            user_content = clean or "Please describe the image (download failed)."

    else:

        user_content = clean

    messages.append({"role": "user", "content": user_content})

    telegram_client.send_chat_action(chat_id, "typing")

    # ââ Context Injection: Business Snapshot ââ

    if _BUSINESS_PATTERNS.search(clean):

        biz_context = _biz.format_for_llm(chat_id)

        if biz_context:

            messages.insert(1, {"role": "system", "content": biz_context})

            logger.debug("Injected business snapshot context for chat_id=%s", chat_id)

    # ââ Context Injection: Web Search ââ

    if _WEB_SEARCH_PATTERNS.search(clean) and not is_vision:

        search_results = _web.search(clean, max_results=5)

        if search_results:

            web_ctx = _web.format_for_llm(search_results, query=clean)

            messages.insert(1, {"role": "system", "content": web_ctx})

            logger.debug(

                "Injected %d web search results for chat_id=%s",

                len(search_results), chat_id,

            )

    # 9. Dynamic Task Routing

    composio = ComposioClient()

    has_tools = composio.is_configured() and bool(composio.get_connected_accounts())

    session_model_override = _get_session_model(chat_id)

    decision = _router.route(

        clean,

        history=history,

        has_photo=is_vision,

        tools_available=has_tools,

        manual_model_override=session_model_override,

        active_providers=_registry.available_providers(),

    )

    # 10. LLM call with dynamic candidate failover

    try:

        reply, provider_used, model_used = _registry.chat_completion(

            messages,

            candidates=decision.candidates,

            vision=is_vision,

        )

    except RuntimeError as e:

        err_msg = f"â ï¸ All AI providers failed:\n{e}"

        telegram_client.send_message(chat_id, err_msg)

        return {"status": "error", "error": str(e)}

    # 11. Mid-Task Escalation Check

    if _router.should_escalate(reply, decision.tier):

        logger.info("Escalating task to POWERFUL tier for chat_id=%s", chat_id)

        powerful_candidates = [

            c for c in decision.candidates if c[2].tier == ModelTier.POWERFUL

        ]

        if powerful_candidates:

            try:

                esc_reply, esc_prov, esc_model = _registry.chat_completion(

                    messages,

                    candidates=powerful_candidates,

                    vision=is_vision,

                )

                reply, provider_used, model_used = esc_reply, esc_prov, esc_model

                decision.escalated = True

                decision.tier = ModelTier.POWERFUL

            except Exception as esc_err:

                logger.warning("Escalation turn failed (retaining initial reply): %s", esc_err)

    # Telemetry recording

    _last_routing_telemetry[chat_id] = {

        "tier": str(decision.tier),

        "provider": provider_used,

        "model": model_used,

        "reason": decision.reason,

        "escalated": decision.escalated,

        "is_manual": decision.is_manual_override,

    }

    # 12. Persist messages

    _save_message(chat_id, session_id, "user", user_content)

    _save_message(chat_id, session_id, "assistant", reply)

    # 13. Send reply

    telegram_client.send_message(chat_id, reply)

    # 14. Bounded memory update (fast 256 max_tokens cap)

    try:

        def _llm_for_memory(prompt: str) -> str:

            text, _, _ = _registry.chat_completion(

                [{"role": "user", "content": prompt}],

                max_tokens=256,

            )

            return text

        updated_msgs = messages + [{"role": "assistant", "content": reply}]

        _memory.extract_and_update(chat_id, updated_msgs, _llm_for_memory)

    except Exception as e:

        logger.warning("Memory update failed (non-critical): %s", e)

    return {

        "status": "success",

        "response": reply,

        "provider": provider_used,

        "model": model_used,

        "tier": str(decision.tier),

    }

# ---------------------------------------------------------------------------

# Image generation handler

# ---------------------------------------------------------------------------

def _handle_image_generation(

    chat_id: int,

    clean: str,

    msg_lower: str,

    tg: TelegramClient,

) -> Dict[str, Any]:

    tg.send_chat_action(chat_id, "upload_photo")

    prompt = clean

    for prefix in [

        "/image", "/generate_image",

        "generate an image of", "generate an image",

        "create an image of", "create an image",

        "draw an image of", "draw an image",

        "draw me a", "draw me",

        "generate image of",

    ]:

        if msg_lower.startswith(prefix):

            prompt = clean[len(prefix):].strip(" :,-")

            break

    if not prompt:

        prompt = "A futuristic cyberpunk cityscape at sunset with flying vehicles"

    img = ImageGenClient()

    result = img.generate_image(prompt)

    if result.get("success"):

        url = result["image_url"]

        caption = f"ð¨ *Generated Image*\n*Prompt:* {prompt[:200]}\n*Provider:* {result.get('provider', 'Pollinations')}"

        res = tg.send_photo(chat_id, photo_url=url, caption=caption)

        if not res.get("ok"):

            tg.send_message(chat_id, f"ð¨ Image generated!\nð [View Image]({url})", parse_mode="Markdown")

        return {"status": "success", "type": "image", "image_url": url}

    else:

        tg.send_message(chat_id, f"â ï¸ Image generation failed: {result.get('error')}")

        return {"status": "error", "error": result.get("error")}

# ---------------------------------------------------------------------------

# Command center

# ---------------------------------------------------------------------------

def _handle_command(

    chat_id: int,

    text: str,

    tg: TelegramClient,

) -> Optional[Dict[str, Any]]:

    """Handle bot commands. Returns a result dict or None (not a command)."""

    if not text.startswith("/"):

        return None

    parts = text.split(maxsplit=2)

    p2 = handle_phase2_command(parts[0].lower().split("@")[0], parts, chat_id, tg)
    if p2 is not None:
        return p2
    cmd = parts[0].lower().split("@")[0]  # strip @BotName suffix if present

    # /start  /help

    if cmd in ("/start", "/help"):

        avail = ", ".join(_registry.available_providers()) or "none configured"

        stt_status = "â Ready (Groq Whisper)" if _stt.is_configured() else "â ï¸ GROQ_API_KEY missing"

        mem_status = "â Supabase" if _memory.is_configured() else "â ï¸ in-process only"

        msg = (

            "ð¤ *Iris â Personal AI Agent*\n\n"

            f"ð§  *LLM Providers:* `{avail}`\n"

            f"ð¯ *Routing:* Dynamic Task-Based (`/model auto` active)\n"

            f"ð¤ *Speech-to-Text:* {stt_status}\n"

            f"ð¾ *Memory:* {mem_status}\n\n"

            "ð *Commands:*\n"

            "â¢ `/image <prompt>` â Generate AI image\n"

            "â¢ `/models` â List available free LLM models\n"

            "â¢ `/model auto` â Enable dynamic task routing\n"

            "â¢ `/model <name>` â Override active model\n"

            "â¢ `/memory` â View stored memory\n"

            "â¢ `/forget` â Clear stored memory\n"

            "â¢ `/tools` â List Composio connected tools\n"

            "â¢ `/status` â System status & router telemetry\n"

            "â¢ `/cron list` â List scheduled jobs\n"

            "â¢ `/cron add <expr> <task>` â Add cron job\n"

            "â¢ `/cron del <job_id>` â Delete cron job\n\n"

            "ð¬ Just send any message, image, or voice note to chat!"

        )

        tg.send_message(chat_id, msg, parse_mode="Markdown")

        return {"status": "success", "command": "help"}

    # /status

    if cmd == "/status":

        avail = _registry.available_providers()

        mem = _memory.load(chat_id)

        has_mem = bool(mem["memory_md"] or mem["user_md"])

        composio = ComposioClient()

        accounts = composio.get_connected_accounts() if composio.is_configured() else []

        tools_summary = ", ".join(

            a.get("toolkit", {}).get("slug", "?") for a in accounts[:5]

        ) or "None"

        current_override = _get_session_model(chat_id)

        mode_str = "â¡ Auto (Dynamic Task Router)" if current_override == "auto" else f"ð Manual Override (`{current_override}`)"

        telem = _last_routing_telemetry.get(chat_id, {})

        last_tier = telem.get("tier", "N/A")

        last_model = telem.get("model", "N/A")

        last_prov = telem.get("provider", "N/A")

        last_reason = telem.get("reason", "N/A")

        escalated_str = "Yes ð" if telem.get("escalated") else "No"

        msg = (

            "ð *Iris Agent Status*\n\n"

            f"ð¯ *Routing Mode:* {mode_str}\n"

            f"â¡ *Last Tier Used:* `{last_tier}` ({last_prov} / `{last_model}`)\n"

            f"ð *Last Routing Reason:* _{last_reason}_\n"

            f"ð *Escalated Turn:* {escalated_str}\n\n"

            f"ð§  *Available Providers:* {', '.join(avail) or 'none'}\n"

            f"ð¤ *Groq STT:* {'â' if _stt.is_configured() else 'â GROQ_API_KEY missing'}\n"

            f"ð¾ *Memory:* {'Loaded â' if has_mem else 'Empty'}\n"

            f"ð *Composio Tools:* `{tools_summary}`\n"

            f"ð¼ *Image Gen:* Pollinations.ai (Flux) â\n"

        )

        tg.send_message(chat_id, msg, parse_mode="Markdown")

        return {"status": "success", "command": "status"}

    # /models

    if cmd == "/models":

        tg.send_chat_action(chat_id, "typing")

        lines = ["ð *Model Catalog & Tiers*\n"]

        fast_specs = [s for s in MODEL_CATALOG if s.tier == ModelTier.FAST]

        bal_specs = [s for s in MODEL_CATALOG if s.tier == ModelTier.BALANCED]

        pow_specs = [s for s in MODEL_CATALOG if s.tier == ModelTier.POWERFUL]

        lines.append("â¡ *FAST Tiers (Simple Q&A / Quick Chat):*")

        for s in fast_specs:

            lines.append(f"  â¢ `{s.provider}` / `{s.model_id}`")

        lines.append("\nâï¸ *BALANCED Tiers (Research / Summaries / Chat):*")

        for s in bal_specs:

            lines.append(f"  â¢ `{s.provider}` / `{s.model_id}`")

        lines.append("\nð§  *POWERFUL Tiers (Coding / Architecture / Debugging):*")

        for s in pow_specs:

            lines.append(f"  â¢ `{s.provider}` / `{s.model_id}`")

        lines.append("\nð¡ Use `/model auto` for automatic task routing, or `/model <model-id>` to pin.")

        tg.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")

        return {"status": "success", "command": "models"}

    # /model <name|auto>

    if cmd == "/model":

        if len(parts) < 2:

            curr = _get_session_model(chat_id)

            tg.send_message(

                chat_id,

                f"ð *Current Model Mode:* `{curr}`\n\n"

                "â¢ `/model auto` â Enable dynamic task routing (recommended)\n"

                "â¢ `/model <model-id>` â Pin a specific model\n"

                "  Example: `/model gemini-2.5-flash`",

                parse_mode="Markdown",

            )

        else:

            target = parts[1].strip().lower()

            if target == "auto":

                _set_session_model(chat_id, "auto")

                tg.send_message(

                    chat_id,

                    "â *Dynamic Task Routing Enabled* (`auto` mode).\n"

                    "Iris will now select the best model engine for each task automatically.",

                    parse_mode="Markdown",

                )

            else:

                _set_session_model(chat_id, target)

                tg.send_message(

                    chat_id,

                    f"ð Model pinned to `{target}` for this session.\n"

                    "Run `/model auto` anytime to return to dynamic task routing.",

                    parse_mode="Markdown",

                )

        return {"status": "success", "command": "model"}

    # /memory

    if cmd == "/memory":

        mem = _memory.load(chat_id)

        memory_md = mem["memory_md"] or "(empty)"

        user_md = mem["user_md"] or "(empty)"

        msg = (

            f"ð§  *MEMORY.md:*\n```\n{memory_md[:800]}\n```\n\n"

            f"ð¤ *USER.md:*\n```\n{user_md[:400]}\n```"

        )

        tg.send_message(chat_id, msg, parse_mode="Markdown")

        return {"status": "success", "command": "memory"}

    # /forget

    if cmd == "/forget":

        _memory.clear(chat_id)

        tg.send_message(chat_id, "ð Memory cleared. I'll start fresh with no prior knowledge of you.")

        return {"status": "success", "command": "forget"}

    # /tools

    if cmd == "/tools":

        composio = ComposioClient()

        if not composio.is_configured():

            tg.send_message(chat_id, "â ï¸ Composio not configured. Set COMPOSIO_API_KEY.")

        else:

            accounts = composio.get_connected_accounts()

            if not accounts:

                tg.send_message(chat_id, "No Composio tools connected yet.")

            else:

                lines = ["ð *Connected Composio Tools:*\n"]

                for acc in accounts:

                    slug = acc.get("toolkit", {}).get("slug", "?")

                    status = acc.get("status", "?")

                    lines.append(f"  â¢ `{slug}` â {status}")

                tg.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")

        return {"status": "success", "command": "tools"}

    # /cron

    if cmd == "/cron":

        if len(parts) < 2:

            tg.send_message(

                chat_id,

                "â° *Cron Job Manager*\n\n"

                "â¢ `/cron list` â View scheduled jobs\n"

                "â¢ `/cron add <expr> <task>` â Add job\n"

                "  Example: `/cron add 0 9 * * * Send me the weather forecast`\n"

                "â¢ `/cron del <job_id>` â Remove a job",

                parse_mode="Markdown",

            )

            return {"status": "success", "command": "cron"}

        subcmd = parts[1].lower()

        if subcmd == "list":

            jobs = _list_cron_jobs(chat_id)

            if not jobs:

                tg.send_message(chat_id, "No scheduled jobs. Use `/cron add <expr> <task>` to create one.", parse_mode="Markdown")

            else:

                lines = ["â° *Scheduled Jobs:*\n"]

                for j in jobs:

                    status_icon = "â" if j.get("enabled") else "â¸"

                    jid = str(j["job_id"])[:8]

                    lines.append(

                        f"{status_icon} `{jid}` â `{j['cron_expression']}`\n"

                        f"   _{j['task_description'][:80]}_"

                    )

                tg.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")

            return {"status": "success", "command": "cron_list"}

        if subcmd == "add":

            # Format: /cron add <expr_5_parts> <task description>

            # e.g.: /cron add 0 9 * * * Good morning briefing

            remaining = parts[2] if len(parts) > 2 else ""

            tokens = remaining.split()

            if len(tokens) < 6:

                tg.send_message(

                    chat_id,

                    "â ï¸ Usage: `/cron add <min> <hr> <day> <mon> <dow> <task>`\n"

                    "Example: `/cron add 0 9 * * * Daily morning briefing`",

                    parse_mode="Markdown",

                )

            else:

                cron_expr = " ".join(tokens[:5])

                task_desc = " ".join(tokens[5:])

                ok = _add_cron_job(chat_id, cron_expr, task_desc)

                if ok:

                    tg.send_message(chat_id, f"â Cron job added!\nâ° `{cron_expr}`\nð _{task_desc}_", parse_mode="Markdown")

                else:

                    tg.send_message(chat_id, "â ï¸ Failed to add cron job. Check that Supabase is configured.")

            return {"status": "success", "command": "cron_add"}

        if subcmd == "del":

            job_id = parts[2].strip() if len(parts) > 2 else ""

            if not job_id:

                tg.send_message(chat_id, "Usage: `/cron del <job_id>`", parse_mode="Markdown")

            else:

                # Allow short ID prefix matching

                jobs = _list_cron_jobs(chat_id)

                matched = [j for j in jobs if j["job_id"].startswith(job_id)]

                if not matched:

                    tg.send_message(chat_id, f"â ï¸ No job found with ID starting with `{job_id}`.", parse_mode="Markdown")

                elif len(matched) > 1:

                    tg.send_message(chat_id, "â ï¸ Multiple jobs match that prefix. Please be more specific.")

                else:

                    full_id = matched[0]["job_id"]

                    ok = _delete_cron_job(full_id)

                    if ok:

                        tg.send_message(chat_id, f"ð Cron job `{full_id[:8]}` deleted.", parse_mode="Markdown")

                    else:

                        tg.send_message(chat_id, "â ï¸ Failed to delete cron job.")

            return {"status": "success", "command": "cron_del"}

        tg.send_message(chat_id, "Unknown cron subcommand. Try `/cron` for help.", parse_mode="Markdown")

        return {"status": "success", "command": "cron_unknown"}

    return None  # Not a recognized command

# ---------------------------------------------------------------------------

# Phase 2 commands added to _handle_command (injected as standalone functions

# so they can reference _registry, _biz, _web from module scope)

# ---------------------------------------------------------------------------

def handle_phase2_command(cmd: str, parts, chat_id: int, tg) -> Optional[Dict[str, Any]]:

    """Handle Phase 2 commands: /automate, /jobs, /cancel, /timezone, /search, /sync."""

    # /automate <natural language description>

    if cmd == "/automate":

        description = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

        if not description:

            tg.send_message(

                chat_id,

                "?? *Automate a Task*\n\nDescribe what to schedule in plain English:\n\n"

                " `/automate Every day at 9 AM IST post to Instagram and X`\n"

                " `/automate Every Friday 6 PM send me sales summary`\n"

                " `/automate Daily at 8 AM check Amazon orders`",

                parse_mode="Markdown",

            )

            return {"status": "success", "command": "automate_help"}

        tg.send_chat_action(chat_id, "typing")

        try:

            from lib.automation_parser import AutomationParser, AutomationParseError

            from lib.cron_manager import next_run_for

            def _llm_for_parser(prompt: str) -> str:

                text, _, _ = _registry.chat_completion(

                    [{"role": "user", "content": prompt}], max_tokens=512,

                )

                return text

            parser = AutomationParser(_llm_for_parser)

            automation = parser.parse(description)

            existing = _list_cron_jobs(chat_id)

            dup_id = parser.is_duplicate(

                automation.task_description, automation.cron_expression, existing

            )

            if dup_id:

                tg.send_message(

                    chat_id,

                    f"?? Identical automation already exists (`{dup_id[:8]}`).\n"

                    "Use `/jobs` to see your automations.",

                    parse_mode="Markdown",

                )

                return {"status": "success", "command": "automate_duplicate"}

            ok = _add_cron_job_v2(

                chat_id, automation.cron_expression, automation.task_description,

                timezone=automation.timezone, action_type=automation.action_type,

            )

            if ok:

                next_run = next_run_for(automation.cron_expression)

                try:

                    from datetime import datetime

                    nxt_str = datetime.fromisoformat(

                        next_run.replace("Z", "+00:00")

                    ).strftime("%Y-%m-%d %H:%M UTC")

                except Exception:

                    nxt_str = next_run

                tg.send_message(

                    chat_id,

                    f"? *Automation Created!*\n\n"

                    f"?? *Task:* _{automation.summary}_\n"

                    f"? *Schedule:* `{automation.cron_expression}`\n"

                    f"?? *Timezone:* {automation.timezone}\n"

                    f"?? *Next run:* {nxt_str}\n\n"

                    "Use `/jobs` to manage.",

                    parse_mode="Markdown",

                )

            else:

                tg.send_message(chat_id, "?? Failed to save automation. Check Supabase config.")

        except Exception as e:

            logger.warning("Automation creation failed: %s", e)

            tg.send_message(

                chat_id,

                f"?? *Could not create automation:*\n{str(e)[:300]}\n\n"

                "Be specific: e.g. 'every day at 9 AM IST'.",

                parse_mode="Markdown",

            )

        return {"status": "success", "command": "automate"}

    # /jobs  list all automations with status

    if cmd == "/jobs":

        tg.send_chat_action(chat_id, "typing")

        jobs = _list_cron_jobs_v2(chat_id)

        if not jobs:

            tg.send_message(

                chat_id,

                "No automations scheduled.\n\nUse `/automate <description>` to create one.",

            )

        else:

            lines = ["?? *Your Automations*\n"]

            for j in jobs:

                icon = "?" if j.get("enabled") else "?"

                running = " ??" if j.get("running_since") else ""

                jid = str(j["job_id"])[:8]

                tz = j.get("timezone", "UTC")

                retry = j.get("retry_count", 0)

                line = (

                    f"{icon} `{jid}`  `{j['cron_expression']}` ({tz}){running}\n"

                    f"   _{j['task_description'][:70]}_"

                )

                if retry:

                    line += f"\n   ?? Retry #{retry}"

                if j.get("last_error") and not j.get("enabled"):

                    line += f"\n   ?? {j['last_error'][:80]}"

                lines.append(line)

            lines.append("\n?? `/cancel <id>` to stop, `/automate <desc>` to add.")

            tg.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")

        return {"status": "success", "command": "jobs"}

    # /cancel <job_id_prefix>

    if cmd == "/cancel":

        job_id_prefix = parts[1].strip() if len(parts) > 1 else ""

        if not job_id_prefix:

            tg.send_message(chat_id, "Usage: `/cancel <job_id>`  use `/jobs` to see IDs.", parse_mode="Markdown")

        else:

            jobs = _list_cron_jobs_v2(chat_id)

            matched = [j for j in jobs if j["job_id"].startswith(job_id_prefix)]

            if not matched:

                tg.send_message(chat_id, f"?? No automation with ID `{job_id_prefix}`.", parse_mode="Markdown")

            elif len(matched) > 1:

                tg.send_message(chat_id, "?? Multiple matches  be more specific.")

            else:

                full_id = matched[0]["job_id"]

                ok = _disable_cron_job(full_id)

                if ok:

                    tg.send_message(chat_id, f"? Automation `{full_id[:8]}` cancelled.", parse_mode="Markdown")

                else:

                    tg.send_message(chat_id, "?? Failed to cancel.")

        return {"status": "success", "command": "cancel"}

    # /timezone <IANA tz>

    if cmd == "/timezone":

        tz_name = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

        if not tz_name:

            tg.send_message(

                chat_id,

                "? *Set Your Timezone*\n\nUsage: `/timezone <IANA timezone>`\n\n"

                "Examples:\n `/timezone Asia/Kolkata` (IST)\n `/timezone America/New_York`",

                parse_mode="Markdown",

            )

        elif not _is_valid_timezone(tz_name):

            tg.send_message(chat_id, f"?? Unknown timezone: `{tz_name}`", parse_mode="Markdown")

        else:

            _set_session_timezone(chat_id, tz_name)

            tg.send_message(chat_id, f"? Timezone set to `{tz_name}`.", parse_mode="Markdown")

        return {"status": "success", "command": "timezone"}

    # /search <query>

    if cmd == "/search":

        query = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

        if not query:

            tg.send_message(chat_id, "Usage: `/search <query>`", parse_mode="Markdown")

        else:

            tg.send_chat_action(chat_id, "typing")

            results = _web.search(query, max_results=6)

            if results:

                lines = [f"?? *Search: {query}*\n"]

                for i, r in enumerate(results, 1):

                    snippet = r.get("snippet", "")[:100]

                    lines.append(

                        f"{i}. *{r['title']}*\n   {r['url']}"

                        + (f"\n   _{snippet}_" if snippet else "")

                    )

                tg.send_message(chat_id, "\n\n".join(lines), parse_mode="Markdown")

            else:

                tg.send_message(chat_id, f"No results found for: {query}")

        return {"status": "success", "command": "search"}

    # /sync [amazon|flipkart|all]

    if cmd == "/sync":

        platform_arg = parts[1].strip().lower() if len(parts) > 1 else "all"

        tg.send_chat_action(chat_id, "typing")

        tg.send_message(chat_id, "?? Syncing business data...")

        platforms = ["amazon", "flipkart"] if platform_arg == "all" else [platform_arg]

        lines = ["?? *Sync Results*\n"]

        for platform in platforms:

            if platform == "amazon":

                res = _biz.sync_amazon(chat_id)

            elif platform == "flipkart":

                res = _biz.sync_flipkart(chat_id)

            else:

                lines.append(f"?? Unknown: `{platform}`")

                continue

            if res.get("success"):

                data = res.get("data", {})

                lines.append(f"? *{platform.title()}* synced  Orders: {data.get('total_orders', '?')}")

            else:

                lines.append(f"? *{platform.title()}:* {res.get('error', 'Unknown')[:150]}")

        tg.send_message(chat_id, "\n\n".join(lines), parse_mode="Markdown")

        return {"status": "success", "command": "sync"}

    return None

# ---------------------------------------------------------------------------

# Phase 2 helper functions (called by handle_phase2_command and _handle_command)

# ---------------------------------------------------------------------------

def _add_cron_job_v2(

    chat_id: int, cron_expression: str, task: str,

    *, timezone: str = "UTC", action_type: str = "custom",

) -> bool:

    """Add a cron job with Phase 2 fields."""

    import requests

    from lib.cron_manager import next_run_for

    base = _supabase_url()

    if not base:

        return False

    try:

        next_run = next_run_for(cron_expression)

        r = requests.post(

            f"{base}/rest/v1/cron_jobs",

            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates"},

            json={

                "chat_id": chat_id, "cron_expression": cron_expression,

                "task_description": task, "timezone": timezone,

                "action_type": action_type, "next_run_at": next_run,

            },

            timeout=6,

        )

        return r.status_code in (200, 201)

    except Exception as e:

        logger.error("_add_cron_job_v2 error: %s", e)

        return False

def _list_cron_jobs_v2(chat_id: int) -> List[Dict[str, Any]]:

    """List all cron jobs with Phase 2 fields."""

    import requests

    base = _supabase_url()

    if not base:

        return []

    try:

        r = requests.get(

            f"{base}/rest/v1/cron_jobs",

            headers=_supabase_headers(),

            params={

                "chat_id": f"eq.{chat_id}",

                "select": (

                    "job_id,cron_expression,task_description,timezone,"

                    "enabled,next_run_at,retry_count,last_error,running_since"

                ),

                "order": "created_at.asc",

            },

            timeout=6,

        )

        if r.status_code == 200:

            return r.json()

    except Exception:

        pass

    return []

def _disable_cron_job(job_id: str) -> bool:

    """Disable (cancel) a cron job without deleting it."""

    import requests

    base = _supabase_url()

    if not base:

        return False

    try:

        r = requests.patch(

            f"{base}/rest/v1/cron_jobs",

            headers=_supabase_headers(),

            params={"job_id": f"eq.{job_id}"},

            json={"enabled": False},

            timeout=6,

        )

        return r.status_code in (200, 204)

    except Exception as e:

        logger.error("_disable_cron_job failed: %s", e)

        return False

def _set_session_timezone(chat_id: int, tz_name: str) -> None:

    """Persist user timezone preference in the sessions table."""

    import requests

    base = _supabase_url()

    if not base:

        return

    try:

        requests.patch(

            f"{base}/rest/v1/sessions",

            headers=_supabase_headers(),

            params={"chat_id": f"eq.{chat_id}"},

            json={"timezone": tz_name},

            timeout=6,

        )

    except Exception:

        pass

def _is_valid_timezone(tz_name: str) -> bool:

    """Check if a timezone name is a valid IANA timezone."""

    try:

        import zoneinfo

        zoneinfo.ZoneInfo(tz_name)

        return True

    except Exception:

        pass

    try:

        import pytz

        pytz.timezone(tz_name)

        return True

    except Exception:

        pass

    # Hardcoded fallback for common zones

    return tz_name in {

        "UTC", "Asia/Kolkata", "America/New_York", "America/Los_Angeles",

        "Europe/London", "Europe/Paris", "Asia/Tokyo", "Asia/Singapore",

        "Australia/Sydney", "America/Chicago", "America/Denver",

    }

