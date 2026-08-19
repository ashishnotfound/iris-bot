"""

lib/hermes_runner.py — Hermes/Iris Agent Core Runner (Phase 2)

Full multi-modal agent turn loop:

  • Multi-provider LLM (Gemini → OpenRouter → NVIDIA NIM) with key-ring failover

  • Voice message speech-to-text via Groq Whisper (free)

  • Photo/image understanding via vision-capable LLM

  • Image generation via Pollinations.ai (Flux, zero-key, free)

  • Persistent memory (MEMORY.md + USER.md) via Supabase

  • Composio v3 tool integration (1000+ app actions)

  • Autonomous cron job management with retry + idempotency

  • Natural-language automation creation

  • Web/internet search (DuckDuckGo, no key)

  • Amazon/Flipkart business intelligence snapshot

  • Job status and cancellation

  • Full Telegram command center

"""

import base64

import json

import logging

import os

import re

import requests

from pathlib import Path
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

_PENDING_ACTION_CONFIRMATIONS: Dict[int, Dict[str, Any]] = {}

CONFIRMATION_AFFIRMATIVE_RE = re.compile(
    r"\b(yes|yeah|yep|confirm|confirmed|send it|do it|proceed|go ahead|ok|okay|sure)\b",
    re.IGNORECASE,
)
CONFIRMATION_NEGATIVE_RE = re.compile(
    r"\b(no|nop|nope|cancel|stop|dont|don't|abort|nevermind)\b",
    re.IGNORECASE,
)

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

def _local_sqlite_db_path() -> Path:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / "state.db"

def _init_local_db():
    try:
        db_path = _local_sqlite_db_path()
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS local_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_local_msg_chat_sess ON local_messages(chat_id, session_id)")
            conn.commit()
    except Exception as e:
        logger.warning("Failed to initialize local message SQLite DB: %s", e)

def _local_load_messages(chat_id: Any, session_id: str, limit: int = 40) -> List[Dict[str, Any]]:
    try:
        _init_local_db()
        db_path = _local_sqlite_db_path()
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, id FROM local_messages
                    WHERE chat_id = ? AND session_id = ?
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (str(chat_id), str(session_id), limit)
            )
            rows = cursor.fetchall()
            msgs = []
            for role, raw_content in rows:
                try:
                    content = json.loads(raw_content)
                except Exception:
                    content = raw_content
                msgs.append({"role": role, "content": content})
            return msgs
    except Exception as e:
        logger.warning("Local SQLite message load failed: %s", e)
        return []

def _local_save_message(chat_id: Any, session_id: str, role: str, content: Any) -> None:
    try:
        _init_local_db()
        db_path = _local_sqlite_db_path()
        import sqlite3
        val = json.dumps(content) if not isinstance(content, str) else content
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO local_messages (chat_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (str(chat_id), str(session_id), str(role), val)
            )
            conn.commit()
    except Exception as e:
        logger.warning("Local SQLite message save failed: %s", e)

def _ensure_session(chat_id: int, model: str = "auto") -> str:
    """Get or create a session_id for a chat. Returns session_id."""
    base = _supabase_url()
    fallback_sid = f"local_{chat_id}"
    if not base:
        return fallback_sid

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
        return fallback_sid

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
    """Load recent conversation history from Supabase or local SQLite fallback."""
    base = _supabase_url()
    if not base or session_id.startswith("local"):
        return _local_load_messages(chat_id, session_id, limit)

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
            if rows:
                msgs = []
                for row in rows:
                    content = row["content"]
                    msgs.append({"role": row["role"], "content": content})
                return msgs
    except Exception as e:
        logger.warning("Failed to load messages from Supabase: %s", e)

    return _local_load_messages(chat_id, session_id, limit)

def _save_message(chat_id: int, session_id: str, role: str, content: Any) -> None:
    """Persist a single message to local SQLite storage and Supabase."""
    _local_save_message(chat_id, session_id, role, content)

    base = _supabase_url()
    if not base or session_id.startswith("local"):
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
        logger.warning("Failed to save message to Supabase: %s", e)

# ---------------------------------------------------------------------------

# Image / voice download helpers

# ---------------------------------------------------------------------------

def _download_photo(tg: TelegramClient, photo_list: List[Dict[str, Any]]) -> Optional[bytes]:

    """Download the highest-resolution photo from a Telegram message."""

    if not photo_list:

        return None

    # Telegram sends photo sizes sorted ascending — last is largest

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

def _detect_image_mime(b: bytes) -> str:
    """Detect image MIME type from raw magic bytes."""
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif b.startswith(b"RIFF") and len(b) > 12 and b[8:12] == b"WEBP":
        return "image/webp"
    elif b.startswith(b"GIF8"):
        return "image/gif"
    return "image/jpeg"

def _photo_to_content_part(photo_bytes: bytes) -> Dict[str, Any]:
    """Convert raw photo bytes to an OpenAI vision content part with MIME detection."""
    mime = _detect_image_mime(photo_bytes)
    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
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
    user_message: str = "",
    *,
    photo: Optional[List[Dict[str, Any]]] = None,
    voice: Optional[Dict[str, Any]] = None,
    supabase_client=None,
    telegram_client: Optional[TelegramClient] = None,
) -> Dict[str, Any]:
    """Execute a single turn of Iris Agent with top-level error handling."""
    if not telegram_client:
        telegram_client = TelegramClient()

    try:
        return _execute_agent_turn_inner(
            chat_id=chat_id,
            user_message=user_message,
            photo=photo,
            voice=voice,
            supabase_client=supabase_client,
            telegram_client=telegram_client,
        )
    except Exception as turn_err:
        logger.error("Unhandled exception in execute_agent_turn for chat_id=%s: %s", chat_id, turn_err, exc_info=True)
        fallback = "⚠️ Iris encountered an internal error processing your request. Please try again in a moment."
        try:
            telegram_client.send_message(chat_id, fallback)
        except Exception as tg_err:
            logger.error("Failed to send fallback error message to Telegram: %s", tg_err)
        return {"status": "error", "error": str(turn_err)}

def _execute_agent_turn_inner(

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

        telegram_client.send_message(chat_id, "⚠️ Access denied. Your Telegram ID is not authorized.")

        return {"status": "unauthorized"}

    clean = (user_message or "").strip()

    if not clean and not photo and not voice:

        greeting = "Hello! How can I help you today?"

        telegram_client.send_message(chat_id, greeting)

        return {"status": "success", "reply": greeting}

    # 2. Route to command handler first

    cmd_result = _handle_command(chat_id, clean, telegram_client)

    if cmd_result is not None:

        return cmd_result

    # ── Application-Level Confirmation Gate Check ──
    import time, uuid
    is_user_confirmed = False
    now_ts = time.time()

    if chat_id in _PENDING_ACTION_CONFIRMATIONS:
        pending = _PENDING_ACTION_CONFIRMATIONS[chat_id]
        if now_ts > pending.get("expires_at", 0):
            _PENDING_ACTION_CONFIRMATIONS.pop(chat_id, None)
            logger.info("Pending action %s for chat_id=%s expired.", pending.get("action_id"), chat_id)
            if CONFIRMATION_AFFIRMATIVE_RE.search(clean):
                telegram_client.send_message(
                    chat_id,
                    "⚠️ That pending action has expired (10-minute limit). Please request the action again."
                )
                return {"status": "expired"}
        elif CONFIRMATION_NEGATIVE_RE.search(clean):
            _PENDING_ACTION_CONFIRMATIONS.pop(chat_id, None)
            telegram_client.send_message(chat_id, "Action cancelled. No external changes were made.")
            return {"status": "cancelled"}
        elif CONFIRMATION_AFFIRMATIVE_RE.search(clean):
            is_user_confirmed = True

    # 3. Voice message → STT → text

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

            telegram_client.send_message(chat_id, "⚠️ Failed to download voice message.")

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

    # 6. Load persistent memory → build system prompt

    mem = _memory.load(chat_id)

    system_prompt = _memory.build_system_prompt(mem["memory_md"], mem["user_md"])

    # 7. Build message list for LLM

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    messages.extend(history)

    # 8. Build user content (text and/or vision)

    is_vision = False

    if photo:

        telegram_client.send_chat_action(chat_id, "typing")

        try:

            photo_bytes = _download_photo(telegram_client, photo)

        except Exception as dl_err:

            logger.error("Photo download exception for chat_id=%s: %s", chat_id, dl_err)

            photo_bytes = None

        if photo_bytes:

            is_vision = True

            user_content = [

                {"type": "text", "text": clean if clean else "Describe this image in detail."},

                _photo_to_content_part(photo_bytes),

            ]

        else:

            logger.warning("Photo download returned no bytes for chat_id=%s", chat_id)

            telegram_client.send_message(

                chat_id,

                "⚠️ I detected your image, but was unable to download it from Telegram right now. Please try sending it again."

            )

            return {"status": "error", "error": "photo download failed"}

    else:

        user_content = clean

    messages.append({"role": "user", "content": user_content})

    telegram_client.send_chat_action(chat_id, "typing")

    # ── Context Injection: Business Snapshot ──

    if _BUSINESS_PATTERNS.search(clean):

        biz_context = _biz.format_for_llm(chat_id)

        if biz_context:

            messages.insert(1, {"role": "system", "content": biz_context})

            logger.debug("Injected business snapshot context for chat_id=%s", chat_id)

    # ── Context Injection: Web Search ──

    if _WEB_SEARCH_PATTERNS.search(clean) and not is_vision:

        search_results = _web.search(clean, max_results=5)

        if search_results:

            web_ctx = _web.format_for_llm(search_results, query=clean)

            messages.insert(1, {"role": "system", "content": web_ctx})

            logger.debug(

                "Injected %d web search results for chat_id=%s",

                len(search_results), chat_id,

            )

    # 9. Dynamic Task Routing & Composio Tool Discovery

    composio = ComposioClient()

    has_tools = composio.is_configured() and bool(composio.get_connected_accounts())

    tool_schemas = []

    if has_tools:

        tool_schemas = composio.get_tool_schemas_for_request(clean)

        if tool_schemas:

            logger.info("Exposing %d Composio tool schemas to LLM for request: %s", len(tool_schemas), clean[:50])

    session_model_override = _get_session_model(chat_id)

    decision = _router.route(

        clean,

        history=history,

        has_photo=is_vision,

        tools_available=has_tools and bool(tool_schemas),

        manual_model_override=session_model_override,

        active_providers=_registry.available_providers(),

    )

    # 10. Agent Tool Execution Loop (up to 5 tool iterations)

    max_tool_iterations = 5

    iteration = 0

    reply = ""

    provider_used = "unknown"

    model_used = "unknown"

    while iteration < max_tool_iterations:

        iteration += 1

        try:

            content, tool_calls, prov, model = _registry.chat_completion(

                messages,

                candidates=decision.candidates,

                vision=is_vision,

                tools=tool_schemas if tool_schemas else None,

            )

            provider_used, model_used = prov, model

        except (RuntimeError, Exception) as e:

            logger.error("All AI providers failed for chat_id=%s: %s", chat_id, e)

            clean_user_msg = (

                "⚠️ Iris is temporarily unable to process this request because the AI service is unavailable. "

                "Please try again in a moment."

            )

            telegram_client.send_message(chat_id, clean_user_msg)

            return {"status": "error", "error": "AI service unavailable"}

        if not tool_calls:

            reply = content

            break

        assistant_tool_msg = {"role": "assistant", "content": content or None, "tool_calls": []}

        for tc in tool_calls:

            assistant_tool_msg["tool_calls"].append({

                "id": tc.id,

                "type": "function",

                "function": {

                    "name": tc.function.name,

                    "arguments": tc.function.arguments,

                }

            })

        messages.append(assistant_tool_msg)

        for tc in tool_calls:

            fn_name = tc.function.name

            try:

                fn_args = json.loads(tc.function.arguments or "{}")

            except Exception:

                fn_args = {}

            from lib.composio_client import is_consequential_action

            if is_consequential_action(fn_name) and not is_user_confirmed:

                action_id = str(uuid.uuid4())

                expires_at = now_ts + 600.0

                summary_parts = [f"Tool: {fn_name}"]

                for k, v in fn_args.items():

                    summary_parts.append(f"{k}: {v}")

                action_summary = "\n".join(summary_parts)

                logger.info("INTERCEPTED consequential tool call (action_id=%s): %s", action_id, fn_name)

                _PENDING_ACTION_CONFIRMATIONS[chat_id] = {

                    "action_id": action_id,

                    "user_id": str(chat_id),

                    "chat_id": chat_id,

                    "tool_name": fn_name,

                    "args": fn_args,

                    "action_summary": action_summary,

                    "created_at": now_ts,

                    "expires_at": expires_at,

                    "status": "PENDING_CONFIRMATION",

                }

                tool_result = {

                    "successful": False,

                    "requires_confirmation": True,

                    "action_id": action_id,

                    "error": (

                        f"Confirmation required. Present details of '{fn_name}' and its parameters ({json.dumps(fn_args)}) "

                        f"to the user and ask them to explicitly reply with 'yes' or 'confirm' before proceeding."

                    ),

                }

            else:

                if is_user_confirmed and chat_id in _PENDING_ACTION_CONFIRMATIONS:

                    pending = _PENDING_ACTION_CONFIRMATIONS.pop(chat_id, None)

                    if pending:

                        fn_name = pending["tool_name"]

                        fn_args = pending["args"]

                logger.info("Executing tool call: %s with args: %s", fn_name, fn_args)
                telegram_client.send_chat_action(chat_id, "typing")
                try:
                    tool_result = composio.execute_tool(fn_name, fn_args)
                except Exception as tool_err:
                    logger.error("Composio tool execution exception for %s: %s", fn_name, tool_err)
                    tool_result = {
                        "successful": False,
                        "error": f"Failed to execute tool {fn_name}: {tool_err}",
                    }

            logger.info("Tool execution result for %s: %s", fn_name, tool_result)

            from lib.job_runner import make_dedup_key, record_action

            dedup_k = make_dedup_key(str(chat_id), fn_name)

            is_ok = bool(tool_result.get("successful", True))

            record_action(

                dedup_k,

                chat_id=chat_id,

                job_id=None,

                action_type=fn_name,

                status="success" if is_ok else "failed",

                request_data=fn_args,

                response_data=tool_result,

                error=tool_result.get("error") if not is_ok else None,

            )

            messages.append({

                "role": "tool",

                "tool_call_id": tc.id,

                "name": fn_name,

                "content": json.dumps(tool_result),

            })

    # 11. Mid-Task Escalation Check

    if _router.should_escalate(reply, decision.tier):

        logger.info("Escalating task to POWERFUL tier for chat_id=%s", chat_id)

        powerful_candidates = [

            c for c in decision.candidates if c[2].tier == ModelTier.POWERFUL

        ]

        if powerful_candidates:

            try:

                esc_reply, esc_tc, esc_prov, esc_model = _registry.chat_completion(

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

    if not reply or not str(reply).strip():

        logger.warning("LLM returned empty reply for chat_id=%s; using default fallback response", chat_id)

        reply = "I'm here! How can I help you today?"

    telegram_client.send_message(chat_id, reply)

    # 14. Bounded memory update (fast 256 max_tokens cap)

    try:

        def _llm_for_memory(prompt: str) -> str:

            text, _, _, _ = _registry.chat_completion(

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

        tg.send_message(chat_id, f"⚠️ Image generation failed: {result.get('error')}")

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

        stt_status = "✅ Ready (Groq Whisper)" if _stt.is_configured() else "⚠️ GROQ_API_KEY missing"

        mem_status = "✅ Supabase" if _memory.is_configured() else "⚠️ in-process only"

        msg = (

            "🤖 *Iris — Personal AI Agent*\n\n"

            f"ð§  *LLM Providers:* `{avail}`\n"

            f"ð¯ *Routing:* Dynamic Task-Based (`/model auto` active)\n"

            f"ð¤ *Speech-to-Text:* {stt_status}\n"

            f"ð¾ *Memory:* {mem_status}\n\n"

            "ð *Commands:*\n"

            "• `/image <prompt>` — Generate AI image\n"

            "• `/models` — List available free LLM models\n"

            "• `/model auto` — Enable dynamic task routing\n"

            "• `/model <name>` — Override active model\n"

            "• `/memory` — View stored memory\n"

            "• `/forget` — Clear stored memory\n"

            "• `/tools` — List Composio connected tools\n"

            "• `/status` — System status & router telemetry\n"

            "• `/cron list` — List scheduled jobs\n"

            "• `/cron add <expr> <task>` — Add cron job\n"

            "• `/cron del <job_id>` — Delete cron job\n\n"

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

        mode_str = "⚡ Auto (Dynamic Task Router)" if current_override == "auto" else f"📌 Manual Override (`{current_override}`)"

        telem = _last_routing_telemetry.get(chat_id, {})

        last_tier = telem.get("tier", "N/A")

        last_model = telem.get("model", "N/A")

        last_prov = telem.get("provider", "N/A")

        last_reason = telem.get("reason", "N/A")

        escalated_str = "Yes ð" if telem.get("escalated") else "No"

        msg = (

            "ð *Iris Agent Status*\n\n"

            f"ð¯ *Routing Mode:* {mode_str}\n"

            f"⚡ *Last Tier Used:* `{last_tier}` ({last_prov} / `{last_model}`)\n"

            f"ð *Last Routing Reason:* _{last_reason}_\n"

            f"ð *Escalated Turn:* {escalated_str}\n\n"

            f"ð§  *Available Providers:* {', '.join(avail) or 'none'}\n"

            f"🎤 *Groq STT:* {'✅' if _stt.is_configured() else '❌ GROQ_API_KEY missing'}\n"

            f"💾 *Memory:* {'Loaded ✅' if has_mem else 'Empty'}\n"

            f"ð *Composio Tools:* `{tools_summary}`\n"

            f"🖼 *Image Gen:* Pollinations.ai (Flux) ✅\n"

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

        lines.append("⚡ *FAST Tiers (Simple Q&A / Quick Chat):*")

        for s in fast_specs:

            lines.append(f"  • `{s.provider}` / `{s.model_id}`")

        lines.append("\n⚖️ *BALANCED Tiers (Research / Summaries / Chat):*")

        for s in bal_specs:

            lines.append(f"  • `{s.provider}` / `{s.model_id}`")

        lines.append("\nð§  *POWERFUL Tiers (Coding / Architecture / Debugging):*")

        for s in pow_specs:

            lines.append(f"  • `{s.provider}` / `{s.model_id}`")

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

                "• `/model auto` — Enable dynamic task routing (recommended)\n"

                "• `/model <model-id>` — Pin a specific model\n"

                "  Example: `/model gemini-2.5-flash`",

                parse_mode="Markdown",

            )

        else:

            target = parts[1].strip().lower()

            if target == "auto":

                _set_session_model(chat_id, "auto")

                tg.send_message(

                    chat_id,

                    "✅ *Dynamic Task Routing Enabled* (`auto` mode).\n"

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

            tg.send_message(chat_id, "⚠️ Composio not configured. Set COMPOSIO_API_KEY.")

        else:

            accounts = composio.get_connected_accounts()

            if not accounts:

                tg.send_message(chat_id, "No Composio tools connected yet.")

            else:

                lines = ["ð *Connected Composio Tools:*\n"]

                for acc in accounts:

                    slug = acc.get("toolkit", {}).get("slug", "?")

                    status = acc.get("status", "?")

                    lines.append(f"  • `{slug}` — {status}")

                tg.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")

        return {"status": "success", "command": "tools"}

    # /cron

    if cmd == "/cron":

        if len(parts) < 2:

            tg.send_message(

                chat_id,

                "⏰ *Cron Job Manager*\n\n"

                "• `/cron list` — View scheduled jobs\n"

                "• `/cron add <expr> <task>` — Add job\n"

                "  Example: `/cron add 0 9 * * * Send me the weather forecast`\n"

                "• `/cron del <job_id>` — Remove a job",

                parse_mode="Markdown",

            )

            return {"status": "success", "command": "cron"}

        subcmd = parts[1].lower()

        if subcmd == "list":

            jobs = _list_cron_jobs(chat_id)

            if not jobs:

                tg.send_message(chat_id, "No scheduled jobs. Use `/cron add <expr> <task>` to create one.", parse_mode="Markdown")

            else:

                lines = ["⏰ *Scheduled Jobs:*\n"]

                for j in jobs:

                    status_icon = "✅" if j.get("enabled") else "⏸"

                    jid = str(j["job_id"])[:8]

                    lines.append(

                        f"{status_icon} `{jid}` — `{j['cron_expression']}`\n"

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

                    "⚠️ Usage: `/cron add <min> <hr> <day> <mon> <dow> <task>`\n"

                    "Example: `/cron add 0 9 * * * Daily morning briefing`",

                    parse_mode="Markdown",

                )

            else:

                cron_expr = " ".join(tokens[:5])

                task_desc = " ".join(tokens[5:])

                ok = _add_cron_job(chat_id, cron_expr, task_desc)

                if ok:

                    tg.send_message(chat_id, f"✅ Cron job added!\n⏰ `{cron_expr}`\n📝 _{task_desc}_", parse_mode="Markdown")

                else:

                    tg.send_message(chat_id, "⚠️ Failed to add cron job. Check that Supabase is configured.")

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

                    tg.send_message(chat_id, f"⚠️ No job found with ID starting with `{job_id}`.", parse_mode="Markdown")

                elif len(matched) > 1:

                    tg.send_message(chat_id, "⚠️ Multiple jobs match that prefix. Please be more specific.")

                else:

                    full_id = matched[0]["job_id"]

                    ok = _delete_cron_job(full_id)

                    if ok:

                        tg.send_message(chat_id, f"ð Cron job `{full_id[:8]}` deleted.", parse_mode="Markdown")

                    else:

                        tg.send_message(chat_id, "⚠️ Failed to delete cron job.")

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

                text, _, _, _ = _registry.chat_completion(

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

