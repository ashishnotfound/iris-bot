"""
lib/memory_manager.py — Persistent Memory via Supabase

Each Telegram chat has two memory blobs stored in the Supabase `memory` table:

  memory_md  — MEMORY.md: agent's long-term memory about tasks, projects, preferences.
  user_md    — USER.md: profile of the user (name, timezone, communication style, etc.)

The agent:
  1. Loads memory at the start of each turn → injects into system prompt.
  2. After responding, extracts new facts and updates memory via the LLM.

Requires:
  SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) in environment.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Memory extraction prompt — sent to the LLM to extract new knowledge from turns
MEMORY_EXTRACT_PROMPT = """You are a memory curator for an AI agent named Iris.

Given the conversation below and the existing memory, extract any NEW facts worth remembering:
- User preferences, name, location, occupation, communication style
- Ongoing projects, tasks, deadlines
- Tools, APIs, or services the user frequently mentions
- Explicit instructions (e.g. "always reply in English")
- Important context not captured yet

Rules:
- Keep the updated MEMORY.md under 1500 words. Merge/summarize if needed.
- Keep the updated USER.md under 500 words.
- Return ONLY valid JSON with keys "memory_md" and "user_md".
- Do NOT include markdown code fences.
- If nothing new was learned, return the existing blobs unchanged.

Existing MEMORY.md:
{memory_md}

Existing USER.md:
{user_md}

Recent conversation:
{conversation}

Return JSON:
"""


class MemoryManager:
    """Persistent per-chat memory backed by Supabase.

    Falls back gracefully when Supabase is not configured — memory is still
    held in-process for the duration of the request, but not persisted.
    """

    def __init__(self) -> None:
        self._url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self._key = (
            os.environ.get("SUPABASE_SERVICE_KEY", "")
            or os.environ.get("SUPABASE_KEY", "")
        ).strip()
        self._available = bool(self._url and self._key)
        if not self._available:
            logger.warning(
                "MemoryManager: SUPABASE_URL or SUPABASE_SERVICE_KEY not set. "
                "Memory will not be persisted across requests."
            )

    def is_configured(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Core load / save
    # ------------------------------------------------------------------

    def load(self, chat_id: int) -> Dict[str, str]:
        """Load MEMORY.md and USER.md for a chat.

        Returns:
            {"memory_md": str, "user_md": str}
            Both are empty strings when the chat has no stored memory yet.
        """
        empty = {"memory_md": "", "user_md": ""}
        if not self._available:
            return empty
        try:
            import requests
            r = requests.get(
                f"{self._url}/rest/v1/memory",
                headers=self._headers(),
                params={"chat_id": f"eq.{chat_id}", "select": "memory_md,user_md"},
                timeout=8,
            )
            if r.status_code == 200:
                rows = r.json()
                if rows:
                    row = rows[0]
                    return {
                        "memory_md": row.get("memory_md", ""),
                        "user_md": row.get("user_md", ""),
                    }
        except Exception as e:
            logger.error("MemoryManager.load error: %s", e)
        return empty

    def save(self, chat_id: int, memory_md: str, user_md: str) -> bool:
        """Upsert MEMORY.md and USER.md for a chat.

        Returns True on success, False on failure.
        """
        if not self._available:
            return False
        try:
            import requests
            payload = {
                "chat_id": chat_id,
                "memory_md": memory_md,
                "user_md": user_md,
            }
            r = requests.post(
                f"{self._url}/rest/v1/memory",
                headers={**self._headers(), "Prefer": "resolution=merge-duplicates"},
                json=payload,
                timeout=8,
            )
            ok = r.status_code in (200, 201, 204)
            if not ok:
                logger.error("MemoryManager.save HTTP %d: %s", r.status_code, r.text[:256])
            return ok
        except Exception as e:
            logger.error("MemoryManager.save error: %s", e)
            return False

    def clear(self, chat_id: int) -> bool:
        """Delete all memory for a chat."""
        return self.save(chat_id, "", "")

    # ------------------------------------------------------------------
    # LLM-driven memory extraction
    # ------------------------------------------------------------------

    def extract_and_update(
        self,
        chat_id: int,
        messages: List[Dict[str, Any]],
        llm_callable,
    ) -> Tuple[str, str]:
        """Extract new facts from recent messages and update stored memory.

        Args:
            chat_id:      Telegram chat ID.
            messages:     Recent conversation turns (OpenAI format).
            llm_callable: A callable that accepts a single string prompt and
                          returns a string response. Typically a wrapper
                          around ProviderRegistry.chat_completion.

        Returns:
            (updated_memory_md, updated_user_md)
        """
        current = self.load(chat_id)
        memory_md = current["memory_md"]
        user_md = current["user_md"]

        # Only use last N turns to avoid huge prompts
        recent = messages[-10:] if len(messages) > 10 else messages
        conversation_text = _format_conversation(recent)

        prompt = MEMORY_EXTRACT_PROMPT.format(
            memory_md=memory_md or "(empty)",
            user_md=user_md or "(empty)",
            conversation=conversation_text,
        )

        try:
            raw = llm_callable(prompt)
            import json
            # Strip any accidental markdown code fences
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            data = json.loads(raw)
            new_memory = data.get("memory_md", memory_md)
            new_user = data.get("user_md", user_md)
        except Exception as e:
            logger.warning("Memory extraction failed (non-critical): %s", e)
            return memory_md, user_md

        self.save(chat_id, new_memory, new_user)
        return new_memory, new_user

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        memory_md: str,
        user_md: str,
        *,
        base_prompt: Optional[str] = None,
    ) -> str:
        """Construct the system prompt with memory injected.

        Args:
            memory_md:    MEMORY.md content (long-term agent memory).
            user_md:      USER.md content (user profile).
            base_prompt:  Optional custom base instruction. Defaults to
                          a sensible Iris persona prompt.

        Returns:
            Full system prompt string to pass as the first message.
        """
        base = base_prompt or (
            "You are Iris, a powerful personal AI agent. "
            "You are direct, helpful, and capable. "
            "You can reason deeply, use tools, and complete complex multi-step tasks. "
            "You speak to the user conversationally but get straight to the point. "
            f"Current date/time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."
        )

        sections: List[str] = [base]

        if memory_md and memory_md.strip():
            sections.append(f"\n## Your Memory\n{memory_md.strip()}")

        if user_md and user_md.strip():
            sections.append(f"\n## About the User\n{user_md.strip()}")

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format OpenAI-format messages as readable text for the extraction prompt."""
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multipart content (vision messages)
            text_parts = [
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = " ".join(text_parts)
        if content and role in ("User", "Assistant"):
            lines.append(f"{role}: {str(content)[:500]}")
    return "\n".join(lines)
