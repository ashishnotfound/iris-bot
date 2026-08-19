"""
lib/telegram_client.py — Telegram Bot API Client for Iris / Hermes Agent

Handles sending messages, photos/images, notifications, and status updates.
"""

import os
import json
import logging
import re
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MESSAGE_LEN = 4096

# Characters that must be escaped in Markdown v2 mode
_MDV2_ESCAPE = re.compile(r'([_*\[\]()~`>#+\-=|{}.!\\])')


def _escape_mdv2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MDV2_ESCAPE.sub(r'\\\1', text)


def _strip_markdown(text: str) -> str:
    """Very basic Markdown stripping to plain text fallback."""
    # Remove bold/italic markers, inline code, headers
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'#{1,6}\s+', '', text)
    return text.strip()


class TelegramClient:
    def __init__(self, bot_token: Optional[str] = None):
        token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        if not token:
            logger.error(
                "TelegramClient: TELEGRAM_BOT_TOKEN is not set. "
                "All API calls will fail."
            )
        self.bot_token = token
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def get_me(self) -> Dict[str, Any]:
        """Fetch bot identity details from Telegram Bot API."""
        try:
            r = requests.get(f"{self.base_url}/getMe", timeout=10)
            return r.json()
        except Exception as e:
            logger.error("Telegram getMe error: %s", e)
            return {"ok": False, "error": str(e)}

    def send_message(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a text message to a Telegram chat.

        Automatically truncates messages that exceed Telegram's 4096-char
        limit and falls back to plain text if Markdown rendering fails.
        """
        url = f"{self.base_url}/sendMessage"

        text = str(text or "").strip()
        if not text:
            text = "I'm here! How can I help you today?"

        # Truncate if necessary
        if len(text) > _MAX_MESSAGE_LEN:
            text = text[:_MAX_MESSAGE_LEN - 20] + "\n\n…[truncated]"

        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            r = requests.post(url, json=payload, timeout=15)
            data = r.json()
            if not data.get("ok"):
                logger.error(
                    "Telegram sendMessage failed (chat_id=%s): %s",
                    chat_id,
                    data.get("description", str(data)),
                )
                if parse_mode:
                    # Markdown rendering failed — retry with plain text
                    plain_text = _strip_markdown(text)
                    plain_payload = {"chat_id": chat_id, "text": plain_text}
                    r2 = requests.post(url, json=plain_payload, timeout=15)
                    data2 = r2.json()
                    if not data2.get("ok"):
                        logger.error("Telegram plain text retry failed (chat_id=%s): %s", chat_id, data2.get("description", str(data2)))
                    return data2
            return data
        except Exception as e:
            logger.error("Telegram sendMessage error: %s", e)
            return {"ok": False, "error": str(e)}

    def send_photo(
        self,
        chat_id: int | str,
        photo_url: str,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an image photo to a Telegram chat via URL."""
        url = f"{self.base_url}/sendPhoto"
        payload: Dict[str, Any] = {"chat_id": chat_id, "photo": photo_url}
        if caption:
            payload["caption"] = caption[:1024]
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            r = requests.post(url, json=payload, timeout=20)
            data = r.json()
            if not data.get("ok") and parse_mode:
                # Retry without parse_mode on Markdown failure
                payload.pop("parse_mode", None)
                r2 = requests.post(url, json=payload, timeout=20)
                return r2.json()
            return data
        except Exception as e:
            logger.error("Telegram sendPhoto error: %s", e)
            return {"ok": False, "error": str(e)}

    def send_chat_action(
        self, chat_id: int | str, action: str = "typing"
    ) -> Dict[str, Any]:
        """Send typing/upload_photo status indicator to chat."""
        url = f"{self.base_url}/sendChatAction"
        try:
            r = requests.post(
                url,
                json={"chat_id": chat_id, "action": action},
                timeout=5,
            )
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_file(self, file_id: str) -> Dict[str, Any]:
        """Get file path info from Telegram to facilitate download."""
        try:
            r = requests.get(
                f"{self.base_url}/getFile",
                params={"file_id": file_id},
                timeout=10,
            )
            return r.json()
        except Exception as e:
            logger.error("Telegram getFile error: %s", e)
            return {"ok": False, "error": str(e)}

    def download_file(self, file_path: str) -> Optional[bytes]:
        """Download file content from Telegram servers."""
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        try:
            r = requests.get(url, timeout=30)
            return r.content if r.status_code == 200 else None
        except Exception as e:
            logger.error("Telegram downloadFile error: %s", e)
            return None
