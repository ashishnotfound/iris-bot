"""
lib/telegram_client.py — Telegram Bot API Client for Iris / Hermes Agent

Handles sending messages, photos/images, notifications, and status updates for @IrisOpusBot.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TelegramClient:
    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "8916712872:AAGPR875g-RrxX-1iwKsLORjS0p2Oifg5jE")).strip()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def get_me(self) -> Dict[str, Any]:
        """Fetch bot identity details from Telegram Bot API."""
        try:
            r = requests.get(f"{self.base_url}/getMe", timeout=10)
            return r.json()
        except Exception as e:
            logger.error(f"Telegram getMe error: {e}")
            return {"ok": False, "error": str(e)}

    def send_message(self, chat_id: int | str, text: str, parse_mode: Optional[str] = None) -> Dict[str, Any]:
        """Send a text message to a Telegram chat."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            r = requests.post(url, json=payload, timeout=10)
            data = r.json()
            if not data.get("ok") and parse_mode:
                # If parse_mode markdown fails, fallback to plain text
                payload.pop("parse_mode", None)
                r = requests.post(url, json=payload, timeout=10)
                data = r.json()
            return data
        except Exception as e:
            logger.error(f"Telegram sendMessage error: {e}")
            return {"ok": False, "error": str(e)}

    def send_photo(self, chat_id: int | str, photo_url: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """Send an image photo to a Telegram chat via URL."""
        url = f"{self.base_url}/sendPhoto"
        payload = {
            "chat_id": chat_id,
            "photo": photo_url
        }
        if caption:
            payload["caption"] = caption[:1024]
        try:
            r = requests.post(url, json=payload, timeout=15)
            data = r.json()
            return data
        except Exception as e:
            logger.error(f"Telegram sendPhoto error: {e}")
            return {"ok": False, "error": str(e)}

    def send_chat_action(self, chat_id: int | str, action: str = "typing") -> Dict[str, Any]:
        """Send typing/upload_photo status indicator to chat."""
        url = f"{self.base_url}/sendChatAction"
        try:
            r = requests.post(url, json={"chat_id": chat_id, "action": action}, timeout=5)
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_file(self, file_id: str) -> Dict[str, Any]:
        """Get file path info from Telegram to facilitate download."""
        try:
            r = requests.get(f"{self.base_url}/getFile", params={"file_id": file_id}, timeout=10)
            return r.json()
        except Exception as e:
            logger.error(f"Telegram getFile error: {e}")
            return {"ok": False, "error": str(e)}

    def download_file(self, file_path: str) -> Optional[bytes]:
        """Download file content from Telegram servers."""
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        try:
            r = requests.get(url, timeout=30)
            return r.content if r.status_code == 200 else None
        except Exception as e:
            logger.error(f"Telegram downloadFile error: {e}")
            return None
