"""
api/webhook.py - Vercel Python Serverless HTTP Entrypoint for Telegram Webhook
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Ensure api directory and workspace root are in sys.path
api_dir = Path(__file__).resolve().parent
workspace_dir = api_dir.parent

for p in [str(api_dir), str(workspace_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.auth import is_allowed
from lib.hermes_runner import (
    execute_agent_turn,
    _is_update_processed,
    _mark_update_processed,
)

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
        except Exception as e:
            self._send_json(400, {"error": f"Failed to read body: {e}"})
            return

        try:
            update = json.loads(body) if body else {}
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
            return

        update_id = update.get("update_id")
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        text = message.get("text") or message.get("caption") or ""
        photo = message.get("photo")
        voice = message.get("voice")

        if not chat_id:
            self._send_json(200, {"status": "ignored", "reason": "no_chat_id"})
            return

        if not is_allowed(chat_id):
            self._send_json(200, {"status": "unauthorized"})
            return

        if update_id and _is_update_processed(update_id):
            self._send_json(200, {"status": "skipped", "reason": "already_processed"})
            return

        if update_id:
            _mark_update_processed(update_id, chat_id)

        try:
            result = execute_agent_turn(
                chat_id=chat_id,
                user_message=text,
                photo=photo,
                voice=voice,
            )
            self._send_json(200, result)
        except Exception as e:
            logger.error("Webhook processing failed: %s", e, exc_info=True)
            self._send_json(500, {"status": "error", "error": str(e)})

    def do_GET(self):
        self._send_json(200, {"status": "ok", "service": "Iris Telegram Webhook"})

    def _send_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
