"""
api/index.py - Vercel Python Serverless HTTP Entrypoint — Status & Utility Endpoint

GET /api/index              → health check JSON
GET /api/index?sync_webhook → re-registers the Telegram webhook with the configured secret
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure api directory is on sys.path for lib imports
api_dir = Path(__file__).resolve().parent
workspace_dir = api_dir.parent
if str(workspace_dir) not in sys.path:
    sys.path.append(str(workspace_dir))
if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

logger = logging.getLogger(__name__)

_WEBHOOK_URL = "https://iris-bot-111.vercel.app/api/webhook"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "sync_webhook" in params:
            self._sync_webhook()
            return

        self._send_json(200, {"status": "ok", "message": "Iris Serverless API Live"})

    def do_POST(self):
        self._send_json(200, {"status": "ok", "message": "Iris Serverless POST Live"})

    # ------------------------------------------------------------------

    def _sync_webhook(self) -> None:
        """Re-register the Telegram webhook with the configured secret."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()

        if not token:
            self._send_json(500, {"status": "error", "error": "TELEGRAM_BOT_TOKEN not configured"})
            return

        payload_parts: dict = {"url": _WEBHOOK_URL}
        if secret:
            payload_parts["secret_token"] = secret

        data = urllib.parse.urlencode(payload_parts).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/setWebhook",
            data=data,
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as res:
                res_data = json.loads(res.read().decode("utf-8"))
                self._send_json(200, {"status": "synced", "telegram_response": res_data})
        except Exception as e:
            logger.error("sync_webhook failed: %s", e)
            self._send_json(500, {"status": "error", "error": str(e)})

    def _send_json(self, status_code: int, data: dict) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
