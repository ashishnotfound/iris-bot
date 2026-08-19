"""
api/cron.py - Vercel Python Serverless HTTP Entrypoint for Scheduled Cron Trigger
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

if str(workspace_dir) not in sys.path:
    sys.path.append(str(workspace_dir))
if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

from lib.cron_manager import run_due_jobs

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        auth_header = self.headers.get("Authorization") or self.headers.get("authorization") or ""
        expected_secret = os.environ.get("CRON_SECRET", "").strip()

        if expected_secret:
            expected_bearer = f"Bearer {expected_secret}"
            if auth_header != expected_bearer:
                self._send_json(401, {"error": "Unauthorized cron request"})
                return

        try:
            results = run_due_jobs()
            ran_list = results.get("ran", [])
            self._send_json(200, {
                "status": "success",
                "jobs_run": len(ran_list),
                "details": results,
            })
        except Exception as e:
            logger.error("Cron execution failed: %s", e, exc_info=True)
            self._send_json(500, {"status": "error", "error": str(e)})

    def _send_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
