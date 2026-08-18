from http.server import BaseHTTPRequestHandler
import json

import os

import os

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        au = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
        sec = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "allowed_users_raw": repr(au),
            "webhook_secret_set": bool(sec),
            "webhook_secret_len": len(sec),
        }).encode("utf-8"))

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "message": "Iris Serverless POST Live"}).encode("utf-8"))
