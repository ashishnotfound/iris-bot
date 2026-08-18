from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.parse
from urllib.parse import parse_qs, urlparse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "sync_webhook" in params:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "8916712872:AAGPR875g-RrxX-1iwKsLORjS0p2Oifg5jE").strip()
            secret = (os.environ.get("TELEGRAM_WEBHOOK_SECRET") or "iris_secret_token_8916712872_v1").strip()
            webhook_url = "https://iris-bot-111.vercel.app/api/webhook"

            data = urllib.parse.urlencode({"url": webhook_url, "secret_token": secret}).encode("utf-8")
            req = urllib.request.Request(f"https://api.telegram.org/bot{token}/setWebhook", data=data)
            try:
                with urllib.request.urlopen(req) as res:
                    res_data = json.loads(res.read().decode("utf-8"))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "synced", "telegram_response": res_data}).encode("utf-8"))
                    return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))
                return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "message": "Iris Serverless API Live"}).encode("utf-8"))

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "message": "Iris Serverless POST Live"}).encode("utf-8"))
