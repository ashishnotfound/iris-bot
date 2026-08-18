#!/usr/bin/env python3
"""
Hermes Cloud Telegram Agent - Cloud Bootstrapper
Configures Hermes Agent runtime and launches Telegram Gateway in cloud environments.
Includes a lightweight HTTP health check endpoint for Render / Web Service platforms.
Uses Python Standard Library only (zero third-party dependencies).
"""

import os
import sys
import json
import re
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

class HealthCheckHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler to satisfy cloud health checks and keep-alive pings."""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "status": "ok",
            "service": "Hermes Cloud Telegram Agent",
            "platform": "Telegram Bot API",
            "gateway": "active"
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence routine health check GET logs to keep log output clean
        pass

def start_health_server(port: int):
    """Start background HTTP server for health checks."""
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        print(f"[OK] Health check server active on port {port}")
        server.serve_forever()
    except Exception as e:
        print(f"[WARN] Health server error: {e}")

def setup_cloud_environment():
    print("==================================================")
    print("   [Hermes Cloud Telegram Agent Initializing]")
    print("==================================================")

    # Resolve HERMES_HOME directory
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    os.makedirs(hermes_home, exist_ok=True)
    config_path = Path(hermes_home) / "config.yaml"

    # 1. Configure Telegram Gateway
    telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    raw_allowed_users = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()

    if not telegram_bot_token:
        print("[WARNING] TELEGRAM_BOT_TOKEN is not set. Hermes Gateway requires a bot token to communicate via Telegram.")

    # Parse allowed Telegram users (numeric chat IDs)
    allowed_users = []
    if raw_allowed_users:
        for u in re.split(r"[,\s]+", raw_allowed_users.replace("[", "").replace("]", "").replace('"', '')):
            u_str = u.strip()
            if u_str.isdigit():
                allowed_users.append(int(u_str))

    if not allowed_users:
        print("[CAUTION] TELEGRAM_ALLOWED_USERS is empty! For security, set your numeric Telegram ID in TELEGRAM_ALLOWED_USERS.")

    terminal_backend = os.environ.get("TERMINAL_BACKEND", "vercel_sandbox").strip()
    default_model = os.environ.get("DEFAULT_MODEL", "gemini-2.5-flash").strip()

    # Construct YAML content directly
    yaml_lines = [
        "# Hermes Agent Cloud Configuration",
        "messaging:",
        "  telegram:",
        "    enabled: true",
        f"    bot_token: \"{telegram_bot_token}\"",
        f"    allowed_users: {json.dumps(allowed_users)}",
        "",
        "terminal:",
        f"  backend: \"{terminal_backend}\"",
        "",
        "model:",
        f"  default: \"{default_model}\"",
        ""
    ]

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))

    print(f"[OK] Configuration saved to {config_path}")
    print(f" - Telegram Bot Token: {'Configured' if telegram_bot_token else 'Missing'}")
    print(f" - Allowed Users: {allowed_users}")
    print(f" - Terminal Backend: {terminal_backend}")
    print(f" - Default Model: {default_model}")
    print("--------------------------------------------------")

def main():
    setup_cloud_environment()

    # Launch HTTP health server on background thread for cloud platform health checks ($PORT or 10000)
    port = int(os.environ.get("PORT", "10000"))
    health_thread = threading.Thread(target=start_health_server, args=(port,), daemon=True)
    health_thread.start()

    print("[INFO] Launching Hermes Gateway service...")

    # Try 'hermes gateway start' CLI binary first
    try:
        res = subprocess.run(["hermes", "gateway", "start"])
        if res.returncode == 0:
            return
    except FileNotFoundError:
        pass

    # Fallback to python module execution
    try:
        subprocess.run([sys.executable, "-m", "hermes_cli", "gateway", "start"], check=True)
    except Exception as e:
        print(f"[INFO] Bootstrapper fallback completed: {e}")

if __name__ == "__main__":
    main()
