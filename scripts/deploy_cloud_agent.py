#!/usr/bin/env python3
"""
Hermes Cloud Telegram Agent - Deployment & Verification Utility
Validates local configuration, checks Telegram security settings, and prepares cloud release.
Uses Python Standard Library only.
"""

import os
import sys
import json
import re

def validate_telegram_token(token: str) -> bool:
    """Validate Telegram Bot token format (e.g. 123456789:ABCdefGhI...)."""
    if not token:
        return False
    pattern = r"^\d+:[A-Za-z0-9_-]{35,}$"
    return bool(re.match(pattern, token))

def validate_allowed_users(user_input: str) -> list:
    """Parse and validate numeric Telegram user IDs."""
    user_ids = []
    if not user_input:
        return user_ids
    for u in re.split(r"[,\s]+", user_input.replace("[", "").replace("]", "").replace('"', '')):
        u_str = u.strip()
        if u_str.isdigit():
            user_ids.append(int(u_str))
    return user_ids

def run_preflight_checks():
    print("==================================================")
    print("   [Hermes Cloud Telegram Agent Preflight]")
    print("==================================================")

    env_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    env_users = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    env_backend = os.environ.get("TERMINAL_BACKEND", "vercel_sandbox").strip()
    env_provider = os.environ.get("DEFAULT_MODEL", "gemini-2.5-flash").strip()

    print(f"1. Checking Telegram Bot Token:")
    if validate_telegram_token(env_token):
        print(f"   [PASS] Valid Bot Token structure detected ({env_token[:8]}...)")
    else:
        print(f"   [WARN] TELEGRAM_BOT_TOKEN is missing or not set to a valid format.")

    print(f"\n2. Checking Security Allowlist (TELEGRAM_ALLOWED_USERS):")
    parsed_ids = validate_allowed_users(env_users)
    if parsed_ids:
        print(f"   [PASS] Authorized User IDs: {parsed_ids}")
    else:
        print(f"   [WARN] TELEGRAM_ALLOWED_USERS is empty or invalid. Public access must be blocked!")

    print(f"\n3. Execution Backend & Model Configuration:")
    print(f"   [INFO] Terminal Backend: {env_backend}")
    print(f"   [INFO] Default Model:    {env_provider}")

    print("\n--------------------------------------------------")
    print("Genuinely Free Cloud Hosting Options ($0, No Credit Card Required):")
    print(" 1. Render Free Web Service (Docker):")
    print("    - Sign up at https://dashboard.render.com ($0, NO Credit Card Required)")
    print("    - Deploy with Dockerfile cloud/Dockerfile.render")
    print("    - Add TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, and GEMINI_API_KEY in Environment Settings.")
    print("    - Configure free UptimeRobot (10-min ping) to keep service running 24/7.")
    print(" 2. Vercel Dashboard:")
    print("    - Import apps/dashboard into Vercel for live status UI.")
    print("==================================================")

if __name__ == "__main__":
    run_preflight_checks()
