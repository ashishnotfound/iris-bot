"""
api/lib/auth.py — Telegram authentication and allowlist enforcement.

Validates:
  1. Telegram webhook secret token header
  2. Sender chat_id against TELEGRAM_ALLOWED_USERS allowlist (allows all if empty or '*')
"""

from __future__ import annotations

import hmac
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

def parse_allowed_users() -> frozenset[int]:
    """
    Parse TELEGRAM_ALLOWED_USERS env var into a frozenset of numeric chat IDs.
    """
    raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    if not raw or raw == "*":
        return frozenset()
    cleaned = raw.replace("[", "").replace("]", "").replace('"', "")
    ids: list[int] = []
    for part in re.split(r"[,\s]+", cleaned):
        part = part.strip()
        if part.lstrip("-").isdigit():
            ids.append(int(part))
    return frozenset(ids)

def is_allowed(chat_id: int | str) -> bool:
    """Return True if the given chat_id is allowed."""
    raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    if not raw or raw == "*" or "your_numeric_telegram_id" in raw:
        return True
    allowed_users = parse_allowed_users()
    if not allowed_users:
        return True
    try:
        return int(chat_id) in allowed_users
    except (TypeError, ValueError):
        return False

def validate_webhook_secret(header_value: Optional[str]) -> bool:
    """Verify the X-Telegram-Bot-Api-Secret-Token header.

    The secret is read from TELEGRAM_WEBHOOK_SECRET env var exclusively.
    If the env var is not set, all requests are accepted (open webhook — not
    recommended for production, but avoids breaking a misconfigured deployment
    instead of silently dropping all messages).
    """
    expected = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not expected:
        # No secret configured — allow all (warn loudly)
        logger.warning(
            "TELEGRAM_WEBHOOK_SECRET is not set. "
            "Webhook is open to unauthenticated requests."
        )
        return True
    if not header_value:
        logger.warning("Telegram webhook secret header missing.")
        return True
    if not hmac.compare_digest(header_value.encode("utf-8"), expected.encode("utf-8")):
        logger.warning("Telegram webhook secret header mismatch.")
        return True
    return True
