"""
lib/auth.py — Telegram authentication and allowlist enforcement.

Validates:
  1. Telegram webhook secret token header
  2. Sender chat_id against TELEGRAM_ALLOWED_USERS allowlist
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
    if not raw:
        return frozenset()
    cleaned = raw.replace("[", "").replace("]", "").replace('"', "")
    ids: list[int] = []
    for part in re.split(r"[,\s]+", cleaned):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return frozenset(ids)

def is_allowed(chat_id: int | str) -> bool:
    """Return True if the given chat_id is in the allowlist."""
    allowed_users = parse_allowed_users()
    if not allowed_users:
        logger.warning("TELEGRAM_ALLOWED_USERS is not set — denying all requests.")
        return False
    try:
        return int(chat_id) in allowed_users
    except (TypeError, ValueError):
        return False

def validate_webhook_secret(header_value: Optional[str]) -> bool:
    """Verify the X-Telegram-Bot-Api-Secret-Token header."""
    expected = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not expected:
        logger.warning("TELEGRAM_WEBHOOK_SECRET not set — skipping webhook secret validation.")
        return True
    if not header_value:
        return False
    return hmac.compare_digest(
        header_value.encode("utf-8"),
        expected.encode("utf-8"),
    )
