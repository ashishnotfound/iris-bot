"""
api/webhook.py - Vercel Python Serverless HTTP Entrypoint for Telegram Webhook

Receives Telegram update POST requests, performs secret header validation,
enforces idempotency via processed_updates table, and executes agent turn.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from lib.auth import is_allowed, validate_webhook_secret
from lib.hermes_runner import (
    execute_agent_turn,
    _is_update_processed,
    _mark_update_processed,
)

logger = logging.getLogger(__name__)


def handler(request):
    """Vercel Python serverless HTTP entrypoint for Telegram webhook POST."""
    method = getattr(request, "method", "POST")
    if isinstance(method, str):
        method = method.upper()
    else:
        method = "POST"

    if method != "POST":
        return {
            "statusCode": 405,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Method not allowed"}),
        }

    # Validate secret header if set
    secret_header = None
    headers = getattr(request, "headers", {})
    if isinstance(headers, dict):
        secret_header = headers.get("X-Telegram-Bot-Api-Secret-Token") or headers.get("x-telegram-bot-api-secret-token")

    if not validate_webhook_secret(secret_header):
        return {
            "statusCode": 401,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Unauthorized webhook secret"}),
        }

    # Parse update body
    try:
        body = getattr(request, "body", b"")
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        update = json.loads(body) if body else {}
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Invalid JSON payload: {e}"}),
        }

    update_id = update.get("update_id")
    message = update.get("message") or update.get("edited_message") or {}
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text") or message.get("caption") or ""
    photo = message.get("photo")
    voice = message.get("voice")

    if not chat_id:
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "ignored", "reason": "no_chat_id"}),
        }

    if not is_allowed(chat_id):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "unauthorized"}),
        }

    # Idempotency check
    if update_id and _is_update_processed(update_id):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "skipped", "reason": "already_processed"}),
        }

    if update_id:
        _mark_update_processed(update_id, chat_id)

    # Execute turn
    try:
        result = execute_agent_turn(
            chat_id=chat_id,
            user_message=text,
            photo=photo,
            voice=voice,
        )
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }
    except Exception as e:
        logger.error("Webhook processing failed: %s", e, exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "error", "error": str(e)}),
        }
