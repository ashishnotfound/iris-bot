"""
tests/test_final_security_audit.py — Final Security & Session Isolation Audit Test Suite
"""

import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from lib.composio_client import ComposioClient, is_consequential_action, validate_tool_arguments
from lib.llm_provider import ProviderRegistry, LLMKeyExhaustedError
from lib.hermes_runner import execute_agent_turn, _PENDING_ACTION_CONFIRMATIONS


class MockTelegramClient:
    def __init__(self):
        self.sent = []

    def send_message(self, chat_id, text, parse_mode=None):
        self.sent.append((chat_id, text))
        return {"ok": True}

    def send_chat_action(self, chat_id, action="typing"):
        return {"ok": True}


class TestFinalSecurityAudit(unittest.TestCase):
    def setUp(self):
        _PENDING_ACTION_CONFIRMATIONS.clear()

    def test_01_tool_classification_categories(self):
        self.assertFalse(is_consequential_action("composio_gmail_fetch_emails"))
        self.assertFalse(is_consequential_action("composio_gmail_get_contacts"))
        self.assertFalse(is_consequential_action("composio_gmail_list_drafts"))
        self.assertFalse(is_consequential_action("composio_google_maps_search"))

        self.assertTrue(is_consequential_action("composio_gmail_send_email"))
        self.assertTrue(is_consequential_action("composio_gmail_delete_message"))
        self.assertTrue(is_consequential_action("composio_googlecalendar_create_event"))
        self.assertTrue(is_consequential_action("composio_instagram_post"))

        # Unknown tools must default to CONSEQUENTIAL for fail-safe security
        self.assertTrue(is_consequential_action("composio_custom_unknown_action"))

    def test_02_argument_validation(self):
        valid, err = validate_tool_arguments("composio_gmail_send_email", {})
        self.assertFalse(valid)
        self.assertIn("Recipient email address is required", err)

        valid_ok, err_ok = validate_tool_arguments("composio_gmail_send_email", {"recipient_email": "test@example.com"})
        self.assertTrue(valid_ok)

    def test_03_session_isolation_prevents_cross_user_approval(self):
        tg = MockTelegramClient()

        # User A (chat_id=1001) has a pending action
        _PENDING_ACTION_CONFIRMATIONS[1001] = {
            "action_id": "act_user_a",
            "user_id": "1001",
            "chat_id": 1001,
            "tool_name": "composio_gmail_send_email",
            "args": {"recipient_email": "userA@target.com"},
            "created_at": time.time(),
            "expires_at": time.time() + 600,
            "status": "PENDING_CONFIRMATION",
        }

        # User B (chat_id=2002) sends "yes"
        res = execute_agent_turn(
            chat_id=2002,
            user_message="yes",
            telegram_client=tg,
        )

        # User B's "yes" must NOT release User A's pending action
        self.assertIn(1001, _PENDING_ACTION_CONFIRMATIONS)
        self.assertNotIn(2002, _PENDING_ACTION_CONFIRMATIONS)
        self.assertEqual(_PENDING_ACTION_CONFIRMATIONS[1001]["action_id"], "act_user_a")

    def test_04_expired_pending_action_rejected(self):
        tg = MockTelegramClient()

        # Create an expired pending action (>10 min ago)
        _PENDING_ACTION_CONFIRMATIONS[3003] = {
            "action_id": "act_expired",
            "user_id": "3003",
            "chat_id": 3003,
            "tool_name": "composio_gmail_send_email",
            "args": {"recipient_email": "expired@target.com"},
            "created_at": time.time() - 700,
            "expires_at": time.time() - 100,  # Expired 100 seconds ago
            "status": "PENDING_CONFIRMATION",
        }

        res = execute_agent_turn(
            chat_id=3003,
            user_message="yes, send it",
            telegram_client=tg,
        )

        self.assertEqual(res.get("status"), "expired")
        self.assertNotIn(3003, _PENDING_ACTION_CONFIRMATIONS)
        # Verify user was notified of expiration
        sent_texts = [text for cid, text in tg.sent if cid == 3003]
        self.assertTrue(any("expired" in t.lower() for t in sent_texts))

    def test_05_cancellation_clears_pending_action(self):
        tg = MockTelegramClient()

        _PENDING_ACTION_CONFIRMATIONS[4004] = {
            "action_id": "act_cancel",
            "user_id": "4004",
            "chat_id": 4004,
            "tool_name": "composio_gmail_send_email",
            "args": {"recipient_email": "cancel@target.com"},
            "created_at": time.time(),
            "expires_at": time.time() + 600,
            "status": "PENDING_CONFIRMATION",
        }

        res = execute_agent_turn(
            chat_id=4004,
            user_message="no, cancel it",
            telegram_client=tg,
        )

        self.assertEqual(res.get("status"), "cancelled")
        self.assertNotIn(4004, _PENDING_ACTION_CONFIRMATIONS)
        sent_texts = [text for cid, text in tg.sent if cid == 4004]
        self.assertTrue(any("cancelled" in t.lower() for t in sent_texts))


if __name__ == "__main__":
    unittest.main()
