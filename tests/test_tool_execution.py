"""
tests/test_tool_execution.py — Test Suite for Tool Execution, Confirmation Safety Gate & Action State Tracking
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from lib.composio_client import ComposioClient, is_consequential_action
from lib.llm_provider import ProviderRegistry
from lib.hermes_runner import execute_agent_turn, _PENDING_ACTION_CONFIRMATIONS


class MockToolCallFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = MockToolCallFunction(name, arguments)


class MockTelegramClient:
    def __init__(self):
        self.sent = []

    def send_message(self, chat_id, text, parse_mode=None):
        self.sent.append(text)
        return {"ok": True}

    def send_chat_action(self, chat_id, action="typing"):
        return {"ok": True}


class TestToolExecutionPipeline(unittest.TestCase):
    def setUp(self):
        self.composio = ComposioClient()
        _PENDING_ACTION_CONFIRMATIONS.clear()

    def test_01_tool_schema_discovery(self):
        schemas = self.composio.get_tool_schemas_for_request(
            "Send an email to reyostore9@gmail.com asking how are you"
        )
        self.assertGreater(len(schemas), 0)
        names = [s["function"]["name"] for s in schemas]
        self.assertTrue(
            any("gmail" in name for name in names),
            f"Gmail tool schema should be exposed for email requests. Names: {names}",
        )

    def test_02_consequential_action_intercepted_without_user_confirmation(self):
        reg = ProviderRegistry()
        tg = MockTelegramClient()

        tc = MockToolCall(
            "call_123",
            "composio_gmail_send_email",
            json.dumps({
                "recipient_email": "reyostore9@gmail.com",
                "subject": "Hello from Iris!",
                "body": "How are you doing today?",
            }),
        )

        turn1_res = ("", [tc], "openrouter", "nvidia/nemotron-3.5-lightning:free")
        turn2_res = (
            "I've prepared the email to reyostore9@gmail.com. Please reply with 'yes' or 'confirm' to send it.",
            None,
            "openrouter",
            "nvidia/nemotron-3.5-lightning:free",
        )

        with patch.object(reg, "chat_completion", side_effect=[turn1_res, turn2_res]):
            with patch.object(ComposioClient, "execute_tool") as mock_exec:
                with patch("lib.hermes_runner._registry", reg):
                    res = execute_agent_turn(
                        chat_id=987654321,
                        user_message="Send an email to reyostore9@gmail.com asking how are you",
                        telegram_client=tg,
                    )

        self.assertEqual(res.get("status"), "success")
        self.assertIn("Please reply with 'yes' or 'confirm'", tg.sent[-1])
        # Verify Composio execute_tool was NOT called (intercepted by code gate)
        mock_exec.assert_not_called()
        self.assertIn(987654321, _PENDING_ACTION_CONFIRMATIONS)

    def test_03_confirmed_action_executes_successfully(self):
        reg = ProviderRegistry()
        tg = MockTelegramClient()

        tc = MockToolCall(
            "call_123",
            "composio_gmail_send_email",
            json.dumps({
                "recipient_email": "reyostore9@gmail.com",
                "subject": "Hello from Iris!",
                "body": "How are you doing today?",
            }),
        )

        import time
        now_ts = time.time()
        # Set up pending confirmation state
        _PENDING_ACTION_CONFIRMATIONS[987654321] = {
            "action_id": "act_test_123",
            "user_id": "987654321",
            "chat_id": 987654321,
            "tool_name": "composio_gmail_send_email",
            "args": {"recipient_email": "reyostore9@gmail.com"},
            "created_at": now_ts,
            "expires_at": now_ts + 600.0,
            "status": "PENDING_CONFIRMATION",
        }

        turn1_res = ("", [tc], "openrouter", "nvidia/nemotron-3.5-lightning:free")
        turn2_res = (
            "Sent the email to reyostore9@gmail.com successfully. ✅",
            None,
            "openrouter",
            "nvidia/nemotron-3.5-lightning:free",
        )

        with patch.object(reg, "chat_completion", side_effect=[turn1_res, turn2_res]):
            with patch.object(
                ComposioClient,
                "execute_tool",
                return_value={"successful": True, "data": {"message_id": "msg_999"}},
            ) as mock_exec:
                with patch("lib.hermes_runner._registry", reg):
                    res = execute_agent_turn(
                        chat_id=987654321,
                        user_message="yes, send it",
                        telegram_client=tg,
                    )

        self.assertEqual(res.get("status"), "success")
        self.assertIn("Sent the email to reyostore9@gmail.com successfully", tg.sent[-1])
        # Verify Composio execute_tool WAS called after user confirmation
        mock_exec.assert_called_once()
        self.assertNotIn(987654321, _PENDING_ACTION_CONFIRMATIONS)

    def test_04_cancelled_action_prevents_execution(self):
        tg = MockTelegramClient()
        import time
        now_ts = time.time()
        _PENDING_ACTION_CONFIRMATIONS[987654321] = {
            "action_id": "act_test_cancel",
            "user_id": "987654321",
            "chat_id": 987654321,
            "tool_name": "composio_gmail_send_email",
            "args": {"recipient_email": "reyostore9@gmail.com"},
            "created_at": now_ts,
            "expires_at": now_ts + 600.0,
            "status": "PENDING_CONFIRMATION",
        }

        res = execute_agent_turn(
            chat_id=987654321,
            user_message="no, cancel it",
            telegram_client=tg,
        )

        self.assertEqual(res.get("status"), "cancelled")
        self.assertIn("Action cancelled. No external changes were made", tg.sent[-1])
        self.assertNotIn(987654321, _PENDING_ACTION_CONFIRMATIONS)


if __name__ == "__main__":
    unittest.main()
