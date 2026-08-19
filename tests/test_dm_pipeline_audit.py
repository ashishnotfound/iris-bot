"""
tests/test_dm_pipeline_audit.py — DM Pipeline E2E Reliability & Failover Test Suite
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from lib.telegram_client import TelegramClient
from lib.llm_provider import ProviderRegistry, LLMKeyExhaustedError
from lib.composio_client import ComposioClient
from lib.hermes_runner import execute_agent_turn


class MockTelegramClient:
    def __init__(self, fail_api=False):
        self.sent = []
        self.fail_api = fail_api

    def send_message(self, chat_id, text, parse_mode=None):
        if self.fail_api:
            return {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
        self.sent.append((chat_id, text))
        return {"ok": True}

    def send_chat_action(self, chat_id, action="typing"):
        return {"ok": True}


class TestDMPipelineAudit(unittest.TestCase):
    def test_01_basic_text_dm_flow(self):
        reg = ProviderRegistry()
        tg = MockTelegramClient()

        mock_res = ("Hello there! How can I help you today?", None, "openrouter", "google/gemma-4-31b-it:free")

        with patch.object(reg, "chat_completion", return_value=mock_res):
            with patch("lib.hermes_runner._registry", reg):
                res = execute_agent_turn(
                    chat_id=123456,
                    user_message="Hello",
                    telegram_client=tg,
                )

        self.assertEqual(res.get("status"), "success")
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("Hello there!", tg.sent[0][1])

    def test_02_image_free_message_does_not_enter_vision_path(self):
        reg = ProviderRegistry()
        tg = MockTelegramClient()

        mock_res = ("I am a text AI assistant.", None, "openrouter", "google/gemma-4-31b-it:free")

        with patch.object(reg, "chat_completion", return_value=mock_res) as mock_cc:
            with patch("lib.hermes_runner._registry", reg):
                res = execute_agent_turn(
                    chat_id=123456,
                    user_message="What is 2+2?",
                    photo=None,
                    telegram_client=tg,
                )

        self.assertEqual(res.get("status"), "success")
        # Verify vision=False was passed to chat_completion
        kwargs = mock_cc.call_args[1]
        self.assertFalse(kwargs.get("vision"))
        self.assertEqual(tg.sent[0][1], "I am a text AI assistant.")

    def test_03_ai_provider_failure_sends_fallback_message(self):
        reg = ProviderRegistry()
        tg = MockTelegramClient()

        with patch.object(reg, "chat_completion", side_effect=RuntimeError("All AI providers failed")):
            with patch("lib.hermes_runner._registry", reg):
                res = execute_agent_turn(
                    chat_id=123456,
                    user_message="Hello",
                    telegram_client=tg,
                )

        self.assertEqual(res.get("status"), "error")
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("temporarily unable to process this request", tg.sent[0][1])

    def test_04_telegram_api_failure_does_not_crash_handler(self):
        reg = ProviderRegistry()
        tg_failing = MockTelegramClient(fail_api=True)

        mock_res = ("Test reply", None, "openrouter", "model")

        with patch.object(reg, "chat_completion", return_value=mock_res):
            with patch("lib.hermes_runner._registry", reg):
                res = execute_agent_turn(
                    chat_id=123456,
                    user_message="Hello",
                    telegram_client=tg_failing,
                )

        self.assertEqual(res.get("status"), "success")

    def test_05_missing_optional_composio_integration_works(self):
        reg = ProviderRegistry()
        tg = MockTelegramClient()

        mock_res = ("Hello! I can still talk even without Composio.", None, "openrouter", "model")

        with patch.object(ComposioClient, "is_configured", return_value=False):
            with patch.object(reg, "chat_completion", return_value=mock_res):
                with patch("lib.hermes_runner._registry", reg):
                    res = execute_agent_turn(
                        chat_id=123456,
                        user_message="Hello",
                        telegram_client=tg,
                    )

        self.assertEqual(res.get("status"), "success")
        self.assertIn("Hello! I can still talk", tg.sent[0][1])

    def test_06_unhandled_exception_sends_fallback_to_user(self):
        tg = MockTelegramClient()

        # Simulate unexpected exception during prompt construction
        with patch("lib.hermes_runner._handle_command", side_effect=Exception("Database crash")):
            res = execute_agent_turn(
                chat_id=123456,
                user_message="Hello",
                telegram_client=tg,
            )

        self.assertEqual(res.get("status"), "error")
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("Iris encountered an internal error", tg.sent[0][1])


if __name__ == "__main__":
    unittest.main()
