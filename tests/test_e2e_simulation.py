"""
tests/test_e2e_simulation.py — End-to-End Simulation of AI Provider Failover and Error Handling
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from lib.llm_provider import ProviderRegistry, GeminiProvider, OpenRouterProvider, NvidiaProvider
from lib.task_router import TaskRouter, ModelTier
from lib.telegram_client import TelegramClient


class MockTelegramClient:
    def __init__(self):
        self.sent_messages = []
        self.bot_token = "mock_token"

    def send_message(self, chat_id, text, parse_mode=None):
        self.sent_messages.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode})
        return {"ok": True}

    def send_chat_action(self, chat_id, action="typing"):
        return {"ok": True}


class TestE2ESimulation(unittest.TestCase):
    def setUp(self):
        self.tg = MockTelegramClient()
        self.registry = ProviderRegistry()
        self.router = TaskRouter()

    def test_clean_error_message_on_all_failed(self):
        # Simulate all keys failing by unsetting keys
        old_g = os.environ.pop("GEMINI_API_KEY", None)
        old_o = os.environ.pop("OPENROUTER_API_KEY", None)
        old_n = os.environ.pop("NVIDIA_API_KEY", None)

        from lib.hermes_runner import execute_agent_turn
        res = execute_agent_turn(
            chat_id=12345678,
            user_message="So what's the weather in delhi",
            telegram_client=self.tg,
        )

        self.assertEqual(res["status"], "error")
        # Check that sent Telegram message is clean and user-friendly
        self.assertTrue(len(self.tg.sent_messages) > 0)
        last_msg = self.tg.sent_messages[-1]["text"]

        # MUST NOT leak raw exception traces, auth headers, or internal URL errors
        self.assertNotIn("400 - Please pass a valid API key", last_msg)
        self.assertNotIn("401 - Missing Authentication header", last_msg)
        self.assertNotIn("gemini-2.5-flash", last_msg)
        
        # MUST contain user-friendly message with clean emoji
        self.assertIn("⚠️ Iris is temporarily unable to process this request", last_msg)
        self.assertNotIn("â", last_msg)  # No mojibake double encoding!

        # Restore env
        if old_g: os.environ["GEMINI_API_KEY"] = old_g
        if old_o: os.environ["OPENROUTER_API_KEY"] = old_o
        if old_n: os.environ["NVIDIA_API_KEY"] = old_n

    def test_unicode_emojis_render_cleanly(self):
        msg = "⚠️ Warning ✅ Success 🚨 Alert 🔄 Sync"
        self.tg.send_message(12345, msg)
        last_text = self.tg.sent_messages[-1]["text"]
        self.assertEqual(last_text, msg)
        self.assertNotIn("â", last_text)


if __name__ == "__main__":
    unittest.main()
