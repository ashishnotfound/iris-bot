"""
tests/test_dm_pipeline_audit.py — Comprehensive DM Pipeline E2E Reliability & Media Audit Test Suite
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
from lib.llm_provider import ProviderRegistry
from lib.composio_client import ComposioClient
import lib.hermes_runner as hr


class MockTelegramClient:
    def __init__(self, fail_api=False):
        self.sent = []
        self.fail_api = fail_api

    def send_message(self, chat_id, text, parse_mode=None):
        self.sent.append((chat_id, text))
        if self.fail_api:
            return {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
        return {"ok": True}

    def send_chat_action(self, chat_id, action="typing"):
        return {"ok": True}

    def get_file(self, file_id):
        if self.fail_api:
            return {"ok": False, "description": "File not found"}
        return {"ok": True, "result": {"file_path": "photos/test.jpg"}}

    def download_file(self, file_path):
        if self.fail_api:
            return None
        # Return dummy JPEG magic bytes header + data
        return b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB\x00C\x00DummyImageData"


class TestDMPipelineAudit(unittest.TestCase):
    def test_01_text_only_dm_reaches_iris_and_replies(self):
        tg = MockTelegramClient()
        mock_res = ("Hello there! How can I help you today?", None, "openrouter", "google/gemma-4-31b-it:free")

        with patch.object(hr._registry, "chat_completion", return_value=mock_res):
            res = hr.execute_agent_turn(
                chat_id=123456,
                user_message="Hello",
                telegram_client=tg,
            )

        self.assertEqual(res.get("status"), "success")
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("Hello there!", tg.sent[0][1])

    def test_02_image_only_message_reaches_iris_and_uses_vision(self):
        tg = MockTelegramClient()
        mock_res = ("This is a photo of a sunset over the mountains.", None, "openrouter", "google/gemma-4-31b-it:free")

        dummy_photo = [{"file_id": "file_123", "width": 800, "height": 600}]

        with patch.object(hr._registry, "chat_completion", return_value=mock_res) as mock_cc:
            res = hr.execute_agent_turn(
                chat_id=123456,
                user_message="",
                photo=dummy_photo,
                telegram_client=tg,
            )

        self.assertEqual(res.get("status"), "success")
        first_call = mock_cc.call_args_list[0]
        self.assertTrue(first_call.kwargs.get("vision"))

        messages_sent = first_call.args[0]
        user_part = messages_sent[-1]["content"]
        self.assertIsInstance(user_part, list)
        self.assertEqual(user_part[0]["text"], "Describe this image in detail.")
        self.assertEqual(tg.sent[-1][1], "This is a photo of a sunset over the mountains.")

    def test_03_image_and_caption_reaches_iris_with_both(self):
        tg = MockTelegramClient()
        mock_res = ("The graph shows sales increasing by 45%.", None, "openrouter", "google/gemma-4-31b-it:free")

        dummy_photo = [{"file_id": "file_456", "width": 800, "height": 600}]

        with patch.object(hr._registry, "chat_completion", return_value=mock_res) as mock_cc:
            res = hr.execute_agent_turn(
                chat_id=123456,
                user_message="Analyze this chart for me",
                photo=dummy_photo,
                telegram_client=tg,
            )

        self.assertEqual(res.get("status"), "success")
        first_call = mock_cc.call_args_list[0]
        self.assertTrue(first_call.kwargs.get("vision"))

        messages_sent = first_call.args[0]
        user_part = messages_sent[-1]["content"]
        self.assertEqual(user_part[0]["text"], "Analyze this chart for me")
        self.assertIn("data:image/jpeg;base64,", user_part[1]["image_url"]["url"])

    def test_04_image_download_failure_sends_error_response(self):
        tg_failing = MockTelegramClient(fail_api=True)
        dummy_photo = [{"file_id": "bad_file_id"}]

        res = hr.execute_agent_turn(
            chat_id=123456,
            user_message="Check this",
            photo=dummy_photo,
            telegram_client=tg_failing,
        )

        self.assertEqual(res.get("status"), "error")
        # Ensure user received error message on Telegram instead of silent drop
        self.assertEqual(len(tg_failing.sent), 1)
        self.assertIn("unable to download it from Telegram", tg_failing.sent[0][1])

    def test_05_unsupported_or_empty_update_returns_conversational_greeting(self):
        tg = MockTelegramClient()
        res = hr.execute_agent_turn(
            chat_id=123456,
            user_message="",
            photo=None,
            voice=None,
            telegram_client=tg,
        )

        self.assertEqual(res.get("status"), "success")
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("Hello! How can I help you today?", tg.sent[0][1])

    def test_06_ai_provider_failure_sends_fallback_message(self):
        tg = MockTelegramClient()

        with patch.object(hr._registry, "chat_completion", side_effect=RuntimeError("All AI providers failed")):
            res = hr.execute_agent_turn(
                chat_id=123456,
                user_message="Hello",
                telegram_client=tg,
            )

        self.assertEqual(res.get("status"), "error")
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("temporarily unable to process this request", tg.sent[0][1])

    def test_07_telegram_send_failure_is_detected_and_handled(self):
        tg_failing = MockTelegramClient(fail_api=True)
        mock_res = ("Test reply", None, "openrouter", "model")

        with patch.object(hr._registry, "chat_completion", return_value=mock_res):
            res = hr.execute_agent_turn(
                chat_id=123456,
                user_message="Hello",
                telegram_client=tg_failing,
            )

        self.assertEqual(res.get("status"), "success")


if __name__ == "__main__":
    unittest.main()
