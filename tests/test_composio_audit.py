"""
tests/test_composio_audit.py — E2E & Unit Test Suite for Composio Tool Discovery and Execution Pipeline
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.composio_client import ComposioClient
from lib.task_router import TaskRouter, TaskClassifier, ModelTier
import lib.hermes_runner as hr


class MockTelegramClient:
    def __init__(self):
        self.sent = []

    def send_message(self, chat_id, text, parse_mode=None):
        self.sent.append((chat_id, text))
        return {"ok": True}

    def send_chat_action(self, chat_id, action="typing"):
        return {"ok": True}


class TestComposioAudit(unittest.TestCase):
    def setUp(self):
        self.client = ComposioClient()

    def test_01_account_slug_extraction_and_discovery(self):
        """Verify accounts with various JSON key formats are extracted correctly."""
        mock_accounts = [
            {"id": "acc_1", "status": "ACTIVE", "app": {"slug": "instagram"}},
            {"id": "acc_2", "status": "ACTIVE", "toolkit": {"slug": "googlecalendar"}},
            {"id": "acc_3", "status": "ACTIVE", "appName": "gmail"},
            {"id": "acc_4", "status": "ACTIVE", "appUniqueId": "browserbase_tool"},
        ]

        with patch.object(self.client, "get_connected_accounts", return_value=mock_accounts):
            with patch.object(self.client, "get_tools_for_toolkit", return_value=[{"slug": "test_action"}]):
                # 1. Instagram
                schemas = self.client.get_tool_schemas_for_request("how many followers am I having on insta")
                self.assertTrue(len(schemas) > 0)
                self.assertIn("composio_test_action", schemas[0]["function"]["name"])

                # 2. Gmail
                schemas_gmail = self.client.get_tool_schemas_for_request("Do I have any new emails?")
                self.assertTrue(len(schemas_gmail) > 0)

                # 3. Calendar
                schemas_cal = self.client.get_tool_schemas_for_request("What meetings do I have today?")
                self.assertTrue(len(schemas_cal) > 0)

                # 4. Browsebase
                schemas_browse = self.client.get_tool_schemas_for_request("Search the web for news")
                self.assertTrue(len(schemas_browse) > 0)

    def test_02_instagram_followers_pipeline_executes_tool_without_asking_for_id(self):
        """Verify full agent turn executes composio_instagram_get_user_info with {} and produces response."""
        tg = MockTelegramClient()

        # Step 1: LLM returns function call to composio_instagram_get_user_info with {}
        mock_tool_call = MagicMock()
        mock_tool_call.id = "tc_insta_123"
        mock_tool_call.function.name = "composio_instagram_get_user_info"
        mock_tool_call.function.arguments = "{}"

        # Step 2: LLM second turn returns final summary using tool result
        step1_res = ("", [mock_tool_call], "openrouter", "google/gemma-4-31b-it:free")
        step2_res = ("Your Instagram account (@totalsolutions.socials) currently has 0 followers.", None, "openrouter", "google/gemma-4-31b-it:free")

        mock_tool_result = {
            "successful": True,
            "data": {
                "username": "totalsolutions.socials",
                "followers_count": 0,
                "follows_count": 0,
            }
        }

        with patch.object(hr._registry, "chat_completion", side_effect=[step1_res, step2_res]):
            with patch.object(ComposioClient, "execute_tool", return_value=mock_tool_result) as mock_exec:
                res = hr.execute_agent_turn(
                    chat_id=123456,
                    user_message="okay so tell me how many followers am i havin on insta",
                    telegram_client=tg,
                )

        self.assertEqual(res.get("status"), "success")

        # Verify tool was executed with composio_instagram_get_user_info and {}
        mock_exec.assert_called_once_with("composio_instagram_get_user_info", {})

        # Verify final Telegram message sent to user contains username & follower count summary
        self.assertEqual(len(tg.sent), 1)
        self.assertIn("totalsolutions.socials", tg.sent[0][1])

    def test_03_task_classifier_does_not_demote_tool_requests(self):
        """Verify requests requiring tools are not demoted to FAST tier casual chat."""
        tier, reason = TaskClassifier.classify(
            "okay so tell me how many followers am i havin on insta",
            tools_available=True,
        )
        self.assertNotEqual(tier, ModelTier.FAST)


if __name__ == "__main__":
    unittest.main()
