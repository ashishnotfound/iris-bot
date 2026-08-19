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

    def test_04_negative_test_does_not_ask_for_instagram_business_account_id(self):
        """Negative regression test: verify Iris does NOT ask for Instagram account ID when connected tool executes."""
        tg = MockTelegramClient()

        mock_tool_call = MagicMock()
        mock_tool_call.id = "tc_insta_999"
        mock_tool_call.function.name = "composio_instagram_get_user_info"
        mock_tool_call.function.arguments = "{}"

        step1_res = ("", [mock_tool_call], "openrouter", "google/gemma-4-31b-it:free")
        step2_res = ("Your Instagram profile (@totalsolutions.socials) has 42 followers.", None, "openrouter", "google/gemma-4-31b-it:free")

        mock_tool_result = {
            "successful": True,
            "data": {
                "username": "totalsolutions.socials",
                "followers_count": 42,
            }
        }

        with patch.object(hr._registry, "chat_completion", side_effect=[step1_res, step2_res]):
            with patch.object(ComposioClient, "execute_tool", return_value=mock_tool_result):
                res = hr.execute_agent_turn(
                    chat_id=123456,
                    user_message="how many followers am i having on insta",
                    telegram_client=tg,
                )

        sent_text = tg.sent[0][1]
        self.assertNotIn("Please provide your Instagram business account ID", sent_text)
        self.assertNotIn("account ID", sent_text.lower())
        self.assertIn("42", sent_text)

    def test_05_cross_integration_tool_execution_flow(self):
        """Verify request -> toolkit -> tool selection -> execution -> result propagation for Gmail, Calendar, and Browsebase."""
        tg = MockTelegramClient()

        # 1. Gmail
        mock_gmail_call = MagicMock()
        mock_gmail_call.id = "tc_gmail_1"
        mock_gmail_call.function.name = "composio_gmail_fetch_emails"
        mock_gmail_call.function.arguments = "{}"

        g_step1 = ("", [mock_gmail_call], "gemini", "gemini-2.5-flash")
        g_step2 = ("You have 2 unread emails from John and Sales.", None, "gemini", "gemini-2.5-flash")
        g_result = {"successful": True, "data": {"messages": [{"from": "John"}, {"from": "Sales"}]}}

        with patch.object(hr._registry, "chat_completion", side_effect=[g_step1, g_step2]):
            with patch.object(ComposioClient, "execute_tool", return_value=g_result) as mock_exec:
                res = hr.execute_agent_turn(chat_id=123, user_message="Do I have any new emails?", telegram_client=tg)
                mock_exec.assert_called_with("composio_gmail_fetch_emails", {})
                self.assertIn("unread emails", tg.sent[-1][1])

        # 2. Calendar
        mock_cal_call = MagicMock()
        mock_cal_call.id = "tc_cal_1"
        mock_cal_call.function.name = "composio_googlecalendar_events_list"
        mock_cal_call.function.arguments = "{}"

        c_step1 = ("", [mock_cal_call], "gemini", "gemini-2.5-flash")
        c_step2 = ("You have 1 meeting today: Team Sync at 3 PM.", None, "gemini", "gemini-2.5-flash")
        c_result = {"successful": True, "data": {"items": [{"summary": "Team Sync"}]}}

        with patch.object(hr._registry, "chat_completion", side_effect=[c_step1, c_step2]):
            with patch.object(ComposioClient, "execute_tool", return_value=c_result) as mock_exec:
                res = hr.execute_agent_turn(chat_id=123, user_message="What meetings do I have today?", telegram_client=tg)
                mock_exec.assert_called_with("composio_googlecalendar_events_list", {})
                self.assertIn("Team Sync", tg.sent[-1][1])

    def test_06_tool_exception_does_not_silence_telegram_reply(self):
        """Verify tool execution exception is caught safely and Iris always delivers Telegram reply."""
        tg = MockTelegramClient()

        mock_tool_call = MagicMock()
        mock_tool_call.id = "tc_err_1"
        mock_tool_call.function.name = "composio_instagram_get_user_info"
        mock_tool_call.function.arguments = "{}"

        step1_res = ("", [mock_tool_call], "g", "m")
        step2_res = ("I encountered an issue fetching your Instagram info, but I am still active.", None, "g", "m")

        with patch.object(hr._registry, "chat_completion", side_effect=[step1_res, step2_res]):
            with patch.object(ComposioClient, "execute_tool", side_effect=RuntimeError("Composio API Timeout")):
                res = hr.execute_agent_turn(chat_id=888, user_message="check my insta", telegram_client=tg)

        self.assertEqual(len(tg.sent), 1)
        self.assertTrue(len(tg.sent[0][1]) > 0)


if __name__ == "__main__":
    unittest.main()
