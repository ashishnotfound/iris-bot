"""
tests/test_conversation_context.py — Test suite for Short-Term Conversation Context Pipeline
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lib.hermes_runner as hr
from lib.composio_client import ComposioClient


class MockTelegramClient:
    def __init__(self):
        self.sent = []

    def send_message(self, chat_id, text, parse_mode=None):
        self.sent.append((chat_id, text))
        return {"ok": True}

    def send_chat_action(self, chat_id, action="typing"):
        return {"ok": True}


class TestConversationContext(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(os.path.dirname(__file__)) / "test_context_tmp.db"
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception:
                pass
        try:
            import sqlite3
            hr._init_local_db()
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM local_messages")
                conn.commit()
        except Exception:
            pass
        self.db_patcher = patch.object(hr, "_local_sqlite_db_path", return_value=self.db_path)
        self.db_patcher.start()

    def tearDown(self):
        self.db_patcher.stop()
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception:
                pass

    def test_01_basic_followup_retains_context(self):
        """Test 1 — Basic follow-up: 'try checking it now' receives previous request as context."""
        tg = MockTelegramClient()

        # Turn 1
        res1 = ("Checking your Instagram account followers...", None, "gemini", "gemini-2.5-flash")
        with patch.object(hr._registry, "chat_completion", return_value=res1) as mock_cc1:
            hr.execute_agent_turn(
                chat_id=101,
                user_message="How many followers do I have on Instagram?",
                telegram_client=tg,
            )

        # Turn 2: 'try checking it now'
        res2 = ("Your Instagram profile has 100 followers.", None, "gemini", "gemini-2.5-flash")
        with patch.object(hr._registry, "chat_completion", return_value=res2) as mock_cc2:
            hr.execute_agent_turn(
                chat_id=101,
                user_message="try checking it now",
                telegram_client=tg,
            )

            # Inspect LLM messages payload in turn 2
            sent_messages = mock_cc2.call_args_list[0][0][0]

            # Verify system prompt + Turn 1 user msg + Turn 1 assistant reply + Turn 2 user msg
            roles = [m["role"] for m in sent_messages]
            self.assertEqual(roles, ["system", "user", "assistant", "user"])
            self.assertEqual(sent_messages[1]["content"], "How many followers do I have on Instagram?")
            self.assertEqual(sent_messages[2]["content"], "Checking your Instagram account followers...")
            self.assertEqual(sent_messages[3]["content"], "try checking it now")

    def test_02_pronoun_resolution_retains_context(self):
        """Test 2 — Pronoun resolution: 'what about it now?' receives previous context."""
        tg = MockTelegramClient()

        # Turn 1
        with patch.object(hr._registry, "chat_completion", return_value=("Checking Instagram...", None, "g", "m")):
            hr.execute_agent_turn(chat_id=102, user_message="Check my Instagram followers.", telegram_client=tg)

        # Turn 2
        with patch.object(hr._registry, "chat_completion", return_value=("Updated followers...", None, "g", "m")) as mock_cc:
            hr.execute_agent_turn(chat_id=102, user_message="what about it now?", telegram_client=tg)
            sent_messages = mock_cc.call_args_list[0][0][0]
            self.assertIn("Check my Instagram followers.", [m["content"] for m in sent_messages if m["role"] == "user"])

    def test_03_multi_turn_task_retains_context(self):
        """Test 3 — Multi-turn task: 'and tomorrow?' receives previous calendar request."""
        tg = MockTelegramClient()

        # Turn 1
        with patch.object(hr._registry, "chat_completion", return_value=("You have 2 meetings today.", None, "g", "m")):
            hr.execute_agent_turn(chat_id=103, user_message="Check my calendar.", telegram_client=tg)

        # Turn 2
        with patch.object(hr._registry, "chat_completion", return_value=("Tomorrow you have 1 meeting.", None, "g", "m")) as mock_cc:
            hr.execute_agent_turn(chat_id=103, user_message="and tomorrow?", telegram_client=tg)
            sent_messages = mock_cc.call_args_list[0][0][0]
            self.assertIn("Check my calendar.", [m["content"] for m in sent_messages if m["role"] == "user"])

    def test_04_conversation_isolation(self):
        """Test 4 — Conversation isolation: Chat A's history never leaks into Chat B's context."""
        tg = MockTelegramClient()

        # Chat A turn
        with patch.object(hr._registry, "chat_completion", return_value=("Check Instagram reply", None, "g", "m")):
            hr.execute_agent_turn(chat_id=201, user_message="Check Instagram.", telegram_client=tg)

        # Chat B turn
        with patch.object(hr._registry, "chat_completion", return_value=("Check Gmail reply", None, "g", "m")) as mock_cc_b:
            hr.execute_agent_turn(chat_id=202, user_message="Check Gmail.", telegram_client=tg)
            sent_messages_b = mock_cc_b.call_args_list[0][0][0]
            contents_b = [m["content"] for m in sent_messages_b if isinstance(m["content"], str)]
            self.assertNotIn("Check Instagram.", contents_b)
            self.assertIn("Check Gmail.", contents_b)

    def test_05_assistant_history_persisted_and_retrieved(self):
        """Test 5 — Assistant history: Both user AND assistant messages are stored and retrieved."""
        tg = MockTelegramClient()

        with patch.object(hr._registry, "chat_completion", return_value=("Assistant response #1", None, "g", "m")):
            hr.execute_agent_turn(chat_id=301, user_message="User message #1", telegram_client=tg)

        with patch.object(hr._registry, "chat_completion", return_value=("Assistant response #2", None, "g", "m")) as mock_cc:
            hr.execute_agent_turn(chat_id=301, user_message="User message #2", telegram_client=tg)
            sent_messages = mock_cc.call_args_list[0][0][0]
            user_msgs = [m["content"] for m in sent_messages if m["role"] == "user"]
            asst_msgs = [m["content"] for m in sent_messages if m["role"] == "assistant"]
            self.assertIn("User message #1", user_msgs)
            self.assertIn("Assistant response #1", asst_msgs)

    def test_06_tool_history_does_not_break_context(self):
        """Test 6 — Tool history: Tool calls and results do not destroy conversational context."""
        tg = MockTelegramClient()

        mock_tool_call = MagicMock()
        mock_tool_call.id = "tc_insta_777"
        mock_tool_call.function.name = "composio_instagram_get_user_info"
        mock_tool_call.function.arguments = "{}"

        # Turn 1 with tool call
        t1_step1 = ("", [mock_tool_call], "g", "m")
        t1_step2 = ("You have 50 followers.", None, "g", "m")

        with patch.object(hr._registry, "chat_completion", side_effect=[t1_step1, t1_step2]):
            with patch.object(ComposioClient, "execute_tool", return_value={"successful": True, "data": {"followers_count": 50}}):
                hr.execute_agent_turn(chat_id=401, user_message="How many followers do I have on Instagram?", telegram_client=tg)

        # Turn 2 follow-up
        with patch.object(hr._registry, "chat_completion", return_value=("Checked again: 50 followers.", None, "g", "m")) as mock_cc2:
            hr.execute_agent_turn(chat_id=401, user_message="Check it again.", telegram_client=tg)
            sent_messages = mock_cc2.call_args_list[0][0][0]
            contents = [str(m["content"]) for m in sent_messages]
            self.assertTrue(any("followers" in c.lower() for c in contents))

    def test_07_serverless_persistence(self):
        """Test 7 — Serverless persistence: Context survives across independent runner instances."""
        tg1 = MockTelegramClient()
        tg2 = MockTelegramClient()

        # Runner instance 1
        with patch.object(hr._registry, "chat_completion", return_value=("Response turn 1", None, "g", "m")):
            hr.execute_agent_turn(chat_id=501, user_message="First message", telegram_client=tg1)

        # Independent Runner instance 2 (simulating fresh serverless invocation)
        with patch.object(hr._registry, "chat_completion", return_value=("Response turn 2", None, "g", "m")) as mock_cc:
            hr.execute_agent_turn(chat_id=501, user_message="Second message", telegram_client=tg2)
            sent_messages = mock_cc.call_args_list[0][0][0]
            self.assertIn("First message", [m["content"] for m in sent_messages if m["role"] == "user"])


if __name__ == "__main__":
    unittest.main()
