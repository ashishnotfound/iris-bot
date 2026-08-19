"""
tests/test_message_identity_and_image_pipeline.py — E2E test suite for message identity, vision pipeline, and task switching
"""

import json
import os
import random
import sys
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

    def get_file(self, file_id):
        return {"ok": True, "result": {"file_path": "photos/test.jpg"}}

    def download_file(self, file_path):
        # Return valid JPEG magic bytes
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"


class TestMessageIdentityAndImagePipeline(unittest.TestCase):
    def setUp(self):
        import uuid
        self.db_path = Path(os.path.dirname(__file__)) / f"test_img_{uuid.uuid4().hex}.db"
        self.db_patcher = patch.object(hr, "_local_sqlite_db_path", return_value=self.db_path)
        self.db_patcher.start()

    def tearDown(self):
        self.db_patcher.stop()
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception:
                pass

    def test_01_multi_turn_sequence_image_post_does_not_repeat_follower_count(self):
        """Reproduce exact 6-turn sequence: image + 'can you post this up' must NOT return follower count."""
        tg = MockTelegramClient()
        chat_id = random.randint(100000, 999999)

        # 1. hey Iris
        with patch.object(hr._registry, "chat_completion", return_value=("Hey Reyo!", None, "g", "m")):
            hr.execute_agent_turn(chat_id=chat_id, user_message="hey Iris", telegram_client=tg)

        # 2. how many followers am I having on Instagram?
        tc1 = MagicMock()
        tc1.id = "tc_f1"
        tc1.function.name = "composio_instagram_get_user_info"
        tc1.function.arguments = "{}"
        with patch.object(hr._registry, "chat_completion", side_effect=[("", [tc1], "g", "m"), ("You have 0 followers on Instagram.", None, "g", "m")]):
            with patch.object(ComposioClient, "execute_tool", return_value={"successful": True, "data": {"followers_count": 0}}):
                hr.execute_agent_turn(chat_id=chat_id, user_message="how many followers am I having on Instagram?", telegram_client=tg)

        # 3. try checking it now
        tc2 = MagicMock()
        tc2.id = "tc_f2"
        tc2.function.name = "composio_instagram_get_user_info"
        tc2.function.arguments = "{}"
        with patch.object(hr._registry, "chat_completion", side_effect=[("", [tc2], "g", "m"), ("Checked: 0 followers.", None, "g", "m")]):
            with patch.object(ComposioClient, "execute_tool", return_value={"successful": True, "data": {"followers_count": 0}}):
                hr.execute_agent_turn(chat_id=chat_id, user_message="try checking it now", telegram_client=tg)

        # 4. hey Iris try checking my insta and tell me how many followers am I having there
        with patch.object(hr._registry, "chat_completion", return_value=("You have 0 followers on Instagram.", None, "g", "m")):
            hr.execute_agent_turn(chat_id=chat_id, user_message="hey Iris try checking my insta and tell me how many followers am I having there", telegram_client=tg)

        # 5. damn T-T
        with patch.object(hr._registry, "chat_completion", return_value=("I understand!", None, "g", "m")):
            hr.execute_agent_turn(chat_id=chat_id, user_message="damn T-T", telegram_client=tg)

        # 6. [IMAGE] "can you post this up"
        tc_post = MagicMock()
        tc_post.id = "tc_post_1"
        tc_post.function.name = "composio_instagram_create_photo_post"
        tc_post.function.arguments = json.dumps({"caption": "can you post this up"})

        post_step1 = ("", [tc_post], "g", "m")
        post_step2 = ("I am ready to post this photo to Instagram with caption 'can you post this up'! Please confirm to publish.", None, "g", "m")

        with patch.object(hr._registry, "chat_completion", side_effect=[post_step1, post_step2]) as mock_cc:
            with patch.object(ComposioClient, "execute_tool", return_value={"successful": True, "data": {"id": "post_123"}}):
                hr.execute_agent_turn(
                    chat_id=chat_id,
                    user_message="can you post this up",
                    photo=[{"file_id": "photo_file_123"}],
                    telegram_client=tg,
                )

        # Assert final message sent to Telegram in turn 6 does NOT say "You have 0 followers"
        last_sent = tg.sent[-1][1]
        self.assertNotIn("0 followers", last_sent)
        self.assertNotIn("how many followers", last_sent.lower())

    def test_02_vision_history_sanitization(self):
        """Verify _sanitize_content_for_history strips base64 vision URLs."""
        raw_content = [
            {"type": "text", "text": "can you post this up"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,1234567890abcdef"}},
        ]
        sanitized = hr._sanitize_content_for_history(raw_content)
        self.assertNotIn("data:image/jpeg;base64", sanitized)
        self.assertIn("can you post this up", sanitized)
        self.assertIn("[attached image]", sanitized)


if __name__ == "__main__":
    unittest.main()
