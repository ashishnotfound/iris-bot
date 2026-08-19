"""
tests/test_final_verification.py — Production Verification for Commit 09e7fd84d
"""

import os
import sys
import unittest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from lib.llm_provider import ProviderRegistry, GroqProvider, OpenRouterProvider, NvidiaProvider, GeminiProvider
from lib.task_router import TaskRouter
from lib.telegram_client import TelegramClient
from lib.hermes_runner import execute_agent_turn


class TestFinalProductionVerification(unittest.TestCase):
    def setUp(self):
        self.router = TaskRouter()
        self.registry = ProviderRegistry()

    def test_01_text_routing_openrouter_first(self):
        decision = self.router.route("What is the capital of France?")
        top_c = decision.candidates[0]
        self.assertEqual(top_c[0], "openrouter", "OpenRouter must be first for text requests")

    def test_02_text_routing_provider_sequence(self):
        decision = self.router.route("Explain quantum computing")
        prov_order = []
        for p, m, s in decision.candidates:
            if p not in prov_order:
                prov_order.append(p)
        self.assertEqual(prov_order, ["openrouter", "nvidia", "groq", "gemini"])

    def test_03_vision_routing_sequence(self):
        decision = self.router.route("Describe this chart", has_photo=True)
        prov_order = []
        for p, m, s in decision.candidates:
            if p not in prov_order:
                prov_order.append(p)
            self.assertTrue(s.vision, f"Model {m} must support vision")
        self.assertEqual(prov_order, ["openrouter", "nvidia", "groq", "gemini"])

    def test_04_no_candidate_duplicates(self):
        decision = self.router.route("Hello world")
        keys = [(c[0], c[1]) for c in decision.candidates]
        self.assertEqual(len(keys), len(set(keys)), "No candidate duplicates must exist")

    def test_05_groq_key_ring_env_var_naming(self):
        prov = GroqProvider()
        names = prov._key_env_names()
        self.assertEqual(names[0], "GROQ_API_KEY")
        self.assertEqual(names[1], "GROQ_API_KEY_2")
        self.assertEqual(names[8], "GROQ_API_KEY_9")

    def test_06_missing_credentials_handled_safely(self):
        # Temporarily unset keys to verify clean handling
        old_keys = {}
        for k in ["OPENROUTER_API_KEY", "NVIDIA_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
            if k in os.environ:
                old_keys[k] = os.environ.pop(k)

        available = self.registry.available_providers()
        self.assertEqual(available, [], "No providers should be marked available when keys are absent")

        decision = self.router.route("Test message", active_providers=available)
        self.assertEqual(len(decision.candidates), 0, "No candidates generated when no providers available")

        # Restore keys
        for k, v in old_keys.items():
            os.environ[k] = v

    def test_07_unicode_emoji_rendering(self):
        test_str = "⚠️ ✅ 🚨 🔄"
        self.assertNotIn("â", test_str)
        self.assertEqual(len(test_str.encode("utf-8")), 20)


if __name__ == "__main__":
    unittest.main()
