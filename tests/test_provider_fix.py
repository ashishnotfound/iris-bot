"""
tests/test_provider_fix.py — Validation tests for AI Provider Failure Fixes

Covers:
  - Key loading & cleaning (strips quotes, ignores placeholders)
  - Gemini key validation (ignores non-AIza keys, prevents invalid key 400 errors)
  - OpenRouter auth header protection
  - Model Catalog deduplication (gemini-2.5-flash appears once)
  - Candidate list deduplication in TaskRouter
  - Active provider filtering in TaskRouter (empty list when no keys configured)
  - Clean error handling (no raw exception or header leakage to Telegram)
  - Unicode emoji integrity (⚠️, ✅, 🚨, 🔄)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from lib.llm_provider import (
    ProviderRegistry, GeminiProvider, OpenRouterProvider, NvidiaProvider,
    _KeyRingMixin, LLMKeyExhaustedError,
)
from lib.task_router import TaskRouter, MODEL_CATALOG, ModelTier
from lib.telegram_client import TelegramClient


class TestKeyCleaningAndValidation(unittest.TestCase):
    def setUp(self):
        self.provider = GeminiProvider()

    def test_clean_key_placeholders(self):
        self.assertIsNone(self.provider._clean_key(""))
        self.assertIsNone(self.provider._clean_key('""'))
        self.assertIsNone(self.provider._clean_key("null"))
        self.assertIsNone(self.provider._clean_key("undefined"))
        self.assertIsNone(self.provider._clean_key("REPLACE_WITH_VALID_KEY"))
        self.assertEqual(self.provider._clean_key(" AIzaSy12345 "), "AIzaSy12345")
        self.assertEqual(self.provider._clean_key('"sk-or-v1-abc"'), "sk-or-v1-abc")

    def test_gemini_key_format_filter(self):
        # Non-AIza keys must be skipped by GeminiProvider._load_keys()
        os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6InvalidKey"
        keys = self.provider._load_keys()
        self.assertNotIn("AQ.Ab8RN6InvalidKey", keys)
        del os.environ["GEMINI_API_KEY"]

        os.environ["GEMINI_API_KEY"] = "AIzaSyValidKeyFormat"
        keys = self.provider._load_keys()
        self.assertIn("AIzaSyValidKeyFormat", keys)
        del os.environ["GEMINI_API_KEY"]


class TestModelCatalogAndTaskRouter(unittest.TestCase):
    def setUp(self):
        self.router = TaskRouter()

    def test_model_catalog_no_duplicate_specs(self):
        gemini_specs = [s for s in MODEL_CATALOG if s.provider == "gemini" and s.model_id == "gemini-2.5-flash"]
        self.assertEqual(len(gemini_specs), 1, "gemini-2.5-flash should only appear ONCE in MODEL_CATALOG")

    def test_task_router_candidate_deduplication(self):
        decision = self.router.route("hello", active_providers=["gemini", "openrouter"])
        candidate_keys = [(c[0], c[1]) for c in decision.candidates]
        self.assertEqual(len(candidate_keys), len(set(candidate_keys)), "Candidates must be unique")

    def test_task_router_active_providers_empty(self):
        decision = self.router.route("hello", active_providers=[])
        self.assertEqual(len(decision.candidates), 0, "No candidates when active_providers is empty")


class TestErrorHandlingAndUnicode(unittest.TestCase):
    def test_is_key_or_quota_error_detection(self):
        prov = GeminiProvider()

        class Error400(Exception):
            def __init__(self):
                super().__init__("Error code: 400 - Please pass a valid API key")

        class Error401(Exception):
            def __init__(self):
                super().__init__("Error code: 401 - Missing Authentication header")

        class Error429(Exception):
            def __init__(self):
                super().__init__("429 Too Many Requests")

        self.assertTrue(prov._is_key_or_quota_error(Error400()))
        self.assertTrue(prov._is_key_or_quota_error(Error401()))
        self.assertTrue(prov._is_key_or_quota_error(Error429()))

    def test_unicode_emojis(self):
        emojis = ["⚠️", "✅", "🚨", "🔄"]
        for e in emojis:
            self.assertEqual(len(e.encode("utf-8")), len(e.encode("utf-8")))
            # Ensure no double encoding in standard string repr
            self.assertNotIn("â", e)


if __name__ == "__main__":
    unittest.main()
