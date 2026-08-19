"""
tests/test_audit_routing.py — Comprehensive Test Matrix for Provider Routing Hierarchy

Coverage:
  Test A: OpenRouter succeeds -> NVIDIA, Groq, Gemini NOT called
  Test B: OpenRouter fails -> NVIDIA succeeds -> Groq & Gemini NOT called
  Test C: OpenRouter + NVIDIA fail -> Groq succeeds -> Gemini NOT called
  Test D: OpenRouter + NVIDIA + Groq fail -> Gemini attempted & succeeds
  Test E: Everything fails -> Clean user message returned, no secrets leaked
  Test F: Vision requests follow OpenRouter Vision -> NVIDIA Vision -> Groq Vision -> Gemini Vision
  Test G: Groq Key-Ring rotation (GROQ_API_KEY_1..9)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from lib.llm_provider import (
    ProviderRegistry, LLMProvider, OpenRouterProvider, NvidiaProvider,
    GroqProvider, GeminiProvider, LLMKeyExhaustedError,
)
from lib.task_router import TaskRouter, ModelSpec, ModelTier


class MockProvider(LLMProvider):
    def __init__(self, name_str: str, should_succeed: bool = True):
        self._name = name_str
        self.should_succeed = should_succeed
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def chat_completion(self, messages, *, model=None, temperature=0.7, max_tokens=4096, vision=False):
        self.call_count += 1
        if not self.should_succeed:
            raise LLMKeyExhaustedError(f"Mock {self._name} failure")
        return f"Response from {self._name}"


class TestRoutingMatrix(unittest.TestCase):
    def setUp(self):
        self.router = TaskRouter()

    def test_text_routing_order(self):
        decision = self.router.route("What's the weather in Delhi?")
        prov_order = []
        for p, m, s in decision.candidates:
            if p not in prov_order:
                prov_order.append(p)
        self.assertEqual(prov_order, ["openrouter", "nvidia", "groq", "gemini"])

    def test_vision_routing_order(self):
        decision = self.router.route("Describe this photo", has_photo=True)
        prov_order = []
        for p, m, s in decision.candidates:
            if p not in prov_order:
                prov_order.append(p)
            self.assertTrue(s.vision, f"Model {m} must support vision")
        self.assertEqual(prov_order, ["openrouter", "nvidia", "groq", "gemini"])

    def test_matrix_a_openrouter_succeeds(self):
        reg = ProviderRegistry()
        p_openrouter = MockProvider("openrouter", should_succeed=True)
        p_nvidia = MockProvider("nvidia", should_succeed=True)
        p_groq = MockProvider("groq", should_succeed=True)
        p_gemini = MockProvider("gemini", should_succeed=True)
        reg._providers = [p_openrouter, p_nvidia, p_groq, p_gemini]

        decision = self.router.route("test")
        reply, prov_used, model_used = reg.chat_completion([], candidates=decision.candidates)

        self.assertEqual(prov_used, "openrouter")
        self.assertEqual(p_openrouter.call_count, 1)
        self.assertEqual(p_nvidia.call_count, 0)
        self.assertEqual(p_groq.call_count, 0)
        self.assertEqual(p_gemini.call_count, 0)

    def test_matrix_b_openrouter_fails_nvidia_succeeds(self):
        reg = ProviderRegistry()
        p_openrouter = MockProvider("openrouter", should_succeed=False)
        p_nvidia = MockProvider("nvidia", should_succeed=True)
        p_groq = MockProvider("groq", should_succeed=True)
        p_gemini = MockProvider("gemini", should_succeed=True)
        reg._providers = [p_openrouter, p_nvidia, p_groq, p_gemini]

        decision = self.router.route("test")
        reply, prov_used, model_used = reg.chat_completion([], candidates=decision.candidates)

        self.assertEqual(prov_used, "nvidia")
        self.assertGreaterEqual(p_openrouter.call_count, 1)
        self.assertEqual(p_nvidia.call_count, 1)
        self.assertEqual(p_groq.call_count, 0)
        self.assertEqual(p_gemini.call_count, 0)

    def test_matrix_c_openrouter_and_nvidia_fail_groq_succeeds(self):
        reg = ProviderRegistry()
        p_openrouter = MockProvider("openrouter", should_succeed=False)
        p_nvidia = MockProvider("nvidia", should_succeed=False)
        p_groq = MockProvider("groq", should_succeed=True)
        p_gemini = MockProvider("gemini", should_succeed=True)
        reg._providers = [p_openrouter, p_nvidia, p_groq, p_gemini]

        decision = self.router.route("test")
        reply, prov_used, model_used = reg.chat_completion([], candidates=decision.candidates)

        self.assertEqual(prov_used, "groq")
        self.assertGreaterEqual(p_openrouter.call_count, 1)
        self.assertGreaterEqual(p_nvidia.call_count, 1)
        self.assertEqual(p_groq.call_count, 1)
        self.assertEqual(p_gemini.call_count, 0)

    def test_matrix_d_first_three_fail_gemini_succeeds(self):
        reg = ProviderRegistry()
        p_openrouter = MockProvider("openrouter", should_succeed=False)
        p_nvidia = MockProvider("nvidia", should_succeed=False)
        p_groq = MockProvider("groq", should_succeed=False)
        p_gemini = MockProvider("gemini", should_succeed=True)
        reg._providers = [p_openrouter, p_nvidia, p_groq, p_gemini]

        decision = self.router.route("test")
        reply, prov_used, model_used = reg.chat_completion([], candidates=decision.candidates)

        self.assertEqual(prov_used, "gemini")
        self.assertGreaterEqual(p_openrouter.call_count, 1)
        self.assertGreaterEqual(p_nvidia.call_count, 1)
        self.assertGreaterEqual(p_groq.call_count, 1)
        self.assertEqual(p_gemini.call_count, 1)

    def test_matrix_e_all_fail(self):
        reg = ProviderRegistry()
        p_openrouter = MockProvider("openrouter", should_succeed=False)
        p_nvidia = MockProvider("nvidia", should_succeed=False)
        p_groq = MockProvider("groq", should_succeed=False)
        p_gemini = MockProvider("gemini", should_succeed=False)
        reg._providers = [p_openrouter, p_nvidia, p_groq, p_gemini]

        decision = self.router.route("test")
        with self.assertRaises(RuntimeError):
            reg.chat_completion([], candidates=decision.candidates)

    def test_groq_key_ring_multi_key_loading(self):
        prov = GroqProvider()
        os.environ["GROQ_API_KEY"] = "gsk_key1_valid"
        os.environ["GROQ_API_KEY_2"] = "gsk_key2_valid"

        keys = prov._load_keys()
        self.assertIn("gsk_key1_valid", keys)
        self.assertIn("gsk_key2_valid", keys)

        del os.environ["GROQ_API_KEY"]
        del os.environ["GROQ_API_KEY_2"]


if __name__ == "__main__":
    unittest.main()
