"""
lib/llm_provider.py — Multi-Provider LLM with Key-Ring Failover

Providers supported:
  - OpenRouter: Multiple API keys (OPENROUTER_API_KEY, OPENROUTER_API_KEY_2, ..., N)
                Round-robin key selection; auto-marks exhausted keys on 429/402.
  - Gemini:     Google AI Studio free tier via OpenAI-compatible endpoint.
                Uses GEMINI_API_KEY or GOOGLE_API_KEY.
  - NVIDIA NIM: NVIDIA's OpenAI-compatible inference. Uses NVIDIA_API_KEY.

Usage:
    from lib.llm_provider import ProviderRegistry

    registry = ProviderRegistry()
    reply, provider = registry.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
        model="auto",
        vision=False,
    )
"""

from __future__ import annotations

import logging
import os
import site
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure site-packages in virtualenv is loaded
_venv_site = Path(__file__).resolve().parent.parent / "venv" / "Lib" / "site-packages"
if _venv_site.exists():
    site.addsitedir(str(_venv_site))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

DEFAULT_MODELS: Dict[str, str] = {
    "openrouter": "google/gemma-4-31b-it:free",
    "gemini": "gemini-2.5-flash",
    "nvidia": "meta/llama-3.1-405b-instruct",
}

VISION_MODELS: Dict[str, str] = {
    "openrouter": "google/gemma-4-31b-it:free",
    "gemini": "gemini-2.5-flash",
    "nvidia": "microsoft/phi-3-vision-128k-instruct",
}

# ---------------------------------------------------------------------------
# Base Provider ABC
# ---------------------------------------------------------------------------


class LLMKeyExhaustedError(RuntimeError):
    """Raised when a specific API key is rate-limited or out of credits."""


class LLMProvider(ABC):
    """Abstract base class for an LLM backend."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
    ) -> str: ...


# ---------------------------------------------------------------------------
# OpenRouter Provider — multi-key ring
# ---------------------------------------------------------------------------


class OpenRouterProvider(LLMProvider):
    """OpenRouter with automatic multi-key round-robin and failover.

    Keys are read from:
      OPENROUTER_API_KEY      — primary key
      OPENROUTER_API_KEY_2    — second key
      ...
      OPENROUTER_API_KEY_9    — ninth key

    When a key hits 429 or 402, it is marked exhausted for KEY_EXHAUSTED_TTL
    seconds before being retried.
    """

    KEY_EXHAUSTED_TTL = 3600  # 1 hour

    def __init__(self) -> None:
        self._keys: List[str] = self._load_keys()
        self._exhausted_until: Dict[str, float] = {}
        self._key_index = 0

    @property
    def name(self) -> str:
        return "openrouter"

    def _load_keys(self) -> List[str]:
        keys: List[str] = []
        primary = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if primary:
            keys.append(primary)
        for i in range(2, 10):
            k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "").strip()
            if k:
                keys.append(k)
        return keys

    def is_available(self) -> bool:
        return bool(self._active_keys())

    def _active_keys(self) -> List[str]:
        now = time.time()
        return [k for k in self._keys if now >= self._exhausted_until.get(k, 0)]

    def _mark_exhausted(self, key: str) -> None:
        self._exhausted_until[key] = time.time() + self.KEY_EXHAUSTED_TTL
        logger.warning("OpenRouter key ...%s exhausted for %ds", key[-6:], self.KEY_EXHAUSTED_TTL)

    def _pick_key(self) -> Optional[str]:
        active = self._active_keys()
        if not active:
            return None
        self._key_index = (self._key_index + 1) % len(active)
        return active[self._key_index % len(active)]

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        active = self._active_keys()
        if not active:
            raise LLMKeyExhaustedError("All OpenRouter keys are exhausted.")

        use_model = model or (VISION_MODELS["openrouter"] if vision else DEFAULT_MODELS["openrouter"])
        last_error: Optional[Exception] = None

        for key in list(active):
            try:
                client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=key,
                    default_headers={
                        "HTTP-Referer": "https://github.com/iris-agent",
                        "X-Title": "Iris Agent",
                    },
                )
                resp = client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (429, 402) or any(w in err_str for w in ("rate limit", "insufficient", "credits", "quota")):
                    self._mark_exhausted(key)
                    last_error = e
                    continue
                raise RuntimeError(f"OpenRouter API error: {e}") from e

        raise LLMKeyExhaustedError(f"All OpenRouter keys failed. Last: {last_error}")

    def list_free_models(self) -> List[Dict[str, str]]:
        """Fetch free models from OpenRouter catalog."""
        try:
            import requests
            key = self._pick_key()
            if not key:
                return []
            r = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 200:
                models = r.json().get("data", [])
                free = [
                    {"id": m["id"], "name": m.get("name", m["id"])}
                    for m in models
                    if str(m.get("pricing", {}).get("prompt", "1")) == "0"
                    or m["id"].endswith(":free")
                ]
                return free[:25]
        except Exception as e:
            logger.warning("Could not fetch OpenRouter models: %s", e)
        return [
            {"id": "google/gemini-2.0-flash-exp:free", "name": "Gemini 2.0 Flash (free)"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B (free)"},
            {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1 (free)"},
            {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B (free)"},
        ]


# ---------------------------------------------------------------------------
# Gemini Provider
# ---------------------------------------------------------------------------


class GeminiProvider(LLMProvider):
    """Google Gemini via OpenAI-compatible endpoint (free tier)."""

    @property
    def name(self) -> str:
        return "gemini"

    def _key(self) -> Optional[str]:
        return (
            os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        ).strip() or None

    def is_available(self) -> bool:
        return bool(self._key())

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
    ) -> str:
        key = self._key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY not configured.")
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed.")

        use_model = model or (VISION_MODELS["gemini"] if vision else DEFAULT_MODELS["gemini"])
        client = OpenAI(base_url=GEMINI_BASE_URL, api_key=key)
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e


# ---------------------------------------------------------------------------
# NVIDIA NIM Provider
# ---------------------------------------------------------------------------


class NvidiaProvider(LLMProvider):
    """NVIDIA NIM via OpenAI-compatible endpoint (free credits)."""

    @property
    def name(self) -> str:
        return "nvidia"

    def _key(self) -> Optional[str]:
        return os.environ.get("NVIDIA_API_KEY", "").strip() or None

    def is_available(self) -> bool:
        return bool(self._key())

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
    ) -> str:
        key = self._key()
        if not key:
            raise RuntimeError("NVIDIA_API_KEY not configured.")
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed.")

        use_model = model or (VISION_MODELS["nvidia"] if vision else DEFAULT_MODELS["nvidia"])
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"NVIDIA NIM API error: {e}") from e


# ---------------------------------------------------------------------------
# Provider Registry — ordered failover
# ---------------------------------------------------------------------------


class ProviderRegistry:
    """Manages multiple LLM providers and executes ordered failover.

    Default priority: Gemini → OpenRouter → NVIDIA NIM
    """

    _PROVIDER_CLASSES: Dict[str, type] = {
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
        "nvidia": NvidiaProvider,
    }

    def __init__(self, provider_order: Optional[List[str]] = None) -> None:
        order = provider_order or ["gemini", "openrouter", "nvidia"]
        self._providers: List[LLMProvider] = [
            self._PROVIDER_CLASSES[n]()  # type: ignore[abstract]
            for n in order
            if n in self._PROVIDER_CLASSES
        ]

    def available_providers(self) -> List[str]:
        return [p.name for p in self._providers if p.is_available()]

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        candidates: Optional[List[Tuple[str, str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
    ) -> Tuple[str, str, str]:
        """Execute a chat completion with candidate-aware provider failover.

        Args:
            messages:    OpenAI-format messages.
            candidates:  Optional list of (provider_name, model_id, spec) tuples
                         from TaskRouter.
            model:       Optional manual model override.
            provider:    Optional manual provider override.
            temperature: Sampling temperature.
            max_tokens:  Max tokens in completion.
            vision:      If True, enforces vision capability.

        Returns:
            (reply_text, provider_name_used, model_id_used)

        Raises:
            RuntimeError: if all candidate providers fail.
        """
        errors: List[str] = []

        # Case A: Candidates list provided by TaskRouter
        if candidates:
            for prov_name, model_id, spec in candidates:
                p = self.get_provider(prov_name)
                if not p or not p.is_available():
                    errors.append(f"{prov_name}/{model_id}: provider unconfigured or unavailable")
                    continue
                try:
                    text = p.chat_completion(
                        messages,
                        model=model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        vision=vision,
                    )
                    logger.info("LLM success: provider=%s model=%s", prov_name, model_id)
                    return text, prov_name, model_id
                except LLMKeyExhaustedError as e:
                    errors.append(f"{prov_name}/{model_id}: key exhausted — {e}")
                    continue
                except RuntimeError as e:
                    errors.append(f"{prov_name}/{model_id}: {e}")
                    logger.warning("Candidate %s/%s failed: %s", prov_name, model_id, e)
                    continue

        # Case B: Standard provider order fallback
        prov_candidates = self._providers
        if provider:
            prov_candidates = [p for p in self._providers if p.name == provider]
            if not prov_candidates:
                raise RuntimeError(f"Provider not found or not configured: {provider!r}")

        for p in prov_candidates:
            if not p.is_available():
                errors.append(f"{p.name}: not configured (missing API key)")
                continue
            use_model = model
            try:
                text = p.chat_completion(
                    messages,
                    model=use_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    vision=vision,
                )
                used_model = use_model or (VISION_MODELS.get(p.name, "") if vision else DEFAULT_MODELS.get(p.name, ""))
                logger.info("LLM response from provider=%s model=%s", p.name, used_model)
                return text, p.name, used_model
            except LLMKeyExhaustedError as e:
                errors.append(f"{p.name}: key exhausted — {e}")
                continue
            except RuntimeError as e:
                errors.append(f"{p.name}: {e}")
                logger.warning("Provider %s failed, trying next: %s", p.name, e)
                continue

        raise RuntimeError(
            "All LLM providers failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        for p in self._providers:
            if p.name == name:
                return p
        return None
