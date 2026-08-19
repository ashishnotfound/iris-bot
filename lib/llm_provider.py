"""
lib/llm_provider.py — Multi-Provider LLM with Key-Ring Failover

Providers supported:
  - Gemini:     Google AI Studio free tier via OpenAI-compatible endpoint.
                Uses GEMINI_API_KEY or GOOGLE_API_KEY.
                Multi-key ring: GEMINI_API_KEY, GEMINI_API_KEY_2, ..., GEMINI_API_KEY_9
  - OpenRouter: Multiple API keys (OPENROUTER_API_KEY, OPENROUTER_API_KEY_2, ..., N)
                Round-robin key selection; auto-marks exhausted keys on 429/402.
  - NVIDIA NIM: NVIDIA's OpenAI-compatible inference. Uses NVIDIA_API_KEY.

Usage:
    from lib.llm_provider import ProviderRegistry

    registry = ProviderRegistry()
    reply, provider, model = registry.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
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
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODELS: Dict[str, str] = {
    "openrouter": "google/gemma-4-31b-it:free",
    "nvidia": "meta/llama-3.1-8b-instruct",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
}

VISION_MODELS: Dict[str, str] = {
    "openrouter": "google/gemma-4-31b-it:free",
    "nvidia": "meta/llama-3.2-90b-vision-instruct",
    "groq": "llama-3.2-11b-vision-preview",
    "gemini": "gemini-2.5-flash",
}

# How long to back off an exhausted key (seconds)
KEY_EXHAUSTED_TTL = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMKeyExhaustedError(RuntimeError):
    """Raised when a specific API key is rate-limited or out of credits."""


# ---------------------------------------------------------------------------
# Base Provider ABC
# ---------------------------------------------------------------------------


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
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Any]: ...


# ---------------------------------------------------------------------------
# Shared multi-key ring mixin
# ---------------------------------------------------------------------------


class _KeyRingMixin:
    """
    Shared behaviour for providers that support multiple API keys.

    Subclasses must implement `_key_env_names()` to return a list of env var
    names in priority order.
    """

    _exhausted_until: Dict[str, float]
    _key_index: int

    def _key_env_names(self) -> List[str]:
        raise NotImplementedError

    def _clean_key(self, raw_val: str) -> Optional[str]:
        if not raw_val:
            return None
        val = raw_val.strip().strip("'\"")
        val_lower = val.lower()
        if not val or val_lower in (
            "none", "null", "undefined", "your_api_key_here",
            "replace_with_valid_aiza_key", "your_key_here", ""
        ) or val_lower.startswith("replace_with_") or val_lower.startswith("your_"):
            return None
        return val

    def _load_keys(self) -> List[str]:
        keys: List[str] = []
        for name in self._key_env_names():
            k = self._clean_key(os.environ.get(name, ""))
            if k and k not in keys:
                keys.append(k)
        return keys

    def _active_keys(self, keys: List[str]) -> List[str]:
        now = time.time()
        return [k for k in keys if now >= self._exhausted_until.get(k, 0)]

    def _mark_exhausted(self, key: str, ttl: Optional[int] = None) -> None:
        actual_ttl = ttl if ttl is not None else 60
        self._exhausted_until[key] = time.time() + actual_ttl
        logger.warning(
            "%s key ...%s exhausted; backing off for %ds",
            self.__class__.__name__,
            key[-6:],
            actual_ttl,
        )

    def _pick_key(self, keys: List[str]) -> Optional[str]:
        active = self._active_keys(keys)
        if not active:
            return None
        key = active[self._key_index % len(active)]
        self._key_index = (self._key_index + 1) % len(active)
        return key

    def _is_key_or_quota_error(self, e: Exception) -> bool:
        err_str = str(e).lower()
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (400, 401, 402, 403, 429):
            return True
        return any(
            w in err_str
            for w in (
                "rate limit", "insufficient", "credits", "quota",
                "resource_exhausted", "exhausted", "too many requests",
                "invalid api key", "invalid_api_key", "please pass a valid api key",
                "missing authentication header", "unauthorized", "authentication",
            )
        )

    _is_quota_error = _is_key_or_quota_error


# ---------------------------------------------------------------------------
# Gemini Provider — multi-key ring
# ---------------------------------------------------------------------------


class GeminiProvider(LLMProvider, _KeyRingMixin):
    """Google Gemini via OpenAI-compatible endpoint (free tier) with key-ring support.

    Keys are read from:
      GEMINI_API_KEY  / GOOGLE_API_KEY      — primary (both checked)
      GEMINI_API_KEY_2                       — second key
      ...
      GEMINI_API_KEY_9                       — ninth key

    When a key hits 429/quota, it is marked exhausted for KEY_EXHAUSTED_TTL
    seconds before being retried.
    """

    def __init__(self) -> None:
        self._exhausted_until: Dict[str, float] = {}
        self._key_index = 0
        self._keys = self._load_keys()

    def _load_keys(self) -> List[str]:
        keys: List[str] = []
        for name in self._key_env_names():
            cleaned = self._clean_key(os.environ.get(name, ""))
            if cleaned:
                if not cleaned.startswith("AIza"):
                    logger.warning(
                        "GeminiProvider: %s value does not start with 'AIza'. Skipping invalid key format.",
                        name,
                    )
                    continue
                if cleaned not in keys:
                    keys.append(cleaned)
        return keys

    def _key_env_names(self) -> List[str]:
        names = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        for i in range(2, 10):
            names.append(f"GEMINI_API_KEY_{i}")
        return names

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        # Re-load keys on each check so keys set after module init are visible
        self._keys = self._load_keys()
        return bool(self._active_keys(self._keys))

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Any]:
        self._keys = self._load_keys()
        active = self._active_keys(self._keys)
        if not active:
            raise LLMKeyExhaustedError("All Gemini keys are exhausted or unconfigured.")

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        use_model = model or (VISION_MODELS["gemini"] if vision else DEFAULT_MODELS["gemini"])
        last_error: Optional[Exception] = None

        for key in list(active):
            try:
                client = OpenAI(base_url=GEMINI_BASE_URL, api_key=key)
                kwargs = {
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                resp = client.chat.completions.create(**kwargs)
                choice_msg = resp.choices[0].message
                return (choice_msg.content or "", getattr(choice_msg, "tool_calls", None))
            except Exception as e:
                if self._is_key_or_quota_error(e):
                    self._mark_exhausted(key)
                    last_error = e
                    continue
                raise RuntimeError(f"Gemini API error: {e}") from e

        raise LLMKeyExhaustedError(f"All Gemini keys failed. Last: {last_error}")


# ---------------------------------------------------------------------------
# OpenRouter Provider — multi-key ring
# ---------------------------------------------------------------------------


class OpenRouterProvider(LLMProvider, _KeyRingMixin):
    """OpenRouter with automatic multi-key round-robin and failover.

    Keys are read from:
      OPENROUTER_API_KEY      — primary key
      OPENROUTER_API_KEY_2    — second key
      ...
      OPENROUTER_API_KEY_9    — ninth key
    """

    def __init__(self) -> None:
        self._exhausted_until: Dict[str, float] = {}
        self._key_index = 0
        self._keys = self._load_keys()

    def _key_env_names(self) -> List[str]:
        names = ["OPENROUTER_API_KEY"]
        for i in range(2, 10):
            names.append(f"OPENROUTER_API_KEY_{i}")
        return names

    @property
    def name(self) -> str:
        return "openrouter"

    def is_available(self) -> bool:
        self._keys = self._load_keys()
        return bool(self._active_keys(self._keys))

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Any]:
        self._keys = self._load_keys()
        active = self._active_keys(self._keys)
        if not active:
            raise LLMKeyExhaustedError("All OpenRouter keys are exhausted.")

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

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
                kwargs = {
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                resp = client.chat.completions.create(**kwargs)
                choice_msg = resp.choices[0].message
                return (choice_msg.content or "", getattr(choice_msg, "tool_calls", None))
            except Exception as e:
                if self._is_key_or_quota_error(e):
                    self._mark_exhausted(key)
                    last_error = e
                    continue
                raise RuntimeError(f"OpenRouter API error: {e}") from e

        raise LLMKeyExhaustedError(f"All OpenRouter keys failed. Last: {last_error}")

    def list_free_models(self) -> List[Dict[str, str]]:
        """Fetch free models from OpenRouter catalog."""
        try:
            import requests
            self._keys = self._load_keys()
            key = self._pick_key(self._keys)
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
# NVIDIA NIM Provider
# ---------------------------------------------------------------------------


class NvidiaProvider(LLMProvider, _KeyRingMixin):
    """NVIDIA NIM via OpenAI-compatible endpoint (free credits).

    Keys are read from:
      NVIDIA_API_KEY     — primary key
      NVIDIA_API_KEY_2   — second key
      ...
    """

    def __init__(self) -> None:
        self._exhausted_until: Dict[str, float] = {}
        self._key_index = 0
        self._keys = self._load_keys()

    def _key_env_names(self) -> List[str]:
        names = ["NVIDIA_API_KEY"]
        for i in range(2, 10):
            names.append(f"NVIDIA_API_KEY_{i}")
        return names

    @property
    def name(self) -> str:
        return "nvidia"

    def is_available(self) -> bool:
        self._keys = self._load_keys()
        return bool(self._active_keys(self._keys))

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Any]:
        self._keys = self._load_keys()
        active = self._active_keys(self._keys)
        if not active:
            raise LLMKeyExhaustedError("All NVIDIA keys are exhausted or unconfigured.")

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed.")

        use_model = model or (VISION_MODELS["nvidia"] if vision else DEFAULT_MODELS["nvidia"])
        last_error: Optional[Exception] = None

        for key in list(active):
            try:
                client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)
                kwargs = {
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                resp = client.chat.completions.create(**kwargs)
                choice_msg = resp.choices[0].message
                return (choice_msg.content or "", getattr(choice_msg, "tool_calls", None))
            except Exception as e:
                if self._is_key_or_quota_error(e):
                    self._mark_exhausted(key)
                    last_error = e
                    continue
                raise RuntimeError(f"NVIDIA NIM API error: {e}") from e

        raise LLMKeyExhaustedError(f"All NVIDIA keys failed. Last: {last_error}")


# ---------------------------------------------------------------------------
# Provider Registry — ordered failover
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Groq Cloud Provider — ultra-fast LLM & Vision
# ---------------------------------------------------------------------------


class GroqProvider(LLMProvider, _KeyRingMixin):
    """Groq Cloud via OpenAI-compatible endpoint (free tier, ultra-fast & vision).

    Keys are read from:
      GROQ_API_KEY      — primary key
      GROQ_API_KEY_2    — second key
      ...
      GROQ_API_KEY_9    — ninth key
    """

    def __init__(self) -> None:
        self._exhausted_until: Dict[str, float] = {}
        self._key_index = 0
        self._keys = self._load_keys()

    def _key_env_names(self) -> List[str]:
        names = ["GROQ_API_KEY"]
        for i in range(2, 10):
            names.append(f"GROQ_API_KEY_{i}")
        return names

    @property
    def name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        self._keys = self._load_keys()
        return bool(self._active_keys(self._keys))

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        vision: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Any]:
        self._keys = self._load_keys()
        active = self._active_keys(self._keys)
        if not active:
            raise LLMKeyExhaustedError("All Groq keys are exhausted or unconfigured.")

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed.")

        use_model = model or (VISION_MODELS["groq"] if vision else DEFAULT_MODELS["groq"])
        last_error: Optional[Exception] = None

        for key in list(active):
            try:
                client = OpenAI(base_url=GROQ_BASE_URL, api_key=key)
                kwargs = {
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                resp = client.chat.completions.create(**kwargs)
                choice_msg = resp.choices[0].message
                return (choice_msg.content or "", getattr(choice_msg, "tool_calls", None))
            except Exception as e:
                if self._is_key_or_quota_error(e):
                    self._mark_exhausted(key)
                    last_error = e
                    continue
                raise RuntimeError(f"Groq API error: {e}") from e

        raise LLMKeyExhaustedError(f"All Groq keys failed. Last: {last_error}")


class ProviderRegistry:
    """Manages multiple LLM providers and executes ordered failover.

    Default priority: Gemini → OpenRouter → NVIDIA NIM
    """

    _PROVIDER_CLASSES: Dict[str, type] = {
        "openrouter": OpenRouterProvider,
        "nvidia": NvidiaProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
    }

    def __init__(self, provider_order: Optional[List[str]] = None) -> None:
        order = provider_order or ["openrouter", "nvidia", "groq", "gemini"]
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
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Any, str, str]:
        """Execute a chat completion with candidate-aware provider failover.

        Args:
            messages:    OpenAI-format messages.
            candidates:  Optional list of (provider_name, model_id, spec) tuples
                         from TaskRouter. When provided, ONLY these candidates are
                         tried (no fallback to the full provider list).
            model:       Optional manual model override (used in Case B only).
            provider:    Optional manual provider override (Case B only).
            temperature: Sampling temperature.
            max_tokens:  Max tokens in completion.
            vision:      If True, enforces vision capability.

        Returns:
            (reply_text, provider_name_used, model_id_used)

        Raises:
            RuntimeError: if all candidate providers fail.
        """
        errors: List[str] = []

        # Case A: Candidates list provided by TaskRouter — use ONLY these
        if candidates:
            for prov_name, model_id, spec in candidates:
                p = self.get_provider(prov_name)
                if not p or not p.is_available():
                    errors.append(f"{prov_name}/{model_id}: provider unconfigured or unavailable")
                    continue
                try:
                    text, tool_calls = p.chat_completion(
                        messages,
                        model=model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        vision=vision,
                        tools=tools,
                    )
                    logger.info("LLM success: provider=%s model=%s tool_calls=%s", prov_name, model_id, bool(tool_calls))
                    return text, tool_calls, prov_name, model_id
                except LLMKeyExhaustedError as e:
                    errors.append(f"{prov_name}/{model_id}: key exhausted — {e}")
                    continue
                except RuntimeError as e:
                    errors.append(f"{prov_name}/{model_id}: {e}")
                    logger.warning("Candidate %s/%s failed: %s", prov_name, model_id, e)
                    continue
            # All candidates exhausted — raise with full error context
            raise RuntimeError(
                "All candidate LLM providers failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        # Case B: Standard provider order fallback (no candidates provided)
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
                text, tool_calls = p.chat_completion(
                    messages,
                    model=use_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    vision=vision,
                    tools=tools,
                )
                used_model = use_model or (
                    VISION_MODELS.get(p.name, "") if vision else DEFAULT_MODELS.get(p.name, "")
                )
                logger.info("LLM response from provider=%s model=%s tool_calls=%s", p.name, used_model, bool(tool_calls))
                return text, tool_calls, p.name, used_model
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
