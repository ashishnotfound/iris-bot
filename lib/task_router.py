"""
lib/task_router.py — Dynamic Task-Based Model Router for Iris / Hermes Agent

Evaluates request complexity, requirements (vision, tools, context, reasoning, coding),
and provider health to select the optimal model & provider for every turn.

Model Tiers:
  - FAST: Simple Q&A, casual chat, quick edits/transformations.
  - BALANCED: Standard reasoning, summaries, research, general coding, multi-turn chat.
  - POWERFUL: Complex reasoning, deep debugging, system architecture, multi-step planning.

Rules:
  - Strict capability check: Never route vision calls to non-vision models.
  - Preference for free models: Only uses zero-cost or free-tier configured models.
  - Tier preservation on failover: Prioritizes equivalent tier models before falling back.
  - Supports manual override (/model <model-id>) and return to automatic mode (/model auto).
"""

from __future__ import annotations

import logging
import os
import re
import site
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure site-packages in virtualenv is loaded
_venv_site = Path(__file__).resolve().parent.parent / "venv" / "Lib" / "site-packages"
if _venv_site.exists():
    site.addsitedir(str(_venv_site))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ModelTier(Enum):
    FAST = auto()
    BALANCED = auto()
    POWERFUL = auto()

    def __str__(self) -> str:
        return self.name.upper()


@dataclass
class ModelSpec:
    provider: str          # "gemini" | "openrouter" | "nvidia"
    model_id: str          # Model identifier
    tier: ModelTier        # FAST | BALANCED | POWERFUL
    free: bool = True      # Free / zero-cost tier
    vision: bool = False   # Multimodal image input support
    tools: bool = True     # Tool / function calling support
    reasoning: bool = False # Enhanced reasoning / math / coding
    context_window: int = 128000
    enabled: bool = True


@dataclass
class RoutingDecision:
    tier: ModelTier
    candidates: List[Tuple[str, str, ModelSpec]]  # [(provider, model_id, spec), ...]
    reason: str
    vision_required: bool = False
    tools_required: bool = False
    is_manual_override: bool = False
    escalated: bool = False


# ---------------------------------------------------------------------------
# Central Model Catalog
# ---------------------------------------------------------------------------

MODEL_CATALOG: List[ModelSpec] = [
    # ── Gemini (Google AI Studio Free Tier) ──
    ModelSpec(
        provider="gemini",
        model_id="gemini-2.5-flash",
        tier=ModelTier.BALANCED,
        free=True,
        vision=True,
        tools=True,
        reasoning=True,
        context_window=1000000,
        enabled=True,
    ),

    # ── OpenRouter (Free Models Catalog) ──
    ModelSpec(
        provider="openrouter",
        model_id="google/gemma-4-31b-it:free",
        tier=ModelTier.BALANCED,
        free=True,
        vision=True,
        tools=True,
        reasoning=True,
        context_window=128000,
        enabled=True,
    ),
    ModelSpec(
        provider="openrouter",
        model_id="nvidia/nemotron-3.5-lightning:free",
        tier=ModelTier.FAST,
        free=True,
        vision=False,
        tools=True,
        reasoning=False,
        context_window=128000,
        enabled=True,
    ),
    ModelSpec(
        provider="openrouter",
        model_id="google/gemini-2.0-flash-exp:free",
        tier=ModelTier.BALANCED,
        free=True,
        vision=True,
        tools=True,
        reasoning=True,
        context_window=1000000,
        enabled=True,
    ),

        # ── Groq Cloud (Free Tier — Ultra-fast & Vision) ──
    ModelSpec(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        tier=ModelTier.BALANCED,
        free=True,
        vision=False,
        tools=True,
        reasoning=True,
        context_window=128000,
        enabled=True,
    ),
    ModelSpec(
        provider="groq",
        model_id="llama-3.2-11b-vision-preview",
        tier=ModelTier.BALANCED,
        free=True,
        vision=True,
        tools=True,
        reasoning=False,
        context_window=128000,
        enabled=True,
    ),
    ModelSpec(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        tier=ModelTier.FAST,
        free=True,
        vision=False,
        tools=True,
        reasoning=False,
        context_window=128000,
        enabled=True,
    ),

    # ── NVIDIA NIM (Free Credits API) ──
    ModelSpec(
        provider="nvidia",
        model_id="meta/llama-3.3-70b-instruct",
        tier=ModelTier.POWERFUL,
        free=True,
        vision=False,
        tools=True,
        reasoning=True,
        context_window=128000,
        enabled=True,
    ),
    ModelSpec(
        provider="nvidia",
        model_id="meta/llama-3.2-90b-vision-instruct",
        tier=ModelTier.POWERFUL,
        free=True,
        vision=True,
        tools=True,
        reasoning=True,
        context_window=128000,
        enabled=True,
    ),
    ModelSpec(
        provider="nvidia",
        model_id="meta/llama-3.1-8b-instruct",
        tier=ModelTier.FAST,
        free=True,
        vision=False,
        tools=True,
        reasoning=False,
        context_window=128000,
        enabled=True,
    ),
    ModelSpec(
        provider="nvidia",
        model_id="microsoft/phi-3-vision-128k-instruct",
        tier=ModelTier.BALANCED,
        free=True,
        vision=True,
        tools=False,
        reasoning=False,
        context_window=128000,
        enabled=True,
    ),
]


# ---------------------------------------------------------------------------
# Task Classifier
# ---------------------------------------------------------------------------

# Keywords indicating complex coding/debugging or architecture tasks
CODING_KEYWORDS = re.compile(
    r"\b(def|class|function|async|await|refactor|architecture|debug|fix|bug|solve|"
    r"traceback|exception|stacktrace|sql|database|schema|api|endpoint|script|"
    r"algorithm|optimize|performance|concurrency|thread|regex)\b",
    re.IGNORECASE,
)

# Keywords indicating deep reasoning / analytical tasks
REASONING_KEYWORDS = re.compile(
    r"\b(analyze|compare|evaluate|explain|why|pros and cons|strategy|plan|"
    r"architecture|tradeoffs|design pattern|step by step|detailed review|"
    r"critique|proof|calculate|derive)\b",
    re.IGNORECASE,
)

# Keywords for simple casual chat or quick edits
SIMPLE_KEYWORDS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|good morning|good evening|who are you|"
    r"what is your name|bye|ok|okay|cool|awesome|format this|capitalize|spell check)\b",
    re.IGNORECASE,
)


class TaskClassifier:
    """Classifies user request parameters into an appropriate ModelTier."""

    @staticmethod
    def classify(
        message: str,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        has_photo: bool = False,
        tools_available: bool = False,
    ) -> Tuple[ModelTier, str]:
        text = (message or "").strip()
        word_count = len(text.split())

        # 1. Vision tasks -> BALANCED or POWERFUL (must be vision capable)
        if has_photo:
            if word_count > 100 or CODING_KEYWORDS.search(text):
                return ModelTier.POWERFUL, "Vision with complex analysis/coding request"
            return ModelTier.BALANCED, "Vision understanding request"

        # 2. Simple casual conversation or quick formatting (only if tools are NOT active)
        if word_count < 15 and SIMPLE_KEYWORDS.search(text) and not CODING_KEYWORDS.search(text) and not tools_available:
            return ModelTier.FAST, "Casual chat or simple question"

        # 3. Complex coding, debugging, or system architecture
        coding_matches = len(CODING_KEYWORDS.findall(text))
        if coding_matches >= 2 or ("```" in text and coding_matches >= 1) or "bug" in text.lower() or "error" in text.lower():
            return ModelTier.POWERFUL, "Complex coding, debugging, or architecture request"

        # 4. Detailed analytical or multi-step reasoning
        reasoning_matches = len(REASONING_KEYWORDS.findall(text))
        if reasoning_matches >= 2 or word_count > 150:
            return ModelTier.POWERFUL, "Deep analytical or long-form reasoning request"

        # 5. Moderate complexity / default
        if word_count > 40 or reasoning_matches >= 1 or coding_matches >= 1 or tools_available:
            return ModelTier.BALANCED, "Moderate task complexity or tool execution active"

        return ModelTier.FAST, "Standard simple query"


# ---------------------------------------------------------------------------
# Task Router Engine
# ---------------------------------------------------------------------------

class TaskRouter:
    """Dynamic task router for Iris Agent."""

    def __init__(self, catalog: Optional[List[ModelSpec]] = None) -> None:
        self.catalog: List[ModelSpec] = catalog or MODEL_CATALOG

    def route(
        self,
        message: str,
        *,
        history: Optional[List[Dict[str, Any]]] = None,
        has_photo: bool = False,
        tools_available: bool = False,
        manual_model_override: Optional[str] = None,
        active_providers: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """Select a prioritized candidate list of (provider, model_id, spec) tuples.

        Args:
            message:               User message text.
            history:               Conversation history turns.
            has_photo:             True if image attached.
            tools_available:       True if Composio/app tools active.
            manual_model_override: Optional pinned model string, e.g. "gemini-2.5-flash"
                                   or "openrouter/google/gemma-4-31b-it:free".
                                   If "auto" or empty/None, dynamic routing is used.
            active_providers:      List of currently configured provider names.

        Returns:
            RoutingDecision telemetry struct.
        """
        if active_providers is not None:
            active = set(active_providers)
        else:
            active = {"openrouter", "nvidia", "groq", "gemini"}

        # ── 1. Check Manual Override ──
        if manual_model_override and manual_model_override.strip().lower() not in ("auto", ""):
            target = manual_model_override.strip()
            # Parse optional "provider/model_id"
            if "/" in target and not target.startswith("google/"):
                prov, model_id = target.split("/", 1)
            else:
                prov, model_id = None, target

            # Lookup spec or build dynamic fallback spec
            matching_specs = [
                s for s in self.catalog
                if (prov is None or s.provider == prov) and s.model_id.lower() == model_id.lower()
            ]

            if matching_specs:
                spec = matching_specs[0]
            else:
                p = prov or ("openrouter" if "/" in target else "gemini")
                spec = ModelSpec(
                    provider=p,
                    model_id=model_id,
                    tier=ModelTier.BALANCED,
                    free=True,
                    vision=has_photo,
                    tools=tools_available,
                )

            candidates = [(spec.provider, spec.model_id, spec)]
            return RoutingDecision(
                tier=spec.tier,
                candidates=candidates,
                reason=f"Manual override pinned to {spec.model_id}",
                vision_required=has_photo,
                tools_required=tools_available,
                is_manual_override=True,
            )

        # ── 2. Dynamic Task Classification ──
        target_tier, reason = TaskClassifier.classify(
            message,
            history=history,
            has_photo=has_photo,
            tools_available=tools_available,
        )

        # ── 3. Candidate Filtering & Ranking ──
        eligible: List[ModelSpec] = []
        for spec in self.catalog:
            if not spec.enabled:
                continue
            if spec.provider not in active:
                continue
            if has_photo and not spec.vision:
                continue  # NEVER route vision requests to non-vision models
            if tools_available and not spec.tools:
                continue
            if not spec.free:
                continue  # Prefer free models only
            eligible.append(spec)

        if not eligible:
            # Emergency fallback: relax tool constraint if no exact match
            eligible = [
                s for s in self.catalog
                if s.enabled and s.provider in active and (not has_photo or s.vision)
            ]

        # ── 4. Tier Preference Sorting ──
        # Priority order: exact tier match -> next tier match -> provider order (gemini -> openrouter -> nvidia)
        tier_weights = {
            ModelTier.POWERFUL: {ModelTier.POWERFUL: 0, ModelTier.BALANCED: 1, ModelTier.FAST: 2},
            ModelTier.BALANCED: {ModelTier.BALANCED: 0, ModelTier.POWERFUL: 1, ModelTier.FAST: 2},
            ModelTier.FAST:     {ModelTier.FAST: 0, ModelTier.BALANCED: 1, ModelTier.POWERFUL: 2},
        }
        prov_weights = {"openrouter": 0, "nvidia": 1, "groq": 2, "gemini": 3}

        def _sort_key(spec: ModelSpec) -> Tuple[int, int]:
            prov_dist = prov_weights.get(spec.provider, 9)
            tier_dist = tier_weights[target_tier].get(spec.tier, 9)
            return (prov_dist, tier_dist)

        sorted_specs = sorted(eligible, key=_sort_key)
        candidates = [(s.provider, s.model_id, s) for s in sorted_specs]

        # Deduplicate candidates by (provider, model_id)
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c[0], c[1])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        candidates = unique_candidates

        logger.info(
            "TaskRouter: classified tier=%s reason=%r candidates=%s",
            target_tier, reason, [(c[0], c[1]) for c in candidates[:3]],
        )

        return RoutingDecision(
            tier=target_tier,
            candidates=candidates,
            reason=reason,
            vision_required=has_photo,
            tools_required=tools_available,
            is_manual_override=False,
        )

    def should_escalate(self, reply_text: str, current_tier: ModelTier) -> bool:
        """Detect if a FAST or BALANCED response indicates insufficient reasoning."""
        if current_tier == ModelTier.POWERFUL:
            return False  # Already at highest tier

        text = (reply_text or "").lower()
        escalation_signals = [
            "i am a language model",
            "i cannot solve",
            "too complex for me",
            "as a fast model",
            "insufficient reasoning",
            "i apologize, but i am unable to complete this complex",
        ]
        return any(sig in text for sig in escalation_signals)
