"""
lib/automation_parser.py — Natural-Language to Automation Converter

Converts user intent like:
  "Every day at 9 AM post this to Instagram and X"
  "Every Friday tell me what products sold the most"
  "Check Amazon at 8 PM daily"

into a structured automation record suitable for the cron_jobs table.

Relies on the LLM to do the NL understanding; only basic validation
and duplicate detection are done in Python.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt for NL → automation
# ---------------------------------------------------------------------------

_PARSE_PROMPT = """\
You are Iris, a scheduling assistant. Convert the user's request into a structured automation.

User's request: {user_request}

Return ONLY valid JSON (no markdown fences) with this exact schema:
{{
  "cron_expression": "M H D Mo DoW",   // 5-field POSIX cron (UTC)
  "timezone": "UTC",                   // IANA timezone e.g. "Asia/Kolkata"
  "task_description": "...",           // what Iris should do when this runs (natural language)
  "action_type": "...",                // one of: social_post, business_sync, report, reminder, custom
  "summary": "..."                     // short human-readable summary e.g. "Daily 9 AM Instagram post"
}}

Rules:
- cron_expression must be valid 5-field cron in the specified timezone (convert to UTC if needed)
- If the user says "daily at 9 AM IST", convert 9 AM IST (UTC+5:30) to 3:30 AM UTC → "30 3 * * *"
- task_description must be actionable: what should actually be done
- action_type must be exactly one of: social_post, business_sync, report, reminder, custom
- If the request is not schedulable (no time/frequency mentioned), return {{"error": "not_schedulable", "reason": "..."}}
- Return ONLY the JSON object. No explanation.
"""

_DEDUP_PROMPT = """\
You are comparing two automation descriptions to check if they are duplicates.

Automation A: {a}
Automation B: {b}

Are these the same automation (same schedule AND same action)?
Answer with ONLY "yes" or "no".
"""

# ---------------------------------------------------------------------------
# AutomationParser
# ---------------------------------------------------------------------------


class ParsedAutomation:
    """Result of a successful NL → automation parse."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.cron_expression: str = data["cron_expression"]
        self.timezone: str = data.get("timezone", "UTC")
        self.task_description: str = data["task_description"]
        self.action_type: str = data.get("action_type", "custom")
        self.summary: str = data.get("summary", self.task_description[:80])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cron_expression": self.cron_expression,
            "timezone": self.timezone,
            "task_description": self.task_description,
            "action_type": self.action_type,
            "summary": self.summary,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ParsedAutomation {self.cron_expression!r} → {self.summary!r}>"


class AutomationParseError(ValueError):
    """Raised when the user's request cannot be converted to an automation."""


class AutomationParser:
    """Converts natural language descriptions into schedulable automations."""

    def __init__(self, llm_callable: Callable[[str], str]) -> None:
        """
        Args:
            llm_callable: A function that takes a prompt string and returns
                          the LLM's text response. Typically wraps
                          ProviderRegistry.chat_completion.
        """
        self._llm = llm_callable

    def parse(self, user_request: str) -> ParsedAutomation:
        """Parse a natural-language automation request.

        Args:
            user_request: Raw user text describing what they want scheduled.

        Returns:
            ParsedAutomation instance.

        Raises:
            AutomationParseError: If the request is not schedulable or
                                   LLM response is invalid.
        """
        prompt = _PARSE_PROMPT.format(user_request=user_request.strip())

        try:
            raw = self._llm(prompt)
        except Exception as e:
            raise AutomationParseError(f"LLM call failed during parse: {e}") from e

        data = _extract_json(raw)

        if "error" in data:
            raise AutomationParseError(
                f"Cannot create automation: {data.get('reason', data['error'])}"
            )

        # Validate required fields
        missing = [f for f in ("cron_expression", "task_description") if not data.get(f)]
        if missing:
            raise AutomationParseError(
                f"LLM parse result missing required fields: {missing}. Raw: {raw[:200]}"
            )

        # Validate cron expression basic structure
        cron = data["cron_expression"].strip()
        if not _is_valid_cron(cron):
            raise AutomationParseError(
                f"LLM returned invalid cron expression: {cron!r}"
            )

        # Normalize action_type
        valid_types = {"social_post", "business_sync", "report", "reminder", "custom"}
        data["action_type"] = data.get("action_type", "custom")
        if data["action_type"] not in valid_types:
            data["action_type"] = "custom"

        return ParsedAutomation(data)

    def is_duplicate(
        self,
        new_task: str,
        new_cron: str,
        existing_jobs: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Check if a new automation duplicates an existing one.

        Args:
            new_task:      New automation's task description.
            new_cron:      New automation's cron expression.
            existing_jobs: List of existing cron_job dicts from Supabase.

        Returns:
            job_id of the duplicate if found, else None.
        """
        for job in existing_jobs:
            # Fast check: same cron expression
            if job.get("cron_expression") != new_cron:
                continue

            # Exact string match
            existing_task = job.get("task_description", "")
            if existing_task.strip().lower() == new_task.strip().lower():
                return job["job_id"]

            # LLM semantic match (only when cron already matches)
            try:
                prompt = _DEDUP_PROMPT.format(a=new_task, b=existing_task)
                answer = self._llm(prompt).strip().lower()
                if answer.startswith("yes"):
                    return job["job_id"]
            except Exception as e:
                logger.warning("Dedup LLM call failed: %s", e)
                # On failure, fall through (allow creation)

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON from LLM output."""
    import json, re

    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract first JSON object with regex as last resort
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise AutomationParseError(
            f"LLM did not return valid JSON. Raw output: {raw[:300]}"
        ) from e


_CRON_FIELD_RE = re.compile(
    r"^(\*|[0-9,\-*/]+)\s+"   # minute
    r"(\*|[0-9,\-*/]+)\s+"    # hour
    r"(\*|[0-9,\-*/]+)\s+"    # day of month
    r"(\*|[0-9,\-*/]+)\s+"    # month
    r"(\*|[0-9,\-*/]+)$"      # day of week
)


def _is_valid_cron(expr: str) -> bool:
    """Basic structural validation for a 5-field cron expression."""
    return bool(_CRON_FIELD_RE.match(expr.strip()))
