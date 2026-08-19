"""
lib/job_runner.py — Reliable Cron Job Executor

Wraps cron job execution with:
  - Running-lock guard (prevents duplicate concurrent executions)
  - Retry logic with exponential backoff stored in Supabase
  - Composio action idempotency (action_log dedup key)
  - Structured Telegram notifications on success / partial failure / error
  - Job timeout detection (jobs stuck in 'running' for > N minutes)

Vercel Compatibility:
  All state (lock, retry count, result) lives in Supabase.
  No in-process state is assumed to survive between invocations.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Job is considered stuck if it has been "running" for longer than this
JOB_TIMEOUT_MINUTES = 8   # Vercel max function timeout is typically 10s–5m

# Retry backoff minutes: 5m, 15m, 60m
RETRY_BACKOFF_MINUTES = [5, 15, 60]


# ---------------------------------------------------------------------------
# Supabase helpers (shared pattern — all Supabase access is in one place)
# ---------------------------------------------------------------------------


def _sb_url() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/")


def _sb_headers() -> Dict[str, str]:
    key = (
        os.environ.get("SUPABASE_SERVICE_KEY", "")
        or os.environ.get("SUPABASE_KEY", "")
    ).strip()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _sb_patch(table: str, filters: Dict[str, str], body: Dict[str, Any]) -> bool:
    import requests
    base = _sb_url()
    if not base:
        return False
    try:
        params = {k: v for k, v in filters.items()}
        r = requests.patch(
            f"{base}/rest/v1/{table}",
            headers=_sb_headers(),
            params=params,
            json=body,
            timeout=6,
        )
        return r.status_code in (200, 204)
    except Exception as e:
        logger.error("_sb_patch %s failed: %s", table, e)
        return False


def _sb_post(table: str, body: Dict[str, Any], prefer: str = "") -> Optional[Dict]:
    import requests
    base = _sb_url()
    if not base:
        return None
    headers = {**_sb_headers()}
    if prefer:
        headers["Prefer"] = prefer
    try:
        r = requests.post(f"{base}/rest/v1/{table}", headers=headers, json=body, timeout=6)
        if r.status_code in (200, 201, 204):
            return r.json() if r.text else {}
        logger.warning("_sb_post %s HTTP %d: %s", table, r.status_code, r.text[:200])
        return None
    except Exception as e:
        logger.error("_sb_post %s failed: %s", table, e)
        return None


def _sb_get(table: str, params: Dict[str, str]) -> List[Dict]:
    import requests
    base = _sb_url()
    if not base:
        return []
    try:
        r = requests.get(
            f"{base}/rest/v1/{table}",
            headers=_sb_headers(),
            params=params,
            timeout=6,
        )
        return r.json() if r.status_code == 200 else []
    except Exception as e:
        logger.error("_sb_get %s failed: %s", table, e)
        return []


# ---------------------------------------------------------------------------
# Running-lock helpers
# ---------------------------------------------------------------------------


def _acquire_lock(job_id: str) -> bool:
    """Attempt to mark a job as running.

    Returns True if the lock was acquired (job was not already running),
    False if the job is already running (duplicate invocation).
    """
    now_utc = datetime.now(timezone.utc)
    timeout_threshold = now_utc - timedelta(minutes=JOB_TIMEOUT_MINUTES)

    # Check if job is already running and not timed out
    rows = _sb_get(
        "cron_jobs",
        {"job_id": f"eq.{job_id}", "select": "running_since,enabled"},
    )
    if not rows:
        return False

    row = rows[0]
    running_since = row.get("running_since")
    if running_since:
        try:
            rs = datetime.fromisoformat(running_since.replace("Z", "+00:00"))
            if rs > timeout_threshold:
                logger.warning(
                    "Job %s already running since %s — skipping duplicate invocation",
                    job_id[:8],
                    running_since,
                )
                return False
            else:
                logger.warning(
                    "Job %s appears stuck (running since %s) — forcibly taking over",
                    job_id[:8],
                    running_since,
                )
        except Exception:
            pass

    # Mark as running
    ok = _sb_patch(
        "cron_jobs",
        {"job_id": f"eq.{job_id}"},
        {"running_since": now_utc.isoformat()},
    )
    return ok


def _release_lock(
    job_id: str,
    *,
    result: Optional[str] = None,
    error: Optional[str] = None,
    retry_count: int = 0,
    next_run_at: Optional[str] = None,
) -> None:
    """Clear the running lock and write result/error to Supabase."""
    body: Dict[str, Any] = {"running_since": None}
    if result is not None:
        body["last_result"] = result[:2000] if result else None
    if error is not None:
        body["last_error"] = error[:1000] if error else None
        body["retry_count"] = retry_count
    if next_run_at:
        body["next_run_at"] = next_run_at
        body["last_run_at"] = datetime.now(timezone.utc).isoformat()
    _sb_patch("cron_jobs", {"job_id": f"eq.{job_id}"}, body)


# ---------------------------------------------------------------------------
# Action idempotency
# ---------------------------------------------------------------------------


def make_dedup_key(job_id: str, action_type: str, date_str: Optional[str] = None) -> str:
    """Create a deterministic dedup key for a Composio action execution."""
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = f"{job_id}:{action_type}:{date_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def is_action_already_done(dedup_key: str) -> bool:
    """Check if this action has already succeeded (idempotency check)."""
    rows = _sb_get(
        "action_log",
        {"dedup_key": f"eq.{dedup_key}", "status": "eq.success", "select": "dedup_key"},
    )
    return bool(rows)


def record_action(
    dedup_key: str,
    *,
    chat_id: int,
    job_id: Optional[str],
    action_type: str,
    status: str,
    request_data: Optional[Dict] = None,
    response_data: Optional[Dict] = None,
    error: Optional[str] = None,
) -> None:
    """Upsert an action log entry for idempotency tracking."""
    _sb_post(
        "action_log",
        {
            "dedup_key": dedup_key,
            "chat_id": chat_id,
            "job_id": job_id,
            "action_type": action_type,
            "status": status,
            "request_data": request_data or {},
            "response_data": response_data or {},
            "error": error,
        },
        prefer="resolution=merge-duplicates",
    )


# ---------------------------------------------------------------------------
# Structured Telegram notification
# ---------------------------------------------------------------------------


def send_job_notification(
    tg,
    chat_id: int,
    *,
    job_summary: str,
    next_run_at: Optional[str],
    results: List[Dict[str, Any]],   # [{"action": str, "status": "ok"|"fail", "detail": str}]
    failed: bool,
) -> None:
    """Send a structured success/failure notification to the user after a job runs."""
    if not failed:
        header = "✅ *Automation Complete*"
    else:
        all_failed = all(r.get("status") == "fail" for r in results)
        header = "🔴 *Automation Failed*" if all_failed else "⚠️ *Automation Partially Failed*"

    lines = [header, f"_{job_summary}_", ""]

    for r in results:
        icon = "✅" if r.get("status") == "ok" else "❌"
        action = r.get("action", "Action")
        detail = r.get("detail", "")
        lines.append(f"{icon} *{action}:* {detail}")

    if next_run_at:
        try:
            nxt = datetime.fromisoformat(next_run_at.replace("Z", "+00:00"))
            lines.append(f"\n⏰ *Next run:* {nxt.strftime('%Y-%m-%d %H:%M UTC')}")
        except Exception:
            pass

    msg = "\n".join(lines)
    tg.send_message(chat_id, msg, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Compute next retry time
# ---------------------------------------------------------------------------


def next_retry_at(retry_count: int) -> str:
    """Compute the next retry time based on retry_count (exponential backoff)."""
    idx = min(retry_count, len(RETRY_BACKOFF_MINUTES) - 1)
    minutes = RETRY_BACKOFF_MINUTES[idx]
    nxt = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return nxt.isoformat()
