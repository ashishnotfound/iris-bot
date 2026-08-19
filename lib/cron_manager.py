"""
lib/cron_manager.py — Autonomous Cron Job Runner (Phase 2)

Queries the Supabase `cron_jobs` table for jobs that are due to run,
executes each as a full Iris agent turn, sends structured results to
Telegram, and updates the next scheduled run time.

Phase 2 additions:
  - Running-lock guard (prevents duplicate concurrent executions)
  - Retry with exponential backoff on failure
  - Structured success/failure Telegram notifications
  - Idempotency tracking via action_log table
  - Job cancellation

Called by: Vercel Cron endpoint (GET /api/cron, every minute)

Requires: croniter   pip install croniter
          requests   pip install requests
          SUPABASE_URL + SUPABASE_SERVICE_KEY
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cron expression helpers
# ---------------------------------------------------------------------------


def next_run_for(cron_expression: str) -> str:
    """Calculate the next run time for a cron expression.

    Returns an ISO-8601 UTC string suitable for Supabase TIMESTAMPTZ.
    Falls back to "now + 60s" if croniter is unavailable or invalid.
    """
    try:
        from croniter import croniter

        now = datetime.now(timezone.utc)
        cron = croniter(cron_expression, now)
        nxt = cron.get_next(datetime)
        return nxt.isoformat()
    except ImportError:
        logger.warning("croniter not installed; defaulting next_run to now+60s")
        from datetime import timedelta
        return (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    except Exception as e:
        logger.error("Invalid cron expression %r: %s", cron_expression, e)
        from datetime import timedelta
        return (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()


def is_due(next_run_at: str) -> bool:
    """Return True if the job's next_run_at is at or before now."""
    try:
        nxt = datetime.fromisoformat(next_run_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= nxt
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------


def _supabase_url() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/")


def _supabase_headers() -> Dict[str, str]:
    key = (
        os.environ.get("SUPABASE_SERVICE_KEY", "")
        or os.environ.get("SUPABASE_KEY", "")
    ).strip()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _fetch_due_jobs() -> List[Dict[str, Any]]:
    """Fetch all enabled cron jobs whose next_run_at is <= now."""
    import requests

    base = _supabase_url()
    if not base:
        return []
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        r = requests.get(
            f"{base}/rest/v1/cron_jobs",
            headers=_supabase_headers(),
            params={
                "enabled": "eq.true",
                "next_run_at": f"lte.{now_iso}",
                "select": (
                    "job_id,chat_id,cron_expression,task_description,"
                    "timezone,retry_count,max_retries,action_type,action_params,"
                    "running_since"
                ),
            },
            timeout=8,
        )
        if r.status_code == 200:
            return r.json()
        else:
            logger.error("fetch_due_jobs HTTP %d: %s", r.status_code, r.text[:256])
    except Exception as e:
        logger.error("fetch_due_jobs error: %s", e)
    return []


def _update_next_run(job_id: str, cron_expression: str) -> None:
    """Compute and persist the next run time for a job after it executes."""
    import requests

    base = _supabase_url()
    if not base:
        return
    try:
        nxt = next_run_for(cron_expression)
        r = requests.patch(
            f"{base}/rest/v1/cron_jobs",
            headers=_supabase_headers(),
            params={"job_id": f"eq.{job_id}"},
            json={
                "last_run_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": nxt,
                "running_since": None,  # clear lock
            },
            timeout=6,
        )
        if r.status_code not in (200, 204):
            logger.error("update_next_run HTTP %d for job %s", r.status_code, job_id)
    except Exception as e:
        logger.error("update_next_run error for job %s: %s", job_id, e)


def _disable_job(job_id: str, error: str) -> None:
    """Disable a job after max retries are exhausted."""
    import requests

    base = _supabase_url()
    if not base:
        return
    try:
        requests.patch(
            f"{base}/rest/v1/cron_jobs",
            headers=_supabase_headers(),
            params={"job_id": f"eq.{job_id}"},
            json={
                "enabled": False,
                "last_error": error[:1000],
                "running_since": None,
                "last_run_at": datetime.now(timezone.utc).isoformat(),
            },
            timeout=6,
        )
        logger.warning("Job %s disabled after max retries: %s", job_id[:8], error[:200])
    except Exception as e:
        logger.error("Failed to disable job %s: %s", job_id[:8], e)


def _schedule_retry(job_id: str, retry_count: int, error: str) -> None:
    """Schedule a job retry with exponential backoff."""
    import requests
    from lib.job_runner import next_retry_at

    base = _supabase_url()
    if not base:
        return
    try:
        retry_at = next_retry_at(retry_count)
        requests.patch(
            f"{base}/rest/v1/cron_jobs",
            headers=_supabase_headers(),
            params={"job_id": f"eq.{job_id}"},
            json={
                "retry_count": retry_count + 1,
                "last_error": error[:1000],
                "next_run_at": retry_at,
                "running_since": None,
            },
            timeout=6,
        )
        logger.info(
            "Job %s retry #%d scheduled at %s", job_id[:8], retry_count + 1, retry_at
        )
    except Exception as e:
        logger.error("_schedule_retry failed for job %s: %s", job_id[:8], e)


def _reset_retry_count(job_id: str) -> None:
    """Reset retry_count to 0 after a successful run."""
    import requests

    base = _supabase_url()
    if not base:
        return
    try:
        requests.patch(
            f"{base}/rest/v1/cron_jobs",
            headers=_supabase_headers(),
            params={"job_id": f"eq.{job_id}"},
            json={"retry_count": 0, "last_error": None},
            timeout=6,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_due_jobs() -> Dict[str, Any]:
    """Check for due cron jobs and execute each one reliably.

    Returns:
        {"ran": [job_id, ...], "skipped": [job_id, ...], "errors": {job_id: msg, ...}}

    Called by the Vercel cron endpoint (/api/cron) every minute.
    """
    jobs = _fetch_due_jobs()
    if not jobs:
        logger.debug("cron_manager: no jobs due")
        return {"ran": [], "skipped": [], "errors": {}}

    # Import here to avoid circular dependency at module level
    from lib.hermes_runner import execute_agent_turn
    from lib.job_runner import (
        _acquire_lock,
        _release_lock,
        send_job_notification,
    )
    from lib.telegram_client import TelegramClient

    tg = TelegramClient()
    ran: List[str] = []
    skipped: List[str] = []
    errors: Dict[str, str] = {}

    for job in jobs:
        job_id = job["job_id"]
        chat_id = int(job["chat_id"])
        cron_expr = job["cron_expression"]
        task = job["task_description"]
        retry_count = int(job.get("retry_count") or 0)
        max_retries = int(job.get("max_retries") or 3)
        job_summary = task[:100]

        logger.info(
            "Processing cron job %s for chat_id=%s: %s (retry=%d)",
            job_id[:8], chat_id, task[:60], retry_count,
        )

        # ── Running-lock guard: prevents duplicate concurrent executions ──
        if not _acquire_lock(job_id):
            logger.info("Job %s is already running — skipping this invocation", job_id[:8])
            skipped.append(job_id)
            continue

        try:
            # Notify user that the scheduled job is starting
            tg.send_message(
                chat_id,
                f"⏰ *Scheduled Task Running*\n_{task[:100]}_",
                parse_mode="Markdown",
            )

            # Execute as a full agent turn
            result = execute_agent_turn(
                chat_id,
                task,
                telegram_client=tg,
            )

            if result.get("status") == "success":
                ran.append(job_id)
                _reset_retry_count(job_id)
                _update_next_run(job_id, cron_expr)

                # Structured success notification
                next_run = next_run_for(cron_expr)
                send_job_notification(
                    tg,
                    chat_id,
                    job_summary=job_summary,
                    next_run_at=next_run,
                    results=[{"action": "Task", "status": "ok", "detail": "Completed successfully"}],
                    failed=False,
                )
            else:
                err = result.get("error", "agent returned non-success status")
                errors[job_id] = err
                _handle_failure(
                    job_id, chat_id, cron_expr, err,
                    retry_count, max_retries, job_summary, tg
                )

        except Exception as e:
            err_str = str(e)
            errors[job_id] = err_str
            logger.exception("Cron job %s raised an exception: %s", job_id[:8], err_str)
            _handle_failure(
                job_id, chat_id, cron_expr, err_str,
                retry_count, max_retries, job_summary, tg
            )

    logger.info(
        "cron_manager: ran=%d skipped=%d errors=%d",
        len(ran), len(skipped), len(errors),
    )
    return {"ran": ran, "skipped": skipped, "errors": errors}


def _handle_failure(
    job_id: str,
    chat_id: int,
    cron_expr: str,
    error: str,
    retry_count: int,
    max_retries: int,
    job_summary: str,
    tg,
) -> None:
    """Handle a job failure: retry or disable, and notify user."""
    from lib.job_runner import send_job_notification, next_retry_at

    if retry_count < max_retries:
        retry_at = next_retry_at(retry_count)
        _schedule_retry(job_id, retry_count, error)
        tg.send_message(
            chat_id,
            f"⚠️ *Scheduled task failed* (attempt {retry_count + 1}/{max_retries + 1})\n"
            f"_{job_summary}_\n\n"
            f"Error: {error[:200]}\n"
            f"Retrying automatically...",
            parse_mode="Markdown",
        )
    else:
        # Max retries exhausted — disable the job
        _disable_job(job_id, error)
        send_job_notification(
            tg,
            chat_id,
            job_summary=job_summary,
            next_run_at=None,
            results=[{"action": "Task", "status": "fail", "detail": error[:300]}],
            failed=True,
        )
        tg.send_message(
            chat_id,
            f"🔴 *Automation disabled* — failed {max_retries + 1} times consecutively.\n"
            f"Use `/cron list` to see your jobs and `/cron add` to re-enable it.",
            parse_mode="Markdown",
        )
