"""
lib/cron_manager.py — Autonomous Cron Job Runner

Queries the Supabase `cron_jobs` table for jobs that are due to run,
executes each as a full Iris agent turn, sends results to Telegram, and
updates the next scheduled run time.

This module is designed to be called by a Vercel Cron endpoint
(e.g., GET /api/cron, triggered every minute).

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

    Requires the `croniter` package (pip install croniter).
    Falls back to "now + 60s" if croniter is unavailable.
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
                "select": "job_id,chat_id,cron_expression,task_description",
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
            },
            timeout=6,
        )
        if r.status_code not in (200, 204):
            logger.error("update_next_run HTTP %d for job %s", r.status_code, job_id)
    except Exception as e:
        logger.error("update_next_run error for job %s: %s", job_id, e)


def _mark_job_error(job_id: str, error: str) -> None:
    """Disable a job that has repeatedly failed to prevent spam."""
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
                "last_run_at": datetime.now(timezone.utc).isoformat(),
                # Still advance next_run so it doesn't hammer on every poll
                "next_run_at": next_run_for("*/5 * * * *"),
            },
            timeout=6,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_due_jobs() -> Dict[str, Any]:
    """Check for due cron jobs and execute each one.

    Returns:
        {"ran": [job_id, ...], "errors": {job_id: error_msg, ...}}

    Called by the Vercel cron endpoint (/api/cron) every minute.
    """
    jobs = _fetch_due_jobs()
    if not jobs:
        logger.debug("cron_manager: no jobs due")
        return {"ran": [], "errors": {}}

    # Import here to avoid circular dependency at module level
    from lib.hermes_runner import execute_agent_turn
    from lib.telegram_client import TelegramClient

    tg = TelegramClient()
    ran: List[str] = []
    errors: Dict[str, str] = {}

    for job in jobs:
        job_id = job["job_id"]
        chat_id = int(job["chat_id"])
        cron_expr = job["cron_expression"]
        task = job["task_description"]

        logger.info("Running cron job %s for chat_id=%s: %s", job_id[:8], chat_id, task[:60])

        try:
            # Notify the user the scheduled job is starting
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
            else:
                err = result.get("error", "unknown error")
                errors[job_id] = err
                logger.warning("Cron job %s failed: %s", job_id[:8], err)

        except Exception as e:
            err_str = str(e)
            errors[job_id] = err_str
            logger.exception("Cron job %s raised an exception: %s", job_id[:8], err_str)
            try:
                tg.send_message(chat_id, f"⚠️ Scheduled task failed: {err_str[:200]}")
            except Exception:
                pass

        finally:
            # Always advance next_run_at so the job isn't re-executed immediately
            _update_next_run(job_id, cron_expr)

    logger.info("cron_manager: ran=%d errors=%d", len(ran), len(errors))
    return {"ran": ran, "errors": errors}
