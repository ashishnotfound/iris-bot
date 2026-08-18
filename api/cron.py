"""
api/cron.py - Vercel Python Serverless HTTP Entrypoint for Scheduled Cron Trigger

Invoked on a 1-minute schedule by Vercel Cron.
Validates Authorization: Bearer <CRON_SECRET> header, queries due jobs from Supabase,
and executes agent turns.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from lib.cron_manager import run_due_jobs

logger = logging.getLogger(__name__)


def handler(request):
    """Vercel Python serverless HTTP entrypoint for Vercel Cron GET."""
    headers = getattr(request, "headers", {})
    if not isinstance(headers, dict):
        headers = {}

    auth_header = headers.get("Authorization") or headers.get("authorization") or ""
    expected_secret = os.environ.get("CRON_SECRET", "").strip()

    if expected_secret:
        expected_bearer = f"Bearer {expected_secret}"
        if auth_header != expected_bearer:
            return {
                "statusCode": 401,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Unauthorized cron request"}),
            }

    try:
        results = run_due_jobs()
        ran_list = results.get("ran", [])
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "status": "success",
                "jobs_run": len(ran_list),
                "details": results,
            }),
        }
    except Exception as e:
        logger.error("Cron execution failed: %s", e, exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "error", "error": str(e)}),
        }
