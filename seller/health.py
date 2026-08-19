"""Persistent connection health system for seller integrations."""

from datetime import datetime, timezone
import json
import logging
import os
import sqlite3
import threading
from typing import Dict, Optional

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home
from seller.base import ErrorCategory, IntegrationState, SellerHealthStatus

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SellerHealthMonitor:
    """Persistent connection health tracking system for IRIS seller integrations.

    Maintains integration health states (CONNECTED, DEGRADED, AUTHENTICATION_REQUIRED,
    RATE_LIMITED, TEMPORARILY_UNAVAILABLE, OFFLINE) in SQLite, tracking failure counts,
    error categories, retry state, and heartbeat timestamps without exposing credentials.
    """

    _instance: Optional["SellerHealthMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._init_db(db_path)
                cls._instance = instance
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    def _init_db(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = db_path
        else:
            home = get_hermes_home()
            self.db_path = os.path.join(home, "seller_health.db")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._db_lock = threading.Lock()

        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seller_health (
                    marketplace TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    last_successful_request TEXT,
                    last_failed_request TEXT,
                    last_health_check TEXT,
                    failure_count INTEGER DEFAULT 0,
                    last_error_category TEXT,
                    last_error_message TEXT,
                    token_status TEXT DEFAULT 'VALID',
                    retry_state TEXT DEFAULT 'IDLE',
                    last_heartbeat TEXT
                )
            """)
            conn.commit()

    def update_success(self, marketplace: str):
        """Record a successful API request or health check."""
        now = _now_iso()
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO seller_health (
                    marketplace, state, last_successful_request, last_health_check,
                    failure_count, last_error_category, last_error_message, retry_state
                ) VALUES (?, ?, ?, ?, 0, ?, NULL, 'IDLE')
                ON CONFLICT(marketplace) DO UPDATE SET
                    state = ?,
                    last_successful_request = ?,
                    last_health_check = ?,
                    failure_count = 0,
                    last_error_message = NULL,
                    retry_state = 'IDLE'
            """, (
                marketplace, IntegrationState.CONNECTED.value, now, now, ErrorCategory.UNKNOWN.value,
                IntegrationState.CONNECTED.value, now, now
            ))
            conn.commit()

    def update_failure(self, marketplace: str, category: ErrorCategory, message: str):
        """Record an API request failure and transition state based on failure category."""
        now = _now_iso()
        safe_msg = redact_sensitive_text(str(message))[:500]

        # Determine target integration state
        if category == ErrorCategory.AUTHENTICATION or category == ErrorCategory.INVALID_CREDENTIALS:
            new_state = IntegrationState.AUTHENTICATION_REQUIRED.value
        elif category == ErrorCategory.RATE_LIMIT:
            new_state = IntegrationState.RATE_LIMITED.value
        elif category == ErrorCategory.NETWORK_SERVER:
            new_state = IntegrationState.TEMPORARILY_UNAVAILABLE.value
        elif category == ErrorCategory.PERSISTENT_OUTAGE:
            new_state = IntegrationState.OFFLINE.value
        else:
            new_state = IntegrationState.DEGRADED.value

        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT failure_count FROM seller_health WHERE marketplace = ?", (marketplace,))
            row = cursor.fetchone()
            current_failures = (row[0] if row else 0) + 1

            # Transition to OFFLINE if persistent failure count > 3 for temporary errors
            if current_failures >= 4 and new_state in (
                IntegrationState.TEMPORARILY_UNAVAILABLE.value, IntegrationState.DEGRADED.value
            ):
                new_state = IntegrationState.OFFLINE.value

            conn.execute("""
                INSERT INTO seller_health (
                    marketplace, state, last_failed_request, last_health_check,
                    failure_count, last_error_category, last_error_message, retry_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'RETRIEVING')
                ON CONFLICT(marketplace) DO UPDATE SET
                    state = ?,
                    last_failed_request = ?,
                    last_health_check = ?,
                    failure_count = failure_count + 1,
                    last_error_category = ?,
                    last_error_message = ?
            """, (
                marketplace, new_state, now, now, current_failures, category.value, safe_msg,
                new_state, now, now, category.value, safe_msg
            ))
            conn.commit()

    def update_token_status(self, marketplace: str, token_status: str):
        """Update token authentication status e.g. 'VALID', 'REFRESHING', 'EXPIRED', 'INVALID'."""
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO seller_health (marketplace, state, token_status)
                VALUES (?, ?, ?)
                ON CONFLICT(marketplace) DO UPDATE SET token_status = ?
            """, (marketplace, IntegrationState.UNKNOWN.value, token_status, token_status))
            conn.commit()

    def update_heartbeat(self, marketplace: str, success: bool, message: Optional[str] = None):
        """Record execution timestamp of 25-minute idle heartbeat."""
        now = _now_iso()
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO seller_health (marketplace, state, last_heartbeat)
                VALUES (?, ?, ?)
                ON CONFLICT(marketplace) DO UPDATE SET last_heartbeat = ?
            """, (marketplace, IntegrationState.UNKNOWN.value, now, now))
            conn.commit()

        if success:
            self.update_success(marketplace)
        else:
            self.update_failure(
                marketplace,
                ErrorCategory.NETWORK_SERVER,
                message or "Heartbeat connectivity check failed"
            )

    def get_health(self, marketplace: str) -> SellerHealthStatus:
        """Fetch current health status for a specific marketplace."""
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT marketplace, state, last_successful_request, last_failed_request,
                       last_health_check, failure_count, last_error_category,
                       last_error_message, token_status, retry_state, last_heartbeat
                FROM seller_health WHERE marketplace = ?
            """, (marketplace,))
            row = cursor.fetchone()

        if not row:
            return SellerHealthStatus(
                marketplace=marketplace,
                state=IntegrationState.UNKNOWN,
                token_status="UNKNOWN"
            )

        return SellerHealthStatus(
            marketplace=row[0],
            state=IntegrationState(row[1]) if row[1] in IntegrationState.__members__ else IntegrationState.UNKNOWN,
            last_successful_request=row[2],
            last_failed_request=row[3],
            last_health_check=row[4],
            failure_count=row[5] or 0,
            last_error_category=ErrorCategory(row[6]) if row[6] in ErrorCategory.__members__ else ErrorCategory.UNKNOWN,
            last_error_message=row[7],
            token_status=row[8] or "UNKNOWN",
            retry_state=row[9] or "IDLE",
            last_heartbeat=row[10]
        )

    def get_all_health(self) -> Dict[str, SellerHealthStatus]:
        """Fetch health status for all registered seller integrations."""
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT marketplace, state, last_successful_request, last_failed_request,
                       last_health_check, failure_count, last_error_category,
                       last_error_message, token_status, retry_state, last_heartbeat
                FROM seller_health
            """)
            rows = cursor.fetchall()

        result = {}
        for row in rows:
            mp = row[0]
            result[mp] = SellerHealthStatus(
                marketplace=mp,
                state=IntegrationState(row[1]) if row[1] in IntegrationState.__members__ else IntegrationState.UNKNOWN,
                last_successful_request=row[2],
                last_failed_request=row[3],
                last_health_check=row[4],
                failure_count=row[5] or 0,
                last_error_category=ErrorCategory(row[6]) if row[6] in ErrorCategory.__members__ else ErrorCategory.UNKNOWN,
                last_error_message=row[7],
                token_status=row[8] or "UNKNOWN",
                retry_state=row[9] or "IDLE",
                last_heartbeat=row[10]
            )
        return result
