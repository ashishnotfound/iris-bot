"""Proactive user & Telegram notifications for IRIS seller integration alerts."""

import json
import logging
import time
import urllib.request
import urllib.error
from typing import Dict, Optional, Tuple

from agent.redact import redact_sensitive_text
from agent.secret_scope import get_secret
from seller.base import IntegrationState, SellerHealthStatus
from seller.health import SellerHealthMonitor

logger = logging.getLogger(__name__)

# Default alert deduplication cooldown window (in seconds)
_ALERT_COOLDOWN_SECONDS = 1800.0  # 30 minutes


class SellerNotificationManager:
    """Manages integration health alerts and Telegram proactive notifications.

    Deduplicates alerts within a cooldown period so users are not spammed,
    and sends structured messages to Telegram when state transitions occur.
    """

    def __init__(
        self,
        health_monitor: Optional[SellerHealthMonitor] = None,
        cooldown_seconds: float = _ALERT_COOLDOWN_SECONDS,
    ):
        self.health_monitor = health_monitor or SellerHealthMonitor()
        self.cooldown_seconds = cooldown_seconds
        # Map of (marketplace, state_value) -> last_notified_timestamp
        self._last_notified: Dict[Tuple[str, str], float] = {}

    def _get_telegram_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        token = get_secret("TELEGRAM_BOT_TOKEN", "") or ""
        # Accept chat ID from TELEGRAM_CHAT_ID, TELEGRAM_HOME_CHANNEL, or first user in TELEGRAM_ALLOWED_USERS
        chat_id = (
            get_secret("TELEGRAM_CHAT_ID", "")
            or get_secret("TELEGRAM_HOME_CHANNEL", "")
            or ""
        )
        if not chat_id:
            allowed = get_secret("TELEGRAM_ALLOWED_USERS", "") or ""
            if allowed:
                chat_id = allowed.split(",")[0].strip()
        return (token if token else None, chat_id if chat_id else None)

    def send_telegram_alert(self, message: str) -> bool:
        """Send proactive Telegram notification using Bot API."""
        token, chat_id = self._get_telegram_credentials()
        if not token or not chat_id:
            logger.debug("Telegram credentials not configured; skipping proactive alert.")
            return False

        safe_msg = redact_sensitive_text(message)
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": safe_msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status == 200:
                    logger.info("Successfully dispatched Telegram alert notification.")
                    return True
        except Exception as e:
            logger.warning("Failed to send Telegram notification: %s", redact_sensitive_text(str(e)))
            return False
        return False

    def handle_state_transition(
        self,
        marketplace: str,
        old_state: IntegrationState,
        new_state: IntegrationState,
        error_message: Optional[str] = None,
    ) -> Optional[str]:
        """Check if alert should be emitted, update cooldown tracker, and send Telegram alert.

        Returns notification string if sent/emitted, or None if suppressed by cooldown/no transition.
        """
        if old_state == new_state:
            return None

        now = time.monotonic()
        key = (marketplace, new_state.value)

        # Check deduplication cooldown
        last_sent = self._last_notified.get(key, 0.0)
        if now - last_sent < self.cooldown_seconds:
            logger.debug(
                "Alert for %s (%s) suppressed due to active cooldown window.",
                marketplace, new_state.value
            )
            return None

        # Build user-facing notification text
        mp_name = marketplace.capitalize()
        alert_text: Optional[str] = None

        if new_state == IntegrationState.AUTHENTICATION_REQUIRED:
            alert_text = f"⚠️ <b>{mp_name} API Authentication Alert</b>\nAuthentication is invalid or expired. Re-authentication is required."
        elif new_state == IntegrationState.RATE_LIMITED:
            alert_text = f"⏳ <b>{mp_name} API Rate Limit Alert</b>\nRequests are currently throttled by {mp_name}. IRIS will back off automatically."
        elif new_state in (IntegrationState.TEMPORARILY_UNAVAILABLE, IntegrationState.DEGRADED):
            msg_snippet = f": {error_message}" if error_message else ""
            alert_text = f"⚠️ <b>{mp_name} Integration Degraded</b>\nTemporary API/network failure detected{msg_snippet}."
        elif new_state == IntegrationState.OFFLINE:
            msg_snippet = f": {error_message}" if error_message else ""
            alert_text = f"🚨 <b>{mp_name} API Connection Failed</b>\nIntegration is offline. Unable to retrieve seller data{msg_snippet}."
        elif new_state == IntegrationState.CONNECTED and old_state != IntegrationState.CONNECTED:
            alert_text = f"✅ <b>{mp_name} API Connection Restored</b>\n{mp_name} Seller API is responding normally again."

        if alert_text:
            self._last_notified[key] = now
            self.send_telegram_alert(alert_text)
            return alert_text

        return None
