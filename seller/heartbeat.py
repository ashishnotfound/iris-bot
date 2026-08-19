"""25-minute Amazon connection idle heartbeat runner with cross-process advisory locking."""

import contextlib
import logging
import os
import sys
import threading
import time
from typing import Optional

from hermes_constants import get_hermes_home
from seller.amazon import AmazonSellerService
from seller.health import SellerHealthMonitor

logger = logging.getLogger(__name__)

# Cross-process advisory locking for single-instance heartbeat execution
try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

_HEARTBEAT_INTERVAL_SECONDS = 1500.0  # 25 minutes


class AmazonHeartbeatRunner:
    """Manages the 25-minute Amazon connection heartbeat.

    Periodically executes a lightweight API health check ping when idle,
    recording results in the SellerHealthMonitor. Prevents duplicate execution
    across multiple process workers using cross-process file locking.
    """

    def __init__(
        self,
        amazon_service: Optional[AmazonSellerService] = None,
        health_monitor: Optional[SellerHealthMonitor] = None,
        interval_seconds: float = _HEARTBEAT_INTERVAL_SECONDS,
    ):
        self.health_monitor = health_monitor or SellerHealthMonitor()
        self.amazon_service = amazon_service or AmazonSellerService(health_monitor=self.health_monitor)
        self.interval_seconds = interval_seconds
        self._lock_file_path = os.path.join(get_hermes_home(), "amazon_heartbeat.lock")
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @contextlib.contextmanager
    def _acquire_process_lock(self):
        """Try to acquire cross-process advisory lock file. Yields True if acquired, False otherwise."""
        os.makedirs(os.path.dirname(self._lock_file_path), exist_ok=True)
        fh = open(self._lock_file_path, "a+")
        locked = False
        try:
            if fcntl:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except (OSError, IOError):
                    locked = False
            elif msvcrt:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except (OSError, IOError):
                    locked = False
            else:
                locked = True  # Fallback to in-process execution

            yield locked

        finally:
            if locked:
                if fcntl:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                elif msvcrt:
                    try:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
            try:
                fh.close()
            except Exception:
                pass

    def run_heartbeat_once(self) -> bool:
        """Execute lightweight Amazon connection heartbeat if process lock is acquired.

        Returns True if heartbeat was executed by this process, False if skipped/locked out.
        """
        with self._acquire_process_lock() as acquired:
            if not acquired:
                logger.debug("Amazon heartbeat skipped: another worker process currently holds the lock.")
                return False

            logger.info("Running 25-minute Amazon connection idle heartbeat...")
            try:
                status = self.amazon_service.check_health()
                is_healthy = status.state.value in ("CONNECTED", "DEGRADED")
                self.health_monitor.update_heartbeat("amazon", success=is_healthy, message=status.last_error_message)
                logger.info("Amazon heartbeat completed. Health state: %s", status.state.value)
                return True
            except Exception as e:
                err_msg = str(e)
                logger.warning("Amazon heartbeat execution failed: %s", err_msg)
                self.health_monitor.update_heartbeat("amazon", success=False, message=err_msg)
                return True

    def start_background_loop(self):
        """Start background daemon thread executing periodic heartbeat checks."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                self.run_heartbeat_once()
                # Sleep in short increments to allow graceful shutdown
                elapsed = 0.0
                while elapsed < self.interval_seconds and not self._stop_event.is_set():
                    time.sleep(1.0)
                    elapsed += 1.0

        self._thread = threading.Thread(target=_loop, name="AmazonHeartbeatThread", daemon=True)
        self._thread.start()
        logger.info("Started Amazon 25-minute idle heartbeat background runner.")

    def stop_background_loop(self):
        """Signal background heartbeat thread to terminate."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
