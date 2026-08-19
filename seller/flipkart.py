"""Flipkart Seller API integration layer for IRIS."""

from datetime import datetime, timezone, timedelta
import json
import logging
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from agent.redact import redact_sensitive_text
from agent.secret_scope import get_secret
from seller.base import (
    BaseSellerService,
    ErrorCategory,
    IntegrationState,
    NormalizedInventoryItem,
    NormalizedOrder,
    NormalizedOrderItem,
    NormalizedSalesSummary,
    SellerHealthStatus,
)
from seller.health import SellerHealthMonitor
from seller.notifications import SellerNotificationManager

logger = logging.getLogger(__name__)

_FLIPKART_OAUTH_URL = "https://api.flipkart.net/oauth-token"
_FLIPKART_BASE_URL = "https://api.flipkart.net/sellers"


class FlipkartSellerService(BaseSellerService):
    """Flipkart Seller API Service implementation."""

    def __init__(
        self,
        health_monitor: Optional[SellerHealthMonitor] = None,
        notification_manager: Optional[SellerNotificationManager] = None,
    ):
        self.health_monitor = health_monitor or SellerHealthMonitor()
        self.notification_manager = notification_manager or SellerNotificationManager(self.health_monitor)
        self._cached_token: Optional[str] = None

    @property
    def marketplace(self) -> str:
        return "flipkart"

    def _get_credentials(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        app_id = get_secret("FLIPKART_APP_ID", "") or None
        app_secret = get_secret("FLIPKART_APP_SECRET", "") or None
        access_token = get_secret("FLIPKART_ACCESS_TOKEN", "") or None
        return app_id, app_secret, access_token

    def _get_access_token(self, force_refresh: bool = False) -> str:
        """Retrieve Flipkart access token.

        Uses long-lived access token if configured directly via FLIPKART_ACCESS_TOKEN.
        Otherwise exchanges app_id and app_secret for a client_credentials token.
        Does NOT force 1-hour expiration logic unless requested due to authentication failure.
        """
        app_id, app_secret, configured_token = self._get_credentials()

        if configured_token and not force_refresh:
            return configured_token

        if self._cached_token and not force_refresh:
            return self._cached_token

        if not (app_id and app_secret):
            if configured_token:
                return configured_token
            raise ValueError("Flipkart credentials missing (FLIPKART_APP_ID & FLIPKART_APP_SECRET or FLIPKART_ACCESS_TOKEN).")

        params = {
            "grant_type": "client_credentials",
            "scope": "Seller_Access",
        }

        # Flipkart OAuth basic auth header with app_id:app_secret
        import base64
        creds = f"{app_id}:{app_secret}".encode("utf-8")
        basic_auth = base64.b64encode(creds).decode("utf-8")

        url = f"{_FLIPKART_OAUTH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Basic {basic_auth}",
                "User-Agent": "IRIS-Seller-Agent/1.0",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                status = getattr(resp, "status", 200)
                body_bytes = resp.read()
                if isinstance(status, int) and status not in (200, 201):
                    body = body_bytes.decode("utf-8", errors="replace")
                    raise ValueError(f"Flipkart OAuth failed ({status}): {redact_sensitive_text(body)}")

                result = json.loads(body_bytes.decode("utf-8"))
                token = result.get("access_token")
                if not token:
                    raise ValueError("Flipkart OAuth response missing access_token field.")

                self._cached_token = token
                logger.info("Successfully acquired Flipkart OAuth access token.")
                return token
        except Exception as e:
            safe_err = redact_sensitive_text(str(e))
            logger.error("Flipkart OAuth exception: %s", safe_err)
            self._cached_token = None
            raise PermissionError(f"Flipkart authentication failed: {safe_err}") from e

    def _make_api_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Execute a Flipkart API request with retry and error classification."""
        url = f"{_FLIPKART_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        attempt = 0
        backoff = 1.0

        while attempt <= max_retries:
            attempt += 1
            old_health = self.health_monitor.get_health("flipkart")

            try:
                token = self._get_access_token()
                headers = {
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "IRIS-Seller-Agent/1.0",
                    "Accept": "application/json",
                }

                req_data = None
                if data is not None:
                    req_data = json.dumps(data).encode("utf-8")
                    headers["Content-Type"] = "application/json"

                req = urllib.request.Request(url, data=req_data, headers=headers, method=method)

                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    resp_body = resp.read().decode("utf-8")
                    result = json.loads(resp_body) if resp_body else {}
                    self.health_monitor.update_success("flipkart")
                    new_health = self.health_monitor.get_health("flipkart")
                    self.notification_manager.handle_state_transition(
                        "flipkart", old_health.state, new_health.state
                    )
                    return result

            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                safe_err = redact_sensitive_text(f"HTTP {e.code}: {err_body[:300]}")

                # 401 / 403 Invalid Credential -> Refresh token once
                if e.code in (401, 403) and attempt == 1:
                    logger.warning("Flipkart API returned HTTP %d. Re-authenticating.", e.code)
                    try:
                        self._get_access_token(force_refresh=True)
                        continue
                    except Exception as reauth_err:
                        self.health_monitor.update_failure(
                            "flipkart", ErrorCategory.INVALID_CREDENTIALS, str(reauth_err)
                        )
                        new_health = self.health_monitor.get_health("flipkart")
                        self.notification_manager.handle_state_transition(
                            "flipkart", old_health.state, new_health.state, str(reauth_err)
                        )
                        raise

                # 429 Rate Limit -> Backoff
                if e.code == 429:
                    self.health_monitor.update_failure("flipkart", ErrorCategory.RATE_LIMIT, safe_err)
                    new_health = self.health_monitor.get_health("flipkart")
                    self.notification_manager.handle_state_transition(
                        "flipkart", old_health.state, new_health.state, safe_err
                    )
                    if attempt <= max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise RuntimeError(f"Flipkart rate limit exceeded: {safe_err}") from e

                # 5xx Server Error
                if e.code >= 500:
                    self.health_monitor.update_failure("flipkart", ErrorCategory.NETWORK_SERVER, safe_err)
                    new_health = self.health_monitor.get_health("flipkart")
                    self.notification_manager.handle_state_transition(
                        "flipkart", old_health.state, new_health.state, safe_err
                    )
                    if attempt <= max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise ConnectionError(f"Flipkart server error ({e.code}): {safe_err}") from e

                self.health_monitor.update_failure("flipkart", ErrorCategory.MALFORMED_RESPONSE, safe_err)
                new_health = self.health_monitor.get_health("flipkart")
                self.notification_manager.handle_state_transition(
                    "flipkart", old_health.state, new_health.state, safe_err
                )
                raise ValueError(f"Flipkart API error ({e.code}): {safe_err}") from e

            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                safe_err = redact_sensitive_text(str(e))
                self.health_monitor.update_failure("flipkart", ErrorCategory.NETWORK_SERVER, safe_err)
                new_health = self.health_monitor.get_health("flipkart")
                self.notification_manager.handle_state_transition(
                    "flipkart", old_health.state, new_health.state, safe_err
                )
                if attempt <= max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise ConnectionError(f"Flipkart connection failed: {safe_err}") from e

        raise ConnectionError("Flipkart API request retries exhausted.")

    def check_health(self) -> SellerHealthStatus:
        """Lightweight API check ping for Flipkart."""
        try:
            res = self._make_api_request("v3/shipments/filter", method="POST", data={"filter": {"states": ["APPROVED"]}})
            if res is not None:
                self.health_monitor.update_success("flipkart")
            return self.health_monitor.get_health("flipkart")
        except Exception as e:
            logger.warning("Flipkart health check ping failed: %s", redact_sensitive_text(str(e)))
            return self.health_monitor.get_health("flipkart")

    def get_orders(
        self,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        limit: int = 50,
    ) -> List[NormalizedOrder]:
        """Fetch normalized orders from Flipkart Seller API."""
        filter_states = [status] if status else ["APPROVED", "PACKED", "READY_TO_DISPATCH"]
        payload = {
            "filter": {
                "states": filter_states
            },
            "pagination": {
                "pageSize": min(limit, 50)
            }
        }

        res = self._make_api_request("v3/shipments/filter", method="POST", data=payload)
        raw_shipments = res.get("shipments", [])

        normalized: List[NormalizedOrder] = []
        for s in raw_shipments:
            shipment_id = s.get("shipmentId", "UNKNOWN")
            raw_status = s.get("status", "APPROVED")

            items: List[NormalizedOrderItem] = []
            total_amount = 0.0

            for item in s.get("orderItems", []):
                item_sku = item.get("sku", "UNKNOWN")
                title = item.get("title", item_sku)
                qty = int(item.get("quantity", 1))
                price = float(item.get("price", 0.0))
                total_amount += price * qty
                items.append(
                    NormalizedOrderItem(
                        sku=item_sku,
                        title=title,
                        quantity=qty,
                        unit_price=price,
                        currency="INR",
                    )
                )

            needs_attention = raw_status in ("APPROVED", "READY_TO_DISPATCH")
            attention_reason = "Pending dispatch/packing" if needs_attention else None

            normalized.append(
                NormalizedOrder(
                    order_id=shipment_id,
                    marketplace="flipkart",
                    order_date=s.get("orderDate", datetime.now(timezone.utc).isoformat()),
                    status=raw_status,
                    total_amount=round(total_amount, 2),
                    currency="INR",
                    items=items,
                    fulfillment_channel="FLIPKART_FULFILLED" if s.get("fulfillmentType") == "FA" else "MFN",
                    needs_attention=needs_attention,
                    attention_reason=attention_reason,
                )
            )
        return normalized

    def get_inventory(
        self,
        sku: Optional[str] = None,
        limit: int = 50,
    ) -> List[NormalizedInventoryItem]:
        """Fetch inventory listings from Flipkart."""
        res = self._make_api_request("v3/listings/search", method="POST", data={"skus": [sku]} if sku else {})
        raw_listings = res.get("listings", [])

        items: List[NormalizedInventoryItem] = []
        for l in raw_listings:
            item_sku = l.get("sku", "UNKNOWN")
            title = l.get("title", item_sku)
            qty = int(l.get("stockCount", 0))

            status = "IN_STOCK"
            if qty == 0:
                status = "OUT_OF_STOCK"
            elif qty < 5:
                status = "LOW_STOCK"

            items.append(
                NormalizedInventoryItem(
                    sku=item_sku,
                    title=title,
                    marketplace="flipkart",
                    quantity_available=qty,
                    status=status,
                    unit_price=float(l.get("price", 0.0)),
                    fulfillment_channel="MFN",
                )
            )
        return items

    def get_sales_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> NormalizedSalesSummary:
        """Fetch sales summary for Flipkart."""
        orders = self.get_orders(start_date=start_date, limit=100)
        total_orders = len(orders)
        total_revenue = sum(o.total_amount for o in orders)

        return NormalizedSalesSummary(
            marketplace="flipkart",
            period="Custom" if start_date else "Last 30 Days",
            total_orders=total_orders,
            total_revenue=round(total_revenue, 2),
            currency="INR",
            units_sold=total_orders,
            start_date=start_date,
            end_date=end_date,
        )

    def get_attention_needed_orders(self) -> List[NormalizedOrder]:
        """Fetch Flipkart orders requiring attention."""
        orders = self.get_orders(status="APPROVED")
        return [o for o in orders if o.needs_attention]
