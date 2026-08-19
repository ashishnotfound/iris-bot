"""Amazon Selling Partner API (SP-API) integration layer for IRIS."""

from datetime import datetime, timezone, timedelta
import json
import logging
import time
import threading
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

# Default LWA Token Endpoint & SP-API Base Endpoints
_LWA_TOKEN_URL = "https://api.amazon.com/auth/o2/token"
_SPAPI_ENDPOINTS = {
    "na": "https://sellingpartnerapi-na.amazon.com",
    "eu": "https://sellingpartnerapi-eu.amazon.com",
    "fe": "https://sellingpartnerapi-fe.amazon.com",
}


class AmazonTokenManager:
    """Manages Login With Amazon (LWA) OAuth access tokens for SP-API.

    Stores credentials securely via environment variables / secret store,
    obtains and caches access tokens, tracks expiration, automatically refreshes
    prior to expiry, and prevents duplicate refresh calls.
    """

    def __init__(self):
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0  # epoch timestamp
        self._lock = threading.Lock()

    def _get_credentials(self) -> Tuple[str, str, str]:
        client_id = get_secret("AMAZON_CLIENT_ID", "") or ""
        client_secret = get_secret("AMAZON_CLIENT_SECRET", "") or ""
        refresh_token = get_secret("AMAZON_REFRESH_TOKEN", "") or ""

        if not (client_id and client_secret and refresh_token):
            raise ValueError("Amazon SP-API credentials missing (AMAZON_CLIENT_ID, AMAZON_CLIENT_SECRET, AMAZON_REFRESH_TOKEN).")
        return client_id, client_secret, refresh_token

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Retrieve valid access token, refreshing if necessary."""
        with self._lock:
            now = time.time()
            # Refresh if forced or token expires within 300 seconds (5 minutes)
            if not force_refresh and self._access_token and (self._expires_at - now > 300):
                return self._access_token

            client_id, client_secret, refresh_token = self._get_credentials()

            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            }

            data = urllib.parse.urlencode(payload).encode("utf-8")
            req = urllib.request.Request(
                _LWA_TOKEN_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    status = getattr(resp, "status", 200)
                    body_bytes = resp.read()
                    if isinstance(status, int) and status not in (200, 201):
                        body = body_bytes.decode("utf-8", errors="replace")
                        raise ValueError(f"LWA token request failed ({status}): {redact_sensitive_text(body)}")

                    result = json.loads(body_bytes.decode("utf-8"))
                    token = result.get("access_token")
                    expires_in = result.get("expires_in", 3600)

                    if not token:
                        raise ValueError("LWA token response missing access_token field.")

                    self._access_token = token
                    self._expires_at = time.time() + float(expires_in)
                    logger.info("Successfully refreshed Amazon LWA access token (expires in %ds).", expires_in)
                    return token
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                safe_err = redact_sensitive_text(f"HTTP {e.code}: {err_body}")
                logger.error("Amazon LWA token refresh failed: %s", safe_err)
                self._access_token = None
                self._expires_at = 0.0
                raise PermissionError(f"Amazon token refresh failed: {safe_err}") from e
            except Exception as e:
                safe_err = redact_sensitive_text(str(e))
                logger.error("Amazon LWA token request exception: %s", safe_err)
                self._access_token = None
                self._expires_at = 0.0
                raise ConnectionError(f"Amazon LWA token error: {safe_err}") from e


class AmazonSellerService(BaseSellerService):
    """Amazon Selling Partner API (SP-API) Service implementation."""

    def __init__(
        self,
        token_manager: Optional[AmazonTokenManager] = None,
        health_monitor: Optional[SellerHealthMonitor] = None,
        notification_manager: Optional[SellerNotificationManager] = None,
    ):
        self.token_manager = token_manager or AmazonTokenManager()
        self.health_monitor = health_monitor or SellerHealthMonitor()
        self.notification_manager = notification_manager or SellerNotificationManager(self.health_monitor)

    @property
    def marketplace(self) -> str:
        return "amazon"

    def _get_base_url(self) -> str:
        region = (get_secret("AMAZON_REGION", "na") or "na").lower()
        return _SPAPI_ENDPOINTS.get(region, _SPAPI_ENDPOINTS["na"])

    def _get_marketplace_id(self) -> str:
        return get_secret("AMAZON_MARKETPLACE_ID", "A21TJRUUN4KGV") or "A21TJRUUN4KGV"  # Default India marketplace

    def _make_api_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Execute an SP-API HTTP request with token refresh & exponential backoff."""
        base_url = self._get_base_url()
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"

        attempt = 0
        backoff = 1.0

        while attempt <= max_retries:
            attempt += 1
            old_health = self.health_monitor.get_health("amazon")

            try:
                access_token = self.token_manager.get_access_token()
                headers = {
                    "x-amz-access-token": access_token,
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
                    self.health_monitor.update_success("amazon")
                    new_health = self.health_monitor.get_health("amazon")
                    self.notification_manager.handle_state_transition(
                        "amazon", old_health.state, new_health.state
                    )
                    return result

            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                safe_err = redact_sensitive_text(f"HTTP {e.code}: {err_body[:300]}")

                # 401 Unauthorized / Token Expired -> Refresh & Retry once
                if e.code in (401, 403) and attempt == 1:
                    logger.warning("Amazon SP-API returned HTTP %d. Forcing access token refresh.", e.code)
                    try:
                        self.token_manager.get_access_token(force_refresh=True)
                        continue
                    except Exception as refresh_err:
                        self.health_monitor.update_failure(
                            "amazon", ErrorCategory.AUTHENTICATION, str(refresh_err)
                        )
                        new_health = self.health_monitor.get_health("amazon")
                        self.notification_manager.handle_state_transition(
                            "amazon", old_health.state, new_health.state, str(refresh_err)
                        )
                        raise

                # 429 Rate Limit -> Backoff
                if e.code == 429:
                    self.health_monitor.update_failure("amazon", ErrorCategory.RATE_LIMIT, safe_err)
                    new_health = self.health_monitor.get_health("amazon")
                    self.notification_manager.handle_state_transition(
                        "amazon", old_health.state, new_health.state, safe_err
                    )
                    if attempt <= max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise RuntimeError(f"Amazon SP-API rate limit exceeded: {safe_err}") from e

                # 5xx Server Errors
                if e.code >= 500:
                    self.health_monitor.update_failure("amazon", ErrorCategory.NETWORK_SERVER, safe_err)
                    new_health = self.health_monitor.get_health("amazon")
                    self.notification_manager.handle_state_transition(
                        "amazon", old_health.state, new_health.state, safe_err
                    )
                    if attempt <= max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise ConnectionError(f"Amazon SP-API server error ({e.code}): {safe_err}") from e

                # Other HTTP errors (400, 404, etc.)
                self.health_monitor.update_failure("amazon", ErrorCategory.MALFORMED_RESPONSE, safe_err)
                new_health = self.health_monitor.get_health("amazon")
                self.notification_manager.handle_state_transition(
                    "amazon", old_health.state, new_health.state, safe_err
                )
                raise ValueError(f"Amazon SP-API error ({e.code}): {safe_err}") from e

            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                safe_err = redact_sensitive_text(str(e))
                self.health_monitor.update_failure("amazon", ErrorCategory.NETWORK_SERVER, safe_err)
                new_health = self.health_monitor.get_health("amazon")
                self.notification_manager.handle_state_transition(
                    "amazon", old_health.state, new_health.state, safe_err
                )
                if attempt <= max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise ConnectionError(f"Amazon SP-API connection failed: {safe_err}") from e

        raise ConnectionError("Amazon SP-API request retries exhausted.")

    def check_health(self) -> SellerHealthStatus:
        """Lightweight API check via getMarketplaceParticipations endpoint."""
        try:
            res = self._make_api_request("sellers/v1/marketplaceParticipations")
            if "payload" in res or isinstance(res, list) or "errors" not in res:
                self.health_monitor.update_success("amazon")
            return self.health_monitor.get_health("amazon")
        except Exception as e:
            logger.warning("Amazon SP-API health check ping failed: %s", redact_sensitive_text(str(e)))
            return self.health_monitor.get_health("amazon")

    def get_orders(
        self,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        limit: int = 50,
    ) -> List[NormalizedOrder]:
        """Fetch normalized orders from Amazon SP-API."""
        params: Dict[str, Any] = {
            "MarketplaceIds": self._get_marketplace_id(),
            "MaxResultsPerPage": min(limit, 100),
        }

        if start_date:
            params["CreatedAfter"] = start_date
        else:
            # Default to last 30 days
            created_after = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            params["CreatedAfter"] = created_after

        if status:
            params["OrderStatuses"] = status

        res = self._make_api_request("orders/v0/orders", params=params)
        raw_orders = res.get("payload", {}).get("Orders", [])

        normalized: List[NormalizedOrder] = []
        for o in raw_orders:
            order_id = o.get("AmazonOrderId", "")
            raw_status = o.get("OrderStatus", "Pending")

            # Map fulfillment channel: AFN = FBA, MFN = MFN / Easy Ship
            raw_fulfillment = o.get("FulfillmentChannel", "MFN")
            fulfillment_channel = "FBA" if raw_fulfillment == "AFN" else "MFN"

            amount_dict = o.get("OrderTotal", {})
            total_amount = float(amount_dict.get("Amount", 0.0))
            currency = amount_dict.get("CurrencyCode", "INR")

            buyer_info = o.get("BuyerInfo", {})
            buyer_name = buyer_info.get("BuyerName")

            shipping_address = o.get("ShippingAddress", {})
            shipping_city = shipping_address.get("City")

            needs_attention = raw_status in ("Unshipped", "PendingAvailability")
            attention_reason = "Order pending dispatch" if needs_attention else None

            order_obj = NormalizedOrder(
                order_id=order_id,
                marketplace="amazon",
                order_date=o.get("PurchaseDate", ""),
                status=raw_status,
                total_amount=total_amount,
                currency=currency,
                fulfillment_channel=fulfillment_channel,
                buyer_name=buyer_name,
                shipping_city=shipping_city,
                needs_attention=needs_attention,
                attention_reason=attention_reason,
            )
            normalized.append(order_obj)

        return normalized

    def get_inventory(
        self,
        sku: Optional[str] = None,
        limit: int = 50,
    ) -> List[NormalizedInventoryItem]:
        """Fetch inventory levels from Amazon SP-API."""
        params: Dict[str, Any] = {
            "granularityType": "Marketplace",
            "granularityId": self._get_marketplace_id(),
            "marketplaceIds": self._get_marketplace_id(),
            "details": "true",
        }
        if sku:
            params["sellerSkus"] = sku

        res = self._make_api_request("fba/inventory/v1/summaries", params=params)
        raw_summaries = res.get("payload", {}).get("inventorySummaries", [])

        items: List[NormalizedInventoryItem] = []
        for inv in raw_summaries:
            item_sku = inv.get("sellerSku", "UNKNOWN")
            title = inv.get("productName", item_sku)
            qty = int(inv.get("totalQuantity", 0))

            status = "IN_STOCK"
            if qty == 0:
                status = "OUT_OF_STOCK"
            elif qty < 5:
                status = "LOW_STOCK"

            items.append(
                NormalizedInventoryItem(
                    sku=item_sku,
                    title=title,
                    marketplace="amazon",
                    quantity_available=qty,
                    quantity_reserved=0,
                    status=status,
                    fulfillment_channel="FBA" if inv.get("fulfillmentChannel") == "AFN" else "MFN",
                )
            )
        return items

    def get_sales_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> NormalizedSalesSummary:
        """Fetch sales metrics or aggregate from orders."""
        orders = self.get_orders(start_date=start_date, limit=100)
        total_orders = len(orders)
        total_revenue = sum(o.total_amount for o in orders)
        currency = orders[0].currency if orders else "INR"

        return NormalizedSalesSummary(
            marketplace="amazon",
            period="Custom" if start_date else "Last 30 Days",
            total_orders=total_orders,
            total_revenue=round(total_revenue, 2),
            currency=currency,
            units_sold=total_orders,
            start_date=start_date,
            end_date=end_date,
        )

    def get_attention_needed_orders(self) -> List[NormalizedOrder]:
        """Fetch Amazon orders needing attention (pending dispatch)."""
        orders = self.get_orders(status="Unshipped")
        return [o for o in orders if o.needs_attention]
