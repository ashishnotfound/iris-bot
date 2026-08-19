"""
lib/business_snapshot.py — Amazon/Flipkart Business Intelligence Snapshot

Maintains a compact, daily snapshot of key seller metrics per chat.
Data flows: Amazon SP API / Composio → business_snapshots table → LLM context

Design:
  - Snapshot is updated by scheduled sync (cron job) or explicit "check now" command
  - Questions like "How's business today?" use the latest snapshot (fast, no API call)
  - "Check Amazon right now" triggers a live sync and updates the snapshot
  - Incremental sync uses `sync_cursor` to avoid re-downloading everything
  - Supports Amazon Selling Partner API directly (SP-API) and Composio as fallback

Environment variables:
  AMAZON_SELLER_ID        — Amazon Seller ID (Merchant ID)
  AMAZON_MARKETPLACE_ID   — Marketplace ID (e.g. ATVPDKIKX0DER for US, A21TJRUUN4KGV for IN)
  AMAZON_LWA_CLIENT_ID    — Login with Amazon OAuth client ID
  AMAZON_LWA_CLIENT_SECRET— Login with Amazon OAuth client secret
  AMAZON_REFRESH_TOKEN    — LWA refresh token (long-lived)
  FLIPKART_API_KEY        — Flipkart Seller API key (optional)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase helpers (shared pattern)
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


# ---------------------------------------------------------------------------
# Amazon SP-API Client (lightweight, no heavy SDK dependency)
# ---------------------------------------------------------------------------


class AmazonSPClient:
    """Minimal Amazon Selling Partner API client.

    Covers: Orders, Inventory, Sales metrics.
    Uses LWA (Login with Amazon) for OAuth2.
    """

    LWA_TOKEN_URL = "https://api.amazon.com/auth/o2/token"
    SP_API_BASE = "https://sellingpartnerapi-na.amazon.com"  # Default: North America

    # Marketplace → endpoint mapping
    MARKETPLACE_ENDPOINTS = {
        # India
        "A21TJRUUN4KGV": "https://sellingpartnerapi-fe.amazon.com",
        # North America
        "ATVPDKIKX0DER": "https://sellingpartnerapi-na.amazon.com",
        "A2EUQ1WTGCTBG2": "https://sellingpartnerapi-na.amazon.com",  # CA
        # Europe
        "A1F83G8C2ARO7P": "https://sellingpartnerapi-eu.amazon.com",  # UK
        "A1PA6795UKMFR9": "https://sellingpartnerapi-eu.amazon.com",  # DE
    }

    def __init__(self) -> None:
        self.seller_id = os.environ.get("AMAZON_SELLER_ID", "").strip()
        self.marketplace_id = os.environ.get("AMAZON_MARKETPLACE_ID", "").strip()
        self.client_id = os.environ.get("AMAZON_LWA_CLIENT_ID", "").strip()
        self.client_secret = os.environ.get("AMAZON_LWA_CLIENT_SECRET", "").strip()
        self.refresh_token = os.environ.get("AMAZON_REFRESH_TOKEN", "").strip()
        self._access_token: Optional[str] = None
        self._token_expires: float = 0.0
        self.base_url = self.MARKETPLACE_ENDPOINTS.get(
            self.marketplace_id, self.SP_API_BASE
        )

    def is_configured(self) -> bool:
        return bool(
            self.seller_id
            and self.marketplace_id
            and self.client_id
            and self.client_secret
            and self.refresh_token
        )

    def _get_access_token(self) -> Optional[str]:
        """Obtain a short-lived access token via LWA refresh token."""
        import requests, time

        if self._access_token and time.time() < self._token_expires - 30:
            return self._access_token

        try:
            r = requests.post(
                self.LWA_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                self._access_token = data["access_token"]
                self._token_expires = time.time() + data.get("expires_in", 3600)
                return self._access_token
            else:
                logger.error("Amazon LWA token error %d: %s", r.status_code, r.text[:200])
                return None
        except Exception as e:
            logger.error("Amazon LWA token request failed: %s", e)
            return None

    def _headers(self) -> Dict[str, str]:
        token = self._get_access_token()
        return {
            "x-amz-access-token": token or "",
            "Content-Type": "application/json",
        }

    def get_orders(
        self,
        *,
        created_after: Optional[str] = None,
        order_statuses: Optional[List[str]] = None,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """Fetch recent orders from Amazon SP-API Orders endpoint."""
        import requests

        if not created_after:
            # Default: last 24 hours
            created_after = (
                datetime.now(timezone.utc) - timedelta(days=1)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

        params: Dict[str, Any] = {
            "MarketplaceIds": self.marketplace_id,
            "CreatedAfter": created_after,
            "MaxResultsPerPage": min(max_results, 100),
        }
        if order_statuses:
            params["OrderStatuses"] = ",".join(order_statuses)

        try:
            r = requests.get(
                f"{self.base_url}/orders/v0/orders",
                headers=self._headers(),
                params=params,
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("payload", {})
            else:
                logger.warning("Amazon get_orders HTTP %d: %s", r.status_code, r.text[:200])
                return {"error": f"HTTP {r.status_code}", "raw": r.text[:200]}
        except Exception as e:
            logger.error("Amazon get_orders failed: %s", e)
            return {"error": str(e)}

    def get_sales_metrics(
        self,
        *,
        interval: str = "DAY",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch sales metrics from Sales Analytics API."""
        import requests

        today = date.today()
        if not end_date:
            end_date = today.isoformat()
        if not start_date:
            start_date = (today - timedelta(days=6)).isoformat()

        params = {
            "marketplaceIds": self.marketplace_id,
            "interval": interval,
            "granularity": interval,
            "granularityTimeZone": "UTC",
            "dataStartTime": f"{start_date}T00:00:00Z",
            "dataEndTime": f"{end_date}T23:59:59Z",
        }
        try:
            r = requests.get(
                f"{self.base_url}/sales/v1/orderMetrics",
                headers=self._headers(),
                params=params,
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("payload", [])
            return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            logger.error("Amazon get_sales_metrics failed: %s", e)
            return {"error": str(e)}

    def get_inventory(self) -> Dict[str, Any]:
        """Fetch FBA inventory levels."""
        import requests

        try:
            r = requests.get(
                f"{self.base_url}/fba/inventory/v1/summaries",
                headers=self._headers(),
                params={
                    "details": "true",
                    "granularityType": "Marketplace",
                    "granularityId": self.marketplace_id,
                    "marketplaceIds": self.marketplace_id,
                },
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("payload", {})
            return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            logger.error("Amazon get_inventory failed: %s", e)
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Flipkart Seller API Client (basic)
# ---------------------------------------------------------------------------


class FlipkartSPClient:
    """Minimal Flipkart Seller API client."""

    BASE_URL = "https://api.flipkart.net/sellers"

    def __init__(self) -> None:
        self.api_key = os.environ.get("FLIPKART_API_KEY", "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_orders(self, *, state: str = "APPROVED", max_results: int = 50) -> Dict[str, Any]:
        """Fetch orders from Flipkart Seller API."""
        import requests

        try:
            r = requests.get(
                f"{self.BASE_URL}/orders/v2/search",
                headers=self._headers(),
                params={"orderState": state, "pageSize": min(max_results, 100)},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            logger.error("Flipkart get_orders failed: %s", e)
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Business Snapshot Manager
# ---------------------------------------------------------------------------


class BusinessSnapshotManager:
    """Reads/writes business snapshots from Supabase.

    A snapshot is a compact daily summary of seller metrics, stored as JSONB.
    """

    def __init__(self) -> None:
        self._amazon = AmazonSPClient()
        self._flipkart = FlipkartSPClient()

    def get_latest(
        self, chat_id: int, platform: str = "amazon"
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot for a chat+platform."""
        import requests

        base = _sb_url()
        if not base:
            return None
        try:
            r = requests.get(
                f"{base}/rest/v1/business_snapshots",
                headers=_sb_headers(),
                params={
                    "chat_id": f"eq.{chat_id}",
                    "platform": f"eq.{platform}",
                    "select": "data,snapshot_date,synced_at,sync_error",
                    "order": "snapshot_date.desc",
                    "limit": "1",
                },
                timeout=6,
            )
            if r.status_code == 200 and r.json():
                return r.json()[0]
        except Exception as e:
            logger.error("get_latest snapshot failed: %s", e)
        return None

    def save(
        self,
        chat_id: int,
        platform: str,
        data: Dict[str, Any],
        *,
        sync_cursor: Optional[str] = None,
        sync_error: Optional[str] = None,
    ) -> bool:
        """Upsert today's snapshot for this chat+platform."""
        import requests

        base = _sb_url()
        if not base:
            return False
        today = date.today().isoformat()
        payload = {
            "chat_id": chat_id,
            "platform": platform,
            "snapshot_date": today,
            "data": data,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }
        if sync_cursor is not None:
            payload["sync_cursor"] = sync_cursor
        if sync_error is not None:
            payload["sync_error"] = sync_error

        try:
            r = requests.post(
                f"{base}/rest/v1/business_snapshots",
                headers={**_sb_headers(), "Prefer": "resolution=merge-duplicates"},
                json=payload,
                timeout=8,
            )
            return r.status_code in (200, 201, 204)
        except Exception as e:
            logger.error("save snapshot failed: %s", e)
            return False

    def sync_amazon(self, chat_id: int) -> Dict[str, Any]:
        """Perform a live Amazon sync and update the snapshot.

        Returns a summary dict suitable for display.
        """
        if not self._amazon.is_configured():
            return {
                "success": False,
                "error": (
                    "Amazon SP-API not configured. "
                    "Please set AMAZON_SELLER_ID, AMAZON_MARKETPLACE_ID, "
                    "AMAZON_LWA_CLIENT_ID, AMAZON_LWA_CLIENT_SECRET, "
                    "and AMAZON_REFRESH_TOKEN in Vercel environment variables."
                ),
            }

        errors: List[str] = []

        # Load existing snapshot to get cursor
        existing = self.get_latest(chat_id, "amazon") or {}
        last_cursor = (existing.get("data") or {}).get("sync_cursor")

        # Fetch orders (incremental if cursor available)
        created_after = last_cursor or (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        orders_data = self._amazon.get_orders(
            created_after=created_after,
            order_statuses=["Pending", "Unshipped", "Shipped", "Canceled"],
        )

        if "error" in orders_data:
            errors.append(f"Orders: {orders_data['error']}")
            orders_data = {}

        raw_orders = orders_data.get("Orders", [])

        # Aggregate metrics
        total_orders = len(raw_orders)
        pending = sum(1 for o in raw_orders if o.get("OrderStatus") == "Pending")
        unshipped = sum(1 for o in raw_orders if o.get("OrderStatus") == "Unshipped")
        shipped = sum(1 for o in raw_orders if o.get("OrderStatus") == "Shipped")
        canceled = sum(1 for o in raw_orders if o.get("OrderStatus") == "Canceled")

        # Total sales value
        total_sales = 0.0
        currency = "INR"
        for o in raw_orders:
            amt = o.get("OrderTotal", {})
            try:
                total_sales += float(amt.get("Amount", 0))
                currency = amt.get("CurrencyCode", currency)
            except (TypeError, ValueError):
                pass

        # New cursor = latest order date
        new_cursor = None
        if raw_orders:
            dates = [o.get("PurchaseDate", "") for o in raw_orders if o.get("PurchaseDate")]
            if dates:
                new_cursor = max(dates)

        snapshot_data = {
            "platform": "amazon",
            "date": date.today().isoformat(),
            "total_orders": total_orders,
            "pending": pending,
            "unshipped": unshipped,
            "shipped": shipped,
            "canceled": canceled,
            "total_sales": round(total_sales, 2),
            "currency": currency,
            "sync_cursor": new_cursor or created_after,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

        ok = self.save(
            chat_id,
            "amazon",
            snapshot_data,
            sync_cursor=new_cursor,
            sync_error="; ".join(errors) if errors else None,
        )

        return {
            "success": ok and not errors,
            "data": snapshot_data,
            "errors": errors,
        }

    def sync_flipkart(self, chat_id: int) -> Dict[str, Any]:
        """Perform a live Flipkart sync and update the snapshot."""
        if not self._flipkart.is_configured():
            return {
                "success": False,
                "error": (
                    "Flipkart API not configured. "
                    "Please set FLIPKART_API_KEY in Vercel environment variables."
                ),
            }

        errors: List[str] = []
        orders_data = self._flipkart.get_orders()

        if "error" in orders_data:
            errors.append(f"Orders: {orders_data['error']}")
            raw_orders = []
        else:
            raw_orders = orders_data.get("orderItems", [])

        total_orders = len(raw_orders)

        snapshot_data = {
            "platform": "flipkart",
            "date": date.today().isoformat(),
            "total_orders": total_orders,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

        ok = self.save(
            chat_id,
            "flipkart",
            snapshot_data,
            sync_error="; ".join(errors) if errors else None,
        )

        return {
            "success": ok and not errors,
            "data": snapshot_data,
            "errors": errors,
        }

    def format_for_llm(
        self,
        chat_id: int,
        platforms: Optional[List[str]] = None,
    ) -> str:
        """Build a compact business context block for injection into the LLM.

        Returns empty string if no snapshots exist (LLM falls back to model knowledge).
        """
        if platforms is None:
            platforms = ["amazon", "flipkart"]

        blocks: List[str] = []
        for platform in platforms:
            snap = self.get_latest(chat_id, platform)
            if not snap:
                continue
            data = snap.get("data", {})
            synced = snap.get("synced_at", "unknown")
            snap_date = snap.get("snapshot_date", "today")

            if platform == "amazon":
                blocks.append(
                    f"[Amazon Snapshot — {snap_date} — synced {synced[:16]} UTC]\n"
                    f"  Orders: {data.get('total_orders', '?')} | "
                    f"Pending: {data.get('pending', '?')} | "
                    f"Unshipped: {data.get('unshipped', '?')} | "
                    f"Shipped: {data.get('shipped', '?')} | "
                    f"Cancelled: {data.get('canceled', '?')}\n"
                    f"  Revenue: {data.get('currency', 'INR')} {data.get('total_sales', '?')}"
                )
            elif platform == "flipkart":
                blocks.append(
                    f"[Flipkart Snapshot — {snap_date} — synced {synced[:16]} UTC]\n"
                    f"  Orders: {data.get('total_orders', '?')}"
                )

            if snap.get("sync_error"):
                blocks[-1] += f"\n  ⚠️ Last sync warning: {snap['sync_error'][:100]}"

        if not blocks:
            return ""

        header = "## Business Intelligence (Latest Snapshot — NOT model knowledge)\n"
        footer = "\nUse this data to answer business questions. Do NOT invent metrics."
        return header + "\n\n".join(blocks) + footer
