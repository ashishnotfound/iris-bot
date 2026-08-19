"""Unified seller service abstractions combining Amazon and Flipkart integrations."""

import logging
from typing import Any, Dict, List, Optional

from seller.amazon import AmazonSellerService
from seller.base import (
    BaseSellerService,
    IntegrationState,
    NormalizedInventoryItem,
    NormalizedOrder,
    NormalizedSalesSummary,
    SellerHealthStatus,
)
from seller.flipkart import FlipkartSellerService
from seller.health import SellerHealthMonitor
from seller.notifications import SellerNotificationManager

logger = logging.getLogger(__name__)


class UnifiedSellerService:
    """Unified multi-marketplace service layer for IRIS.

    Aggregates data from Amazon and Flipkart, normalizes results into standard schemas,
    preserves marketplace identity ('amazon' vs 'flipkart'), and handles partial outages gracefully.
    """

    def __init__(
        self,
        amazon_service: Optional[AmazonSellerService] = None,
        flipkart_service: Optional[FlipkartSellerService] = None,
        health_monitor: Optional[SellerHealthMonitor] = None,
        notification_manager: Optional[SellerNotificationManager] = None,
    ):
        self.health_monitor = health_monitor or SellerHealthMonitor()
        self.notification_manager = notification_manager or SellerNotificationManager(self.health_monitor)
        self.amazon_service = amazon_service or AmazonSellerService(
            health_monitor=self.health_monitor,
            notification_manager=self.notification_manager,
        )
        self.flipkart_service = flipkart_service or FlipkartSellerService(
            health_monitor=self.health_monitor,
            notification_manager=self.notification_manager,
        )

    def get_orders(
        self,
        marketplace: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch orders from Amazon, Flipkart, or both.

        Returns dict containing:
          - "orders": list of NormalizedOrder dicts
          - "summary": order count breakdown per marketplace
          - "errors": map of marketplace -> error message (if any integration failed)
        """
        target_mp = (marketplace or "combined").lower()
        results: List[NormalizedOrder] = []
        errors: Dict[str, str] = {}
        summary: Dict[str, int] = {"amazon": 0, "flipkart": 0, "total": 0}

        # Query Amazon if target is 'amazon' or 'combined'
        if target_mp in ("amazon", "combined"):
            try:
                amz_orders = self.amazon_service.get_orders(status=status, start_date=start_date, limit=limit)
                results.extend(amz_orders)
                summary["amazon"] = len(amz_orders)
            except Exception as e:
                err_msg = f"Amazon order query failed: {str(e)}"
                logger.warning(err_msg)
                errors["amazon"] = err_msg

        # Query Flipkart if target is 'flipkart' or 'combined'
        if target_mp in ("flipkart", "combined"):
            try:
                fk_orders = self.flipkart_service.get_orders(status=status, start_date=start_date, limit=limit)
                results.extend(fk_orders)
                summary["flipkart"] = len(fk_orders)
            except Exception as e:
                err_msg = f"Flipkart order query failed: {str(e)}"
                logger.warning(err_msg)
                errors["flipkart"] = err_msg

        summary["total"] = len(results)

        return {
            "orders": [o.to_dict() for o in results],
            "summary": summary,
            "errors": errors if errors else None,
        }

    def get_inventory(
        self,
        marketplace: Optional[str] = None,
        sku: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch inventory levels across marketplaces."""
        target_mp = (marketplace or "combined").lower()
        results: List[NormalizedInventoryItem] = []
        errors: Dict[str, str] = {}

        if target_mp in ("amazon", "combined"):
            try:
                amz_inv = self.amazon_service.get_inventory(sku=sku, limit=limit)
                results.extend(amz_inv)
            except Exception as e:
                errors["amazon"] = f"Amazon inventory query failed: {str(e)}"

        if target_mp in ("flipkart", "combined"):
            try:
                fk_inv = self.flipkart_service.get_inventory(sku=sku, limit=limit)
                results.extend(fk_inv)
            except Exception as e:
                errors["flipkart"] = f"Flipkart inventory query failed: {str(e)}"

        return {
            "inventory": [i.to_dict() for i in results],
            "total_items": len(results),
            "errors": errors if errors else None,
        }

    def get_sales_metrics(
        self,
        marketplace: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch sales summary metrics and performance breakdown."""
        target_mp = (marketplace or "combined").lower()
        breakdown: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        total_orders = 0
        total_revenue = 0.0
        currency = "INR"

        if target_mp in ("amazon", "combined"):
            try:
                amz_sales = self.amazon_service.get_sales_metrics(start_date=start_date, end_date=end_date)
                breakdown["amazon"] = amz_sales.to_dict()
                total_orders += amz_sales.total_orders
                total_revenue += amz_sales.total_revenue
                currency = amz_sales.currency
            except Exception as e:
                errors["amazon"] = f"Amazon sales metrics query failed: {str(e)}"

        if target_mp in ("flipkart", "combined"):
            try:
                fk_sales = self.flipkart_service.get_sales_metrics(start_date=start_date, end_date=end_date)
                breakdown["flipkart"] = fk_sales.to_dict()
                total_orders += fk_sales.total_orders
                total_revenue += fk_sales.total_revenue
            except Exception as e:
                errors["flipkart"] = f"Flipkart sales metrics query failed: {str(e)}"

        combined_summary = NormalizedSalesSummary(
            marketplace="combined",
            period="Custom" if start_date else "Last 30 Days",
            total_orders=total_orders,
            total_revenue=round(total_revenue, 2),
            currency=currency,
            units_sold=total_orders,
            start_date=start_date,
            end_date=end_date,
        )

        return {
            "combined": combined_summary.to_dict(),
            "breakdown": breakdown,
            "errors": errors if errors else None,
        }

    def check_all_health(self) -> Dict[str, Any]:
        """Check API connectivity and health status across all seller integrations."""
        amz_health = self.amazon_service.check_health()
        fk_health = self.flipkart_service.check_health()

        return {
            "amazon": amz_health.to_dict(),
            "flipkart": fk_health.to_dict(),
        }

    def get_attention_needed(self, marketplace: Optional[str] = None) -> Dict[str, Any]:
        """Fetch orders requiring immediate action (dispatches, cancellations)."""
        target_mp = (marketplace or "combined").lower()
        results: List[NormalizedOrder] = []
        errors: Dict[str, str] = {}

        if target_mp in ("amazon", "combined"):
            try:
                amz_attn = self.amazon_service.get_attention_needed_orders()
                results.extend(amz_attn)
            except Exception as e:
                errors["amazon"] = str(e)

        if target_mp in ("flipkart", "combined"):
            try:
                fk_attn = self.flipkart_service.get_attention_needed_orders()
                results.extend(fk_attn)
            except Exception as e:
                errors["flipkart"] = str(e)

        return {
            "orders": [o.to_dict() for o in results],
            "total_attention_needed": len(results),
            "errors": errors if errors else None,
        }
