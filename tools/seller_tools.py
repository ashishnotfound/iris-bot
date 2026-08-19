"""Seller tools for IRIS - Amazon & Flipkart marketplace integration.

Registers LLM-callable tools for retrieving live orders, inventory stock,
sales summaries, diagnostic connection health, and urgent action items.
"""

import json
import logging
from typing import Any, Dict, Optional

from agent.secret_scope import get_secret
from seller.unified import UnifiedSellerService
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Single instance of UnifiedSellerService for tool handlers
_unified_service: Optional[UnifiedSellerService] = None


def _get_seller_service() -> UnifiedSellerService:
    global _unified_service
    if _unified_service is None:
        _unified_service = UnifiedSellerService()
    return _unified_service


def _check_seller_available() -> bool:
    """Check if Amazon or Flipkart seller credentials are configured."""
    has_amz = bool(
        get_secret("AMAZON_CLIENT_ID", "")
        and get_secret("AMAZON_CLIENT_SECRET", "")
        and get_secret("AMAZON_REFRESH_TOKEN", "")
    )
    has_fk = bool(
        (get_secret("FLIPKART_APP_ID", "") and get_secret("FLIPKART_APP_SECRET", ""))
        or get_secret("FLIPKART_ACCESS_TOKEN", "")
    )
    return has_amz or has_fk


# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------

SELLER_GET_ORDERS_SCHEMA = {
    "name": "seller_get_orders",
    "description": (
        "Fetch live orders across Amazon Seller, Flipkart Seller, or combined marketplaces. "
        "Returns normalized order lists with items, total amounts, and marketplace identity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "marketplace": {
                "type": "string",
                "enum": ["amazon", "flipkart", "combined"],
                "description": "Target marketplace ('amazon', 'flipkart', or 'combined'). Default is 'combined'.",
            },
            "status": {
                "type": "string",
                "description": "Optional order status filter (e.g. 'Unshipped', 'Pending', 'APPROVED', 'PACKED').",
            },
            "start_date": {
                "type": "string",
                "description": "Optional start date filter in ISO 8601 format (e.g. '2026-08-01T00:00:00Z').",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of orders to return (default 50).",
            },
        },
    },
}

SELLER_GET_INVENTORY_SCHEMA = {
    "name": "seller_get_inventory",
    "description": (
        "Fetch live inventory levels and stock status across Amazon Seller, Flipkart Seller, or combined."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "marketplace": {
                "type": "string",
                "enum": ["amazon", "flipkart", "combined"],
                "description": "Target marketplace ('amazon', 'flipkart', or 'combined'). Default is 'combined'.",
            },
            "sku": {
                "type": "string",
                "description": "Optional specific seller SKU to inspect.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum items to return (default 50).",
            },
        },
    },
}

SELLER_GET_SALES_SCHEMA = {
    "name": "seller_get_sales",
    "description": (
        "Fetch sales metrics, order counts, and comparative performance breakdown between Amazon and Flipkart."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "marketplace": {
                "type": "string",
                "enum": ["amazon", "flipkart", "combined"],
                "description": "Target marketplace ('amazon', 'flipkart', or 'combined'). Default is 'combined'.",
            },
            "start_date": {
                "type": "string",
                "description": "Optional start date filter in ISO format.",
            },
            "end_date": {
                "type": "string",
                "description": "Optional end date filter in ISO format.",
            },
        },
    },
}

SELLER_CHECK_HEALTH_SCHEMA = {
    "name": "seller_check_health",
    "description": (
        "Diagnostic tool to inspect Amazon and Flipkart API connection status, authentication state, "
        "failure logs, and 25-minute idle heartbeat timestamps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "marketplace": {
                "type": "string",
                "enum": ["amazon", "flipkart", "combined"],
                "description": "Target marketplace to check. Default is 'combined'.",
            },
        },
    },
}

SELLER_GET_ATTENTION_NEEDED_SCHEMA = {
    "name": "seller_get_attention_needed",
    "description": (
        "Fetch orders requiring immediate seller action (pending dispatch, cancellation requests, packaging)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "marketplace": {
                "type": "string",
                "enum": ["amazon", "flipkart", "combined"],
                "description": "Target marketplace ('amazon', 'flipkart', or 'combined'). Default is 'combined'.",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

def _handle_seller_get_orders(args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    try:
        mp = args.get("marketplace")
        status = args.get("status")
        start_date = args.get("start_date")
        limit = args.get("limit", 50)

        svc = _get_seller_service()
        res = svc.get_orders(marketplace=mp, status=status, start_date=start_date, limit=limit)
        return json.dumps(res, indent=2)
    except Exception as e:
        return tool_error(f"Failed to fetch seller orders: {str(e)}")


def _handle_seller_get_inventory(args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    try:
        mp = args.get("marketplace")
        sku = args.get("sku")
        limit = args.get("limit", 50)

        svc = _get_seller_service()
        res = svc.get_inventory(marketplace=mp, sku=sku, limit=limit)
        return json.dumps(res, indent=2)
    except Exception as e:
        return tool_error(f"Failed to fetch seller inventory: {str(e)}")


def _handle_seller_get_sales(args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    try:
        mp = args.get("marketplace")
        start_date = args.get("start_date")
        end_date = args.get("end_date")

        svc = _get_seller_service()
        res = svc.get_sales_metrics(marketplace=mp, start_date=start_date, end_date=end_date)
        return json.dumps(res, indent=2)
    except Exception as e:
        return tool_error(f"Failed to fetch seller sales metrics: {str(e)}")


def _handle_seller_check_health(args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    try:
        mp = args.get("marketplace", "combined")
        svc = _get_seller_service()
        health_data = svc.check_all_health()

        if mp and mp in ("amazon", "flipkart"):
            return json.dumps({mp: health_data.get(mp)}, indent=2)
        return json.dumps(health_data, indent=2)
    except Exception as e:
        return tool_error(f"Failed to check seller API health: {str(e)}")


def _handle_seller_get_attention_needed(args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    try:
        mp = args.get("marketplace")
        svc = _get_seller_service()
        res = svc.get_attention_needed(marketplace=mp)
        return json.dumps(res, indent=2)
    except Exception as e:
        return tool_error(f"Failed to fetch attention-needed seller orders: {str(e)}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="seller_get_orders",
    toolset="seller",
    schema=SELLER_GET_ORDERS_SCHEMA,
    handler=_handle_seller_get_orders,
    check_fn=_check_seller_available,
    emoji="📦",
)

registry.register(
    name="seller_get_inventory",
    toolset="seller",
    schema=SELLER_GET_INVENTORY_SCHEMA,
    handler=_handle_seller_get_inventory,
    check_fn=_check_seller_available,
    emoji="🏭",
)

registry.register(
    name="seller_get_sales",
    toolset="seller",
    schema=SELLER_GET_SALES_SCHEMA,
    handler=_handle_seller_get_sales,
    check_fn=_check_seller_available,
    emoji="📈",
)

registry.register(
    name="seller_check_health",
    toolset="seller",
    schema=SELLER_CHECK_HEALTH_SCHEMA,
    handler=_handle_seller_check_health,
    check_fn=_check_seller_available,
    emoji="🩺",
)

registry.register(
    name="seller_get_attention_needed",
    toolset="seller",
    schema=SELLER_GET_ATTENTION_NEEDED_SCHEMA,
    handler=_handle_seller_get_attention_needed,
    check_fn=_check_seller_available,
    emoji="🚨",
)
