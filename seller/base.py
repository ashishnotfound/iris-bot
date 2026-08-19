"""Base abstractions, models, and interfaces for IRIS seller integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class MarketplaceType(str, Enum):
    AMAZON = "amazon"
    FLIPKART = "flipkart"
    COMBINED = "combined"


class IntegrationState(str, Enum):
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    OFFLINE = "OFFLINE"
    UNKNOWN = "UNKNOWN"


class ErrorCategory(str, Enum):
    AUTHENTICATION = "AUTHENTICATION"
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK_SERVER = "NETWORK_SERVER"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    PERSISTENT_OUTAGE = "PERSISTENT_OUTAGE"
    UNKNOWN = "UNKNOWN"


@dataclass
class NormalizedOrderItem:
    sku: str
    title: str
    quantity: int
    unit_price: float
    currency: str = "INR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "title": self.title,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "currency": self.currency,
        }


@dataclass
class NormalizedOrder:
    order_id: str
    marketplace: str  # "amazon" or "flipkart"
    order_date: str
    status: str  # "PENDING", "SHIPPED", "DELIVERED", "CANCELLED", etc.
    total_amount: float
    currency: str
    items: List[NormalizedOrderItem] = field(default_factory=list)
    fulfillment_channel: str = "MFN"  # "FBA", "EASY_SHIP", "MFN", "FLIPKART_FULFILLED"
    buyer_name: Optional[str] = None
    shipping_city: Optional[str] = None
    needs_attention: bool = False
    attention_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "marketplace": self.marketplace,
            "order_date": self.order_date,
            "status": self.status,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "items": [item.to_dict() for item in self.items],
            "fulfillment_channel": self.fulfillment_channel,
            "buyer_name": self.buyer_name,
            "shipping_city": self.shipping_city,
            "needs_attention": self.needs_attention,
            "attention_reason": self.attention_reason,
        }


@dataclass
class NormalizedInventoryItem:
    sku: str
    title: str
    marketplace: str
    quantity_available: int
    quantity_reserved: int = 0
    status: str = "IN_STOCK"  # "IN_STOCK", "LOW_STOCK", "OUT_OF_STOCK"
    unit_price: float = 0.0
    fulfillment_channel: str = "MFN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "title": self.title,
            "marketplace": self.marketplace,
            "quantity_available": self.quantity_available,
            "quantity_reserved": self.quantity_reserved,
            "status": self.status,
            "unit_price": self.unit_price,
            "fulfillment_channel": self.fulfillment_channel,
        }


@dataclass
class NormalizedSalesSummary:
    marketplace: str
    period: str
    total_orders: int
    total_revenue: float
    currency: str
    units_sold: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marketplace": self.marketplace,
            "period": self.period,
            "total_orders": self.total_orders,
            "total_revenue": self.total_revenue,
            "currency": self.currency,
            "units_sold": self.units_sold,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


@dataclass
class SellerHealthStatus:
    marketplace: str
    state: IntegrationState
    last_successful_request: Optional[str] = None
    last_failed_request: Optional[str] = None
    last_health_check: Optional[str] = None
    failure_count: int = 0
    last_error_category: ErrorCategory = ErrorCategory.UNKNOWN
    last_error_message: Optional[str] = None
    token_status: str = "VALID"
    retry_state: str = "IDLE"
    last_heartbeat: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marketplace": self.marketplace,
            "state": self.state.value if isinstance(self.state, Enum) else self.state,
            "last_successful_request": self.last_successful_request,
            "last_failed_request": self.last_failed_request,
            "last_health_check": self.last_health_check,
            "failure_count": self.failure_count,
            "last_error_category": self.last_error_category.value if isinstance(self.last_error_category, Enum) else self.last_error_category,
            "last_error_message": self.last_error_message,
            "token_status": self.token_status,
            "retry_state": self.retry_state,
            "last_heartbeat": self.last_heartbeat,
        }


class BaseSellerService(ABC):
    """Abstract base class for all marketplace seller integrations."""

    @property
    @abstractmethod
    def marketplace(self) -> str:
        """Return marketplace identifier e.g. 'amazon' or 'flipkart'."""
        pass

    @abstractmethod
    def get_orders(
        self,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        limit: int = 50,
    ) -> List[NormalizedOrder]:
        """Fetch normalized orders for the marketplace."""
        pass

    @abstractmethod
    def get_inventory(
        self,
        sku: Optional[str] = None,
        limit: int = 50,
    ) -> List[NormalizedInventoryItem]:
        """Fetch normalized inventory for the marketplace."""
        pass

    @abstractmethod
    def get_sales_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> NormalizedSalesSummary:
        """Fetch sales summary metrics for the given timeframe."""
        pass

    @abstractmethod
    def check_health(self) -> SellerHealthStatus:
        """Perform a lightweight API connectivity check and return current health state."""
        pass

    @abstractmethod
    def get_attention_needed_orders(self) -> List[NormalizedOrder]:
        """Fetch orders requiring immediate action (e.g. pending dispatch, cancellation requests)."""
        pass
