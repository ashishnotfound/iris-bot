"""Comprehensive test suite for IRIS Amazon Seller + Flipkart Seller integration layer."""

import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from seller.amazon import AmazonSellerService, AmazonTokenManager
from seller.base import (
    ErrorCategory,
    IntegrationState,
    NormalizedInventoryItem,
    NormalizedOrder,
    NormalizedOrderItem,
    NormalizedSalesSummary,
    SellerHealthStatus,
)
from seller.flipkart import FlipkartSellerService
from seller.health import SellerHealthMonitor
from seller.heartbeat import AmazonHeartbeatRunner
from seller.notifications import SellerNotificationManager
from seller.unified import UnifiedSellerService
from tools.seller_tools import (
    _check_seller_available,
    _handle_seller_check_health,
    _handle_seller_get_attention_needed,
    _handle_seller_get_inventory,
    _handle_seller_get_orders,
    _handle_seller_get_sales,
)


@pytest.fixture(autouse=True)
def setup_temp_health_db(tmp_path):
    """Provide isolated SQLite database for health monitor tests."""
    db_file = str(tmp_path / "test_seller_health.db")
    SellerHealthMonitor.reset_instance()
    monitor = SellerHealthMonitor(db_path=db_file)
    yield monitor
    SellerHealthMonitor.reset_instance()


def _make_mock_response(data_dict, status=200):
    mock = MagicMock()
    mock.status = status
    mock.read.return_value = json.dumps(data_dict).encode("utf-8")
    cm = MagicMock()
    cm.__enter__.return_value = mock
    return cm


# ---------------------------------------------------------------------------
# Amazon Token Manager Tests
# ---------------------------------------------------------------------------

def test_amazon_token_manager_missing_credentials():
    """Verify ValueError raised when credentials are missing."""
    tm = AmazonTokenManager()
    with patch("seller.amazon.get_secret", return_value=""):
        with pytest.raises(ValueError, match="Amazon SP-API credentials missing"):
            tm.get_access_token()


def test_amazon_token_manager_success():
    """Verify access token acquisition and expiration caching."""
    tm = AmazonTokenManager()

    mock_resp = _make_mock_response({
        "access_token": "Atza|test_token_123",
        "expires_in": 3600,
        "token_type": "bearer",
    })

    with patch("seller.amazon.get_secret", side_effect=lambda key, default="": f"mock_{key}"):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            token = tm.get_access_token()
            assert token == "Atza|test_token_123"

            # Subsequent call within expiry window should return cached token without HTTP request
            with patch("urllib.request.urlopen") as mock_urlopen_2:
                cached_token = tm.get_access_token()
                assert cached_token == "Atza|test_token_123"
                mock_urlopen_2.assert_not_called()


def test_amazon_token_manager_force_refresh():
    """Verify force_refresh re-fetches token even if unexpired."""
    tm = AmazonTokenManager()

    mock_resp_1 = _make_mock_response({"access_token": "token_v1", "expires_in": 3600})
    mock_resp_2 = _make_mock_response({"access_token": "token_v2", "expires_in": 3600})

    with patch("seller.amazon.get_secret", side_effect=lambda key, default="": f"mock_{key}"):
        with patch("urllib.request.urlopen", side_effect=[mock_resp_1, mock_resp_2]):
            t1 = tm.get_access_token()
            assert t1 == "token_v1"

            t2 = tm.get_access_token(force_refresh=True)
            assert t2 == "token_v2"


# ---------------------------------------------------------------------------
# Amazon Seller Service & SP-API Tests
# ---------------------------------------------------------------------------

def test_amazon_service_get_orders_success(setup_temp_health_db):
    """Test normalized order retrieval from Amazon SP-API."""
    amz = AmazonSellerService(health_monitor=setup_temp_health_db)

    mock_orders_response = {
        "payload": {
            "Orders": [
                {
                    "AmazonOrderId": "902-1111111-2222222",
                    "PurchaseDate": "2026-08-18T10:00:00Z",
                    "OrderStatus": "Unshipped",
                    "FulfillmentChannel": "MFN",
                    "OrderTotal": {"Amount": "1499.00", "CurrencyCode": "INR"},
                    "BuyerInfo": {"BuyerName": "Jane Doe"},
                    "ShippingAddress": {"City": "Mumbai"},
                },
                {
                    "AmazonOrderId": "902-3333333-4444444",
                    "PurchaseDate": "2026-08-17T12:00:00Z",
                    "OrderStatus": "Shipped",
                    "FulfillmentChannel": "AFN",  # FBA
                    "OrderTotal": {"Amount": "2999.00", "CurrencyCode": "INR"},
                },
            ]
        }
    }

    with patch.object(amz.token_manager, "get_access_token", return_value="mock_token"):
        with patch.object(amz, "_make_api_request", return_value=mock_orders_response):
            orders = amz.get_orders()
            assert len(orders) == 2

            # First order: MFN, pending dispatch -> needs_attention = True
            o1 = orders[0]
            assert o1.order_id == "902-1111111-2222222"
            assert o1.marketplace == "amazon"
            assert o1.fulfillment_channel == "MFN"
            assert o1.total_amount == 1499.00
            assert o1.needs_attention is True

            # Second order: FBA -> fulfillment_channel = "FBA"
            o2 = orders[1]
            assert o2.order_id == "902-3333333-4444444"
            assert o2.fulfillment_channel == "FBA"
            assert o2.needs_attention is False


def test_amazon_service_401_retry(setup_temp_health_db):
    """Test that HTTP 401 triggers token refresh and retries request."""
    amz = AmazonSellerService(health_monitor=setup_temp_health_db)

    err_401 = urllib.error.HTTPError("url", 401, "Unauthorized", {}, MagicMock(read=lambda: b"Unauthorized"))
    mock_success = _make_mock_response({"payload": {"Orders": []}})

    with patch.object(amz.token_manager, "get_access_token", return_value="mock_token"):
        with patch("urllib.request.urlopen", side_effect=[err_401, mock_success]):
            orders = amz.get_orders()
            assert orders == []


# ---------------------------------------------------------------------------
# Flipkart Seller Service Tests
# ---------------------------------------------------------------------------

def test_flipkart_service_get_orders_success(setup_temp_health_db):
    """Test normalized order retrieval from Flipkart Seller API."""
    fk = FlipkartSellerService(health_monitor=setup_temp_health_db)

    mock_fk_response = {
        "shipments": [
            {
                "shipmentId": "FK-999888",
                "status": "APPROVED",
                "orderDate": "2026-08-18T09:00:00Z",
                "fulfillmentType": "DEFAULT",
                "orderItems": [
                    {
                        "sku": "WIRELESS-MOUSE-BL",
                        "title": "Ergonomic Wireless Mouse",
                        "quantity": 2,
                        "price": 750.00,
                    }
                ],
            }
        ]
    }

    with patch.object(fk, "_get_access_token", return_value="mock_fk_token"):
        with patch.object(fk, "_make_api_request", return_value=mock_fk_response):
            orders = fk.get_orders()
            assert len(orders) == 1
            o = orders[0]
            assert o.order_id == "FK-999888"
            assert o.marketplace == "flipkart"
            assert o.total_amount == 1500.00
            assert o.needs_attention is True


# ---------------------------------------------------------------------------
# Unified Seller Service Tests
# ---------------------------------------------------------------------------

def test_unified_seller_service_combined(setup_temp_health_db):
    """Test combining orders from Amazon and Flipkart with intact marketplace identity."""
    amz_mock = MagicMock(spec=AmazonSellerService)
    fk_mock = MagicMock(spec=FlipkartSellerService)

    amz_mock.get_orders.return_value = [
        NormalizedOrder(
            order_id="AMZ-101",
            marketplace="amazon",
            order_date="2026-08-18T10:00:00Z",
            status="Unshipped",
            total_amount=500.0,
            currency="INR",
        )
    ]
    fk_mock.get_orders.return_value = [
        NormalizedOrder(
            order_id="FK-202",
            marketplace="flipkart",
            order_date="2026-08-18T11:00:00Z",
            status="APPROVED",
            total_amount=1200.0,
            currency="INR",
        )
    ]

    svc = UnifiedSellerService(
        amazon_service=amz_mock,
        flipkart_service=fk_mock,
        health_monitor=setup_temp_health_db,
    )

    res = svc.get_orders(marketplace="combined")
    orders = res["orders"]
    summary = res["summary"]

    assert len(orders) == 2
    assert summary["amazon"] == 1
    assert summary["flipkart"] == 1
    assert summary["total"] == 2
    assert res["errors"] is None

    sources = {o["marketplace"] for o in orders}
    assert sources == {"amazon", "flipkart"}


def test_unified_seller_service_partial_outage(setup_temp_health_db):
    """Test graceful handling when Amazon is down but Flipkart succeeds."""
    amz_mock = MagicMock(spec=AmazonSellerService)
    fk_mock = MagicMock(spec=FlipkartSellerService)

    amz_mock.get_orders.side_effect = ConnectionError("Amazon SP-API Connection Failed")
    fk_mock.get_orders.return_value = [
        NormalizedOrder(
            order_id="FK-303",
            marketplace="flipkart",
            order_date="2026-08-18T11:00:00Z",
            status="APPROVED",
            total_amount=800.0,
            currency="INR",
        )
    ]

    svc = UnifiedSellerService(
        amazon_service=amz_mock,
        flipkart_service=fk_mock,
        health_monitor=setup_temp_health_db,
    )

    res = svc.get_orders(marketplace="combined")
    orders = res["orders"]
    summary = res["summary"]
    errors = res["errors"]

    assert len(orders) == 1
    assert orders[0]["order_id"] == "FK-303"
    assert summary["amazon"] == 0
    assert summary["flipkart"] == 1
    assert "amazon" in errors
    assert "Amazon SP-API Connection Failed" in errors["amazon"]


# ---------------------------------------------------------------------------
# Health Monitor & Notification Tests
# ---------------------------------------------------------------------------

def test_health_monitor_persistence_and_transitions(setup_temp_health_db):
    """Test health monitor SQLite state persistence and transition tracking."""
    monitor = setup_temp_health_db

    # Initial state
    h0 = monitor.get_health("amazon")
    assert h0.state == IntegrationState.UNKNOWN

    # Update success
    monitor.update_success("amazon")
    h1 = monitor.get_health("amazon")
    assert h1.state == IntegrationState.CONNECTED
    assert h1.failure_count == 0

    # Update authentication failure
    monitor.update_failure("amazon", ErrorCategory.AUTHENTICATION, "Token expired")
    h2 = monitor.get_health("amazon")
    assert h2.state == IntegrationState.AUTHENTICATION_REQUIRED
    assert h2.failure_count == 1


def test_notification_manager_cooldown(setup_temp_health_db):
    """Test alert deduplication and Telegram alert dispatch."""
    notifier = SellerNotificationManager(health_monitor=setup_temp_health_db, cooldown_seconds=60.0)

    with patch.object(notifier, "send_telegram_alert", return_value=True) as mock_send:
        # First transition: CONNECTED -> OFFLINE
        alert1 = notifier.handle_state_transition(
            "amazon", IntegrationState.CONNECTED, IntegrationState.OFFLINE, "Server down"
        )
        assert alert1 is not None
        assert mock_send.call_count == 1

        # Second transition to SAME state within cooldown window -> Suppressed
        alert2 = notifier.handle_state_transition(
            "amazon", IntegrationState.CONNECTED, IntegrationState.OFFLINE, "Server down"
        )
        assert alert2 is None
        assert mock_send.call_count == 1

        # Transition to RESTORED state -> Emits recovery notification
        alert3 = notifier.handle_state_transition(
            "amazon", IntegrationState.OFFLINE, IntegrationState.CONNECTED
        )
        assert alert3 is not None
        assert "Restored" in alert3
        assert mock_send.call_count == 2


# ---------------------------------------------------------------------------
# 25-minute Amazon Idle Heartbeat Tests
# ---------------------------------------------------------------------------

def test_amazon_heartbeat_runner(setup_temp_health_db):
    """Test 25-minute Amazon idle heartbeat runner and advisory lock execution."""
    amz_mock = MagicMock(spec=AmazonSellerService)
    amz_mock.check_health.return_value = SellerHealthStatus(
        marketplace="amazon",
        state=IntegrationState.CONNECTED,
    )

    runner = AmazonHeartbeatRunner(
        amazon_service=amz_mock,
        health_monitor=setup_temp_health_db,
    )

    # Execute heartbeat once
    ran = runner.run_heartbeat_once()
    assert ran is True
    amz_mock.check_health.assert_called_once()

    h = setup_temp_health_db.get_health("amazon")
    assert h.last_heartbeat is not None


# ---------------------------------------------------------------------------
# Hermes Tool Layer Tests
# ---------------------------------------------------------------------------

def test_check_seller_available_gating():
    """Verify tool check_fn evaluates True when seller credentials present, False otherwise."""
    with patch("tools.seller_tools.get_secret", return_value=""):
        assert _check_seller_available() is False

    with patch("tools.seller_tools.get_secret", side_effect=lambda key, default="": "secret" if key in ("AMAZON_CLIENT_ID", "AMAZON_CLIENT_SECRET", "AMAZON_REFRESH_TOKEN") else ""):
        assert _check_seller_available() is True


def test_seller_tool_handlers():
    """Test execution of seller tool handlers."""
    mock_unified = MagicMock(spec=UnifiedSellerService)
    mock_unified.get_orders.return_value = {"orders": [], "summary": {"total": 0}}
    mock_unified.check_all_health.return_value = {
        "amazon": {"state": "CONNECTED"},
        "flipkart": {"state": "CONNECTED"},
    }

    with patch("tools.seller_tools._get_seller_service", return_value=mock_unified):
        res_orders = _handle_seller_get_orders({"marketplace": "combined"})
        assert "summary" in res_orders

        res_health = _handle_seller_check_health({"marketplace": "combined"})
        assert "amazon" in res_health
