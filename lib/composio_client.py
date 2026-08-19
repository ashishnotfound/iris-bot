"""
lib/composio_client.py — Composio API v3 Tool Integration for Hermes Agent

Provides:
  - Account & Toolkit Discovery
  - Open AI / Hermes Tool Schema Conversion
  - Tool Execution against Composio API v3
  - Tool Classification (READ_ONLY vs MUTATING / CONSEQUENTIAL)
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

COMPOSIO_BASE_URL = "https://backend.composio.dev/api/v3"

# Keyword mapping to Composio toolkit slugs
TOOLKIT_KEYWORD_MAP = {
    "email": "gmail",
    "mail": "gmail",
    "gmail": "gmail",
    "send email": "gmail",
    "calendar": "googlecalendar",
    "event": "googlecalendar",
    "schedule": "googlecalendar",
    "meeting": "googlecalendar",
    "instagram": "instagram",
    "insta": "instagram",
    "post": "instagram",
    "maps": "google_maps",
    "location": "google_maps",
    "directions": "google_maps",
    "browse": "browserbase_tool",
    "web": "browserbase_tool",
}

# Explicit action categories
READ_ONLY_PATTERNS = [
    "fetch", "get", "list", "read", "search", "info", "query", "find", "show"
]

CONSEQUENTIAL_KEYWORDS = [
    "send", "delete", "create", "remove", "update", "modify", "patch", "post", "cancel", "purchase", "pay", "reply", "exec", "write"
]


def is_consequential_action(tool_name: str) -> bool:
    """Determine whether a tool call represents a consequential external action requiring confirmation.

    Security Rule:
      - Explicit read-only tools (fetch, get, list, search, info) are safe.
      - Known mutating/external tools require confirmation.
      - UNKNOWN tools default to CONSEQUENTIAL (fail-safe security principle).
    """
    clean = tool_name.replace("composio_", "").lower()

    # 1. Check if it contains a consequential mutating keyword
    for kw in CONSEQUENTIAL_KEYWORDS:
        if kw in clean:
            return True

    # 2. Check explicit Read-Only Whitelist
    for ro in READ_ONLY_PATTERNS:
        if ro in clean:
            return False

    # 3. Fail-Safe Default: Unknown tools require confirmation
    logger.warning("Unclassified/Unknown tool '%s' defaulting to CONSEQUENTIAL for safety.", tool_name)
    return True


def validate_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate tool arguments before execution."""
    clean = tool_name.replace("composio_", "").lower()

    if "email" in clean and ("send" in clean or "draft" in clean):
        recipient = arguments.get("recipient_email") or arguments.get("to") or arguments.get("recipient")
        if not recipient:
            return False, "Recipient email address is required."

    return True, None


class ComposioClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("COMPOSIO_API_KEY", "")).strip()
        self.headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._connected_accounts_cache: Optional[List[Dict[str, Any]]] = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_connected_accounts(self) -> List[Dict[str, Any]]:
        """Fetch active connected accounts for this Composio API key."""
        if not self.is_configured():
            return []
        if self._connected_accounts_cache is not None:
            return self._connected_accounts_cache
        url = f"{COMPOSIO_BASE_URL}/connected_accounts"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code == 200:
                items = r.json().get("items", [])
                active = [acc for acc in items if acc.get("status") == "ACTIVE"]
                self._connected_accounts_cache = active
                return active
        except Exception as e:
            logger.error(f"Failed to fetch Composio connected accounts: {e}")
        return []

    def get_tools_for_toolkit(self, toolkit_slug: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch available action tools for a specific toolkit (e.g. 'gmail')."""
        if not self.is_configured():
            return []
        url = f"{COMPOSIO_BASE_URL}/tools"
        params = {"toolkit_slug": toolkit_slug, "limit": limit}
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("items", [])
        except Exception as e:
            logger.error(f"Failed to fetch Composio tools for toolkit {toolkit_slug}: {e}")
        return []

    def get_tools_for_query(self, query: str = "", limit: int = 15) -> List[Dict[str, Any]]:
        """Search available tools from Composio v3 catalog."""
        if not self.is_configured():
            return []
        url = f"{COMPOSIO_BASE_URL}/tools"
        params = {"limit": limit}
        if query:
            params["search"] = query
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("items", [])
        except Exception as e:
            logger.error(f"Failed to fetch Composio tools: {e}")
        return []

    def get_tool_schemas_for_request(self, user_message: str) -> List[Dict[str, Any]]:
        """Retrieve OpenAI function schemas for tools relevant to the user request.

        Matches active connected accounts and user query intent.
        """
        if not self.is_configured():
            return []

        accounts = self.get_connected_accounts()
        if not accounts:
            return []

        active_toolkits = {
            (acc.get("toolkit", {}).get("slug") or "").lower() for acc in accounts
        }

        # Match intent keywords against active connected toolkits
        msg_lower = user_message.lower()
        target_toolkits = set()
        for kw, tk_slug in TOOLKIT_KEYWORD_MAP.items():
            if kw in msg_lower and tk_slug in active_toolkits:
                target_toolkits.add(tk_slug)

        # Only expose tools if specific intent keywords matched
        if not target_toolkits:
            return []

        tool_defs: List[Dict[str, Any]] = []
        for tk in target_toolkits:
            tk_tools = self.get_tools_for_toolkit(tk, limit=10)
            tool_defs.extend(tk_tools)

        schemas: List[Dict[str, Any]] = []
        seen_names = set()
        for t in tool_defs:
            schema = self.convert_tool_to_schema(t)
            name = schema["function"]["name"]
            if name not in seen_names:
                seen_names.add(name)
                schemas.append(schema)

        return schemas[:20]

    def convert_tool_to_schema(self, tool_def: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Composio v3 tool definition into OpenAI function calling schema."""
        slug = tool_def.get("slug", "")
        description = tool_def.get("description") or tool_def.get("human_description") or slug
        input_params = tool_def.get("input_parameters", {})

        # Ensure parameters has valid type and properties
        if not isinstance(input_params, dict) or "type" not in input_params:
            input_params = {
                "type": "object",
                "properties": input_params.get("properties", {}) if isinstance(input_params, dict) else {},
            }

        return {
            "type": "function",
            "function": {
                "name": f"composio_{slug.lower()}",
                "description": str(description)[:1024],
                "parameters": input_params,
            },
        }

    def execute_tool(
        self,
        tool_slug: str,
        arguments: Dict[str, Any],
        connected_account_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a Composio v3 action tool. Automatically resolves connected_account_id
        and user_id/entity_id from active connected accounts if omitted."""
        if not self.is_configured():
            return {"successful": False, "error": "Composio API key not configured"}

        # Validate arguments before execution
        valid, val_err = validate_tool_arguments(tool_slug, arguments)
        if not valid:
            return {"successful": False, "error": f"Invalid tool arguments: {val_err}"}

        # Strip 'composio_' prefix if present
        clean_slug = tool_slug.replace("composio_", "").upper()
        clean_lower = clean_slug.lower()

        # Automatic resolution of connected_account_id & user_id if omitted
        if not connected_account_id or not user_id:
            accounts = self.get_connected_accounts()
            matched_acc: Optional[Dict[str, Any]] = None

            for acc in accounts:
                t_slug = (acc.get("toolkit", {}).get("slug") or "").lower()
                if t_slug and (t_slug in clean_lower or clean_lower.startswith(t_slug.replace("_tool", ""))):
                    matched_acc = acc
                    break

            if not matched_acc and accounts:
                matched_acc = accounts[0]

            if matched_acc:
                if not connected_account_id:
                    connected_account_id = matched_acc.get("id")
                if not user_id:
                    user_id = matched_acc.get("user_id")

        url = f"{COMPOSIO_BASE_URL}/tools/execute/{clean_slug}"
        payload: Dict[str, Any] = {"arguments": arguments or {}}

        if connected_account_id:
            payload["connected_account_id"] = connected_account_id
        if user_id:
            payload["user_id"] = user_id
            payload["entity_id"] = user_id

        try:
            r = requests.post(url, headers=self.headers, json=payload, timeout=25)
            if r.status_code == 200:
                res = r.json()
                if isinstance(res, dict):
                    if "successful" not in res:
                        res["successful"] = True
                    return res
                return {"successful": True, "data": res}
            else:
                return {
                    "successful": False,
                    "status_code": r.status_code,
                    "error": r.text[:500],
                }
        except Exception as e:
            logger.error(f"Error executing Composio tool {clean_slug}: {e}")
            return {"successful": False, "error": str(e)}
