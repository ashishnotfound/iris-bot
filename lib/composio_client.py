"""
lib/composio_client.py — Composio API v3 Tool Integration for Hermes Agent

Provides:
  - Account & Toolkit Discovery
  - Open AI / Hermes Tool Schema Conversion
  - Tool Execution against Composio API v3
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

COMPOSIO_BASE_URL = "https://backend.composio.dev/api/v3"

class ComposioClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("COMPOSIO_API_KEY", "")).strip()
        self.headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_connected_accounts(self) -> List[Dict[str, Any]]:
        """Fetch active connected accounts for this Composio API key."""
        if not self.is_configured():
            return []
        url = f"{COMPOSIO_BASE_URL}/connected_accounts"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code == 200:
                items = r.json().get("items", [])
                return [acc for acc in items if acc.get("status") == "ACTIVE"]
        except Exception as e:
            logger.error(f"Failed to fetch Composio connected accounts: {e}")
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

    def convert_tool_to_schema(self, tool_def: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Composio v3 tool definition into OpenAI function calling schema."""
        slug = tool_def.get("slug", "")
        description = tool_def.get("description") or tool_def.get("human_description") or slug
        input_params = tool_def.get("input_parameters", {})

        return {
            "type": "function",
            "function": {
                "name": f"composio_{slug.lower()}",
                "description": description[:1024],
                "parameters": input_params
            }
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

        # Strip 'composio_' prefix if present
        clean_slug = tool_slug.replace("composio_", "").upper()
        clean_lower = clean_slug.lower()

        # Automatic resolution of connected_account_id & user_id if omitted
        if not connected_account_id or not user_id:
            accounts = self.get_connected_accounts()
            matched_acc: Optional[Dict[str, Any]] = None

            # Attempt matching toolkit slug against clean_lower tool name
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
        payload: Dict[str, Any] = {
            "arguments": arguments or {}
        }

        if connected_account_id:
            payload["connected_account_id"] = connected_account_id
        if user_id:
            payload["user_id"] = user_id
            payload["entity_id"] = user_id

        try:
            r = requests.post(url, headers=self.headers, json=payload, timeout=20)
            if r.status_code == 200:
                return r.json()
            else:
                return {
                    "successful": False,
                    "status_code": r.status_code,
                    "error": r.text
                }
        except Exception as e:
            logger.error(f"Error executing Composio tool {clean_slug}: {e}")
            return {"successful": False, "error": str(e)}

