import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"
USER_ID = "pg-test-11ca6b44-536a-4ccf-b10f-1decafc22371"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def execute_v3_tool(tool_slug, connected_account_id, arguments=None):
    url = f"https://backend.composio.dev/api/v3/tools/execute/{tool_slug}"
    payload = {
        "user_id": USER_ID,
        "entity_id": USER_ID,
        "connected_account_id": connected_account_id,
        "arguments": arguments or {}
    }
    print(f"=== Executing Composio Tool: {tool_slug} ===")
    print(f"  URL: {url}")
    print(f"  Payload: {json.dumps(payload)}")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"  Status Code: {r.status_code}")
        data = r.json()
        print(f"  Result Output:\n{json.dumps(data, indent=2)[:1000]}\n")
        return data
    except Exception as e:
        print(f"  Execution Error: {e}\n")
        return None

if __name__ == "__main__":
    # Test 1: Fetch Emails from connected Gmail
    execute_v3_tool("GMAIL_FETCH_EMAILS", "ca_BMEwAbmU4sdY", {"max_results": 2})

    # Test 2: List Events from connected Google Calendar
    execute_v3_tool("GOOGLECALENDAR_EVENTS_LIST", "ca_UyXnrAXsaGqO", {"calendarId": "primary", "maxResults": 2})
