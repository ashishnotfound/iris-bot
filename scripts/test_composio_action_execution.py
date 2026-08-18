import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def execute_composio_action(action_slug, connected_account_id, arguments=None):
    url = f"https://backend.composio.dev/api/v3/tools/{action_slug}/execute"
    payload = {
        "connected_account_id": connected_account_id,
        "arguments": arguments or {}
    }
    print(f"=== Executing Action: {action_slug} (Account: {connected_account_id}) ===")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Result: {json.dumps(data, indent=2)[:800]}\n")
        return data
    except Exception as e:
        print(f"Error: {e}\n")
        return None

if __name__ == "__main__":
    # Test Google Calendar Events List
    execute_composio_action("GOOGLECALENDAR_EVENTS_LIST", "ca_UyXnrAXsaGqO", {"calendarId": "primary", "maxResults": 3})
    
    # Test Gmail Fetch Emails
    execute_composio_action("GMAIL_FETCH_EMAILS", "ca_BMEwAbmU4sdY", {"max_results": 3})
