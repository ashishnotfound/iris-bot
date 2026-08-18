import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

endpoints_to_test = [
    ("POST", "https://backend.composio.dev/api/v3/tools/GMAIL_FETCH_EMAILS/execute"),
    ("POST", "https://backend.composio.dev/api/v3/tools/GMAIL_FETCH_EMAILS/run"),
    ("POST", "https://backend.composio.dev/api/v3/executions"),
    ("POST", "https://backend.composio.dev/api/v3/tools/execute"),
]

payload = {
    "connected_account_id": "ca_BMEwAbmU4sdY",
    "arguments": {
        "max_results": 3
    }
}

for method, url in endpoints_to_test:
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        print(f"POST {url} -> Status: {r.status_code}")
        print(f"  Response: {r.text[:300]}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
