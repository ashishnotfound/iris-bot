import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

urls = [
    "https://backend.composio.dev/api/v1/actions/GMAIL_FETCH_EMAILS/execute",
    "https://backend.composio.dev/api/v1/actions/GOOGLECALENDAR_EVENTS_LIST/execute",
    "https://backend.composio.dev/api/v1/actions/execute",
]

payloads = [
    {"connectedAccountId": "ca_BMEwAbmU4sdY", "input": {"max_results": 3}},
    {"connected_account_id": "ca_BMEwAbmU4sdY", "arguments": {"max_results": 3}},
]

for url in urls:
    for p in payloads:
        try:
            r = requests.post(url, headers=headers, json=p, timeout=5)
            print(f"POST {url} -> Status: {r.status_code}")
            print(f"  Response: {r.text[:400]}\n")
            if r.status_code == 200:
                print("SUCCESS!")
                break
        except Exception as e:
            print(f"  Error: {e}\n")
