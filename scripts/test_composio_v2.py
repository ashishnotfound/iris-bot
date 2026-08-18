import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Authorization": f"Bearer {COMPOSIO_API_KEY}",
    "Accept": "application/json"
}

print("=== Testing Composio API v2 Endpoints ===")

endpoints = [
    "https://backend.composio.dev/api/v2/connectedAccounts",
    "https://backend.composio.dev/api/v2/apps",
    "https://backend.composio.dev/api/v2/actions",
    "https://backend.composio.dev/api/v2/users/me"
]

for url in endpoints:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"URL: {url} -> Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Response sample: {json.dumps(data, indent=2)[:400]}\n")
        else:
            print(f"  Response: {r.text[:200]}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
