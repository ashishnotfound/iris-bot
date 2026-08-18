import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Authorization": f"Bearer {COMPOSIO_API_KEY}",
    "Accept": "application/json"
}

print("=== Testing Composio API v3 Endpoints ===")

endpoints = [
    "https://backend.composio.dev/api/v3/connectedAccounts",
    "https://backend.composio.dev/api/v3/apps",
    "https://backend.composio.dev/api/v3/actions",
    "https://backend.composio.dev/api/v3/users/me",
    "https://backend.composio.dev/api/v3/tools",
    "https://backend.composio.dev/api/v3/integrations"
]

for url in endpoints:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"URL: {url} -> Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Response sample: {json.dumps(data, indent=2)[:500]}\n")
        else:
            print(f"  Response: {r.text[:300]}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
