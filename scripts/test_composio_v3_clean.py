import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json"
}

print("=== Composio API v3 Exploration ===")

def check_endpoint(endpoint):
    url = f"https://backend.composio.dev/api/v3/{endpoint}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"GET {endpoint} -> Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Result sample: {json.dumps(data, indent=2)[:600]}\n")
            return data
        else:
            print(f"  Response: {r.text[:300]}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
    return None

check_endpoint("tools")
check_endpoint("connected_accounts")
check_endpoint("apps")
check_endpoint("actions")
