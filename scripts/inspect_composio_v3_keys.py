import requests
import json

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json"
}

url = "https://backend.composio.dev/api/v3/tools/GMAIL_FETCH_EMAILS"
r = requests.get(url, headers=headers, timeout=10)
if r.status_code == 200:
    data = r.json()
    print("Top-level keys of tool definition:")
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            print(f"  - {k}: {type(v).__name__}")
        else:
            print(f"  - {k}: {v}")
