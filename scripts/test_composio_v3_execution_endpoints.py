import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json"
}

endpoints_to_test = [
    ("POST", "https://backend.composio.dev/api/v3/execute"),
    ("POST", "https://backend.composio.dev/api/v3/tools/execute"),
    ("POST", "https://backend.composio.dev/api/v3/actions/execute"),
    ("POST", "https://backend.composio.dev/api/v3/tools/0CODEKIT_CALCULATE_BMI/run"),
    ("POST", "https://backend.composio.dev/api/v3/tools/run"),
    ("GET",  "https://backend.composio.dev/api/v3/tools/GMAIL_FETCH_EMAILS"),
    ("GET",  "https://backend.composio.dev/api/v3/actions/GMAIL_FETCH_EMAILS"),
]

for method, url in endpoints_to_test:
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=5)
        else:
            r = requests.post(url, headers=headers, json={"tool_name": "0CODEKIT_CALCULATE_BMI", "arguments": {"weight": 70, "height": 170}}, timeout=5)
        print(f"{method} {url} -> Status: {r.status_code}")
        print(f"  Response: {r.text[:250]}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
