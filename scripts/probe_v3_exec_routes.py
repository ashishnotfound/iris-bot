import requests
import json

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

payload = {
    "connected_account_id": "ca_BMEwAbmU4sdY",
    "arguments": {"max_results": 3}
}

urls = [
    "https://backend.composio.dev/api/v3/tools/GMAIL_FETCH_EMAILS",
    "https://backend.composio.dev/api/v3/tools/GMAIL_FETCH_EMAILS/execute",
    "https://backend.composio.dev/api/v3/tools/execute/GMAIL_FETCH_EMAILS",
    "https://backend.composio.dev/api/v3/toolkits/gmail/tools/GMAIL_FETCH_EMAILS/execute",
    "https://backend.composio.dev/api/v3/tools/GMAIL_FETCH_EMAILS/executions",
    "https://backend.composio.dev/api/v3/execute/GMAIL_FETCH_EMAILS",
    "https://backend.composio.dev/api/v3/actions/GMAIL_FETCH_EMAILS/execute",
    "https://backend.composio.dev/api/v3/actions/execute",
]

for url in urls:
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=5)
        print(f"POST {url} -> Status: {r.status_code}")
        print(f"  Response: {r.text[:300]}\n")
    except Exception as e:
        print(f"  Error {url}: {e}\n")
