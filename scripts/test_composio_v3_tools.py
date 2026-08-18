import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json"
}

def search_tools(query):
    url = "https://backend.composio.dev/api/v3/tools"
    r = requests.get(url, headers=headers, params={"search": query, "limit": 10}, timeout=10)
    print(f"=== Search Tools Query: '{query}' -> Status: {r.status_code} ===")
    if r.status_code == 200:
        items = r.json().get("items", [])
        print(f"Found {len(items)} tools:")
        for t in items:
            print(f"  - [{t.get('slug')}] {t.get('name')} (Toolkit: {t.get('toolkit', {}).get('slug')})")
    else:
        print(f"Error: {r.text[:200]}")
    print()

search_tools("gmail")
search_tools("googlecalendar")
search_tools("maps")
search_tools("send email")
