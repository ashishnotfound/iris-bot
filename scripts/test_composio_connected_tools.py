import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json"
}

toolkits = ["gmail", "googlecalendar", "google_maps"]

for tk in toolkits:
    print(f"=== Fetching Tools for Toolkit: {tk} ===")
    url = f"https://backend.composio.dev/api/v3/tools"
    try:
        r = requests.get(url, headers=headers, params={"toolkit": tk}, timeout=10)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            items = r.json().get("items", [])
            print(f"  Found {len(items)} tools for {tk}:")
            for t in items[:8]:
                print(f"  - [{t.get('slug')}] {t.get('name')}: {t.get('description')[:70] if t.get('description') else ''}")
        else:
            print(f"  Response: {r.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
