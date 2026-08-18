import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json"
}

def get_connected_accounts():
    url = "https://backend.composio.dev/api/v3/connected_accounts"
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        data = r.json()
        items = data.get("items", [])
        print(f"=== Connected Accounts ({len(items)}) ===")
        for item in items:
            toolkit = item.get("toolkit", {}).get("slug", "unknown")
            acc_id = item.get("id")
            status = item.get("status")
            user_id = item.get("user_id")
            alias = item.get("alias")
            word_id = item.get("word_id")
            print(f"  - App: {toolkit}")
            print(f"    Account ID: {acc_id}")
            print(f"    Status: {status}")
            print(f"    User ID: {user_id}")
            print(f"    Word ID / Alias: {word_id} / {alias}")
            print("    ---")
        return items
    else:
        print(f"Failed to fetch connected accounts: {r.status_code} {r.text}")
        return []

def search_telegram_tools():
    url = "https://backend.composio.dev/api/v3/tools"
    # Filter by toolkit if possible or query
    r = requests.get(url, headers=headers, params={"toolkit": "telegram"}, timeout=10)
    if r.status_code != 200:
        r = requests.get(url, headers=headers, timeout=10)
    
    if r.status_code == 200:
        data = r.json()
        items = data.get("items", [])
        print(f"\n=== Total Available Tools ({len(items)}) ===")
        telegram_tools = [t for t in items if "telegram" in t.get("toolkit", {}).get("slug", "").lower() or "telegram" in t.get("slug", "").lower()]
        print(f"=== Telegram Tools ({len(telegram_tools)}) ===")
        for t in telegram_tools:
            print(f"  - [{t.get('slug')}] {t.get('name')}: {t.get('description')}")
        
        # Print sample toolkits available
        toolkits = set(t.get("toolkit", {}).get("slug") for t in items if t.get("toolkit"))
        print(f"\n=== Sample Available Toolkits ({len(toolkits)}) ===")
        print(list(toolkits)[:20])

if __name__ == "__main__":
    get_connected_accounts()
    search_telegram_tools()
