import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"

headers = {
    "x-api-key": COMPOSIO_API_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def test_get_toolkits():
    print("=== 1. Fetching Toolkits from Composio v3 ===")
    url = "https://backend.composio.dev/api/v3/toolkits"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            items = data.get("items", [])
            print(f"Total Toolkits: {len(items)}")
            sample_slugs = [t.get("slug") for t in items[:15]]
            print(f"Sample Slugs: {sample_slugs}")
            
            # Check if telegram or messaging toolkits exist
            tg_toolkits = [t for t in items if "telegram" in t.get("slug", "").lower()]
            print(f"Telegram Toolkits: {tg_toolkits}")
    except Exception as e:
        print(f"Error: {e}")

def test_execute_tool():
    print("\n=== 2. Testing Execution of a Composio Tool (0CODEKIT_CALCULATE_BMI) ===")
    url = "https://backend.composio.dev/api/v3/tools/0CODEKIT_CALCULATE_BMI/execute"
    payload = {
        "arguments": {
            "weight": 70,
            "height": 175
        }
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"Execute Status: {r.status_code}")
        print(f"Result: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"Execution Error: {e}")

if __name__ == "__main__":
    test_get_toolkits()
    test_execute_tool()
