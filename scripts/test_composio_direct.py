import os
import json
import requests

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"
TELEGRAM_BOT_TOKEN = "8916712872:AAGPR875g-RrxX-1iwKsLORjS0p2Oifg5jE"

def test_telegram():
    print("=== 1. Telegram Bot API Test ===")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        r = requests.get(url, timeout=5)
        res = r.json()
        print(f"Status Code: {r.status_code}")
        print(f"Response: {json.dumps(res, indent=2)}")
    except Exception as e:
        print(f"Telegram error: {e}")

def test_composio_rest():
    print("\n=== 2. Composio REST API Test ===")
    headers = {"x-api-key": COMPOSIO_API_KEY, "Accept": "application/json"}
    
    # Check connected accounts
    urls_to_try = [
        "https://backend.composio.dev/api/v1/connectedAccounts",
        "https://api.composio.dev/v1/connectedAccounts"
    ]
    for url in urls_to_try:
        try:
            print(f"Querying: {url}")
            r = requests.get(url, headers=headers, timeout=5)
            print(f"  Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"  Data count: {len(data.get('items', data.get('data', [])) if isinstance(data, dict) else data)}")
                print(f"  Sample: {json.dumps(data, indent=2)[:500]}")
        except Exception as e:
            print(f"  Error querying {url}: {e}")

def test_composio_sdk():
    print("\n=== 3. Composio SDK Quick Test ===")
    try:
        from composio import ComposioToolSet, App
        toolset = ComposioToolSet(api_key=COMPOSIO_API_KEY)
        print("ComposioToolSet initialized successfully!")
        
        # Get telegram tools
        try:
            tools = toolset.get_tools(apps=[App.TELEGRAM])
            print(f"Found {len(tools)} Telegram tools via ComposioToolSet!")
            for t in tools[:5]:
                print(f" - Tool: {t.get('name') if isinstance(t, dict) else getattr(t, 'name', str(t))}")
        except Exception as e:
            print(f"Error fetching App.TELEGRAM tools: {e}")
            
    except Exception as e:
        print(f"Composio SDK error: {e}")

if __name__ == "__main__":
    test_telegram()
    test_composio_rest()
    test_composio_sdk()
