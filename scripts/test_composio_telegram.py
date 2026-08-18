import os
import json
import requests
from composio import Composio

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"
TELEGRAM_BOT_TOKEN = "8916712872:AAGPR875g-RrxX-1iwKsLORjS0p2Oifg5jE"

def test_telegram_bot_api():
    print("--- 1. Testing Telegram Bot API Direct Endpoint ---")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if data.get("ok"):
            bot_info = data.get("result", {})
            print(f"[SUCCESS] Telegram Bot Connected!")
            print(f"  Bot Name: {bot_info.get('first_name')}")
            print(f"  Bot Username: @{bot_info.get('username')}")
            print(f"  Bot ID: {bot_info.get('id')}")
        else:
            print(f"[FAIL] Telegram Bot API error: {data}")
    except Exception as e:
        print(f"[ERROR] Failed to reach Telegram API: {e}")

    # Check webhook info
    webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo"
    try:
        w_res = requests.get(webhook_url, timeout=10).json()
        print(f"  Webhook Info: {w_res.get('result', {})}")
    except Exception as e:
        print(f"  Webhook Check Error: {e}")

def test_composio_connection():
    print("\n--- 2. Testing Composio API Connection & Apps ---")
    os.environ["COMPOSIO_API_KEY"] = COMPOSIO_API_KEY
    client = Composio(api_key=COMPOSIO_API_KEY)

    try:
        # Check connected accounts in Composio
        connected_accounts = client.connected_accounts.get()
        print(f"[SUCCESS] Composio API Connected!")
        print(f"  Total Connected Accounts: {len(connected_accounts)}")
        for acc in connected_accounts:
            acc_dict = acc.model_dump() if hasattr(acc, "model_dump") else str(acc)
            print(f"  - App: {acc_dict.get('appName') or acc_dict.get('appUniqueId') if isinstance(acc_dict, dict) else acc_dict}")
            if isinstance(acc_dict, dict):
                print(f"    ID: {acc_dict.get('id')}, Status: {acc_dict.get('status')}")
    except Exception as e:
        print(f"[ERROR] Error listing Composio connected accounts: {e}")

    # Check available Telegram actions/tools in Composio
    try:
        print("\n--- 3. Checking Available Composio Actions for Telegram ---")
        actions = client.actions.get(apps=["telegram"])
        print(f"  Found {len(actions)} Telegram actions in Composio:")
        for act in actions[:10]:
            name = getattr(act, "name", str(act))
            doc = getattr(act, "description", "")
            print(f"  - {name}: {doc[:80] if doc else ''}")
    except Exception as e:
        print(f"  Error fetching Composio telegram actions: {e}")

if __name__ == "__main__":
    test_telegram_bot_api()
    test_composio_connection()
