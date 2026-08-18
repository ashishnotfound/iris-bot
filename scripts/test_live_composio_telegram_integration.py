import os
import json
from lib.composio_client import ComposioClient
from lib.telegram_client import TelegramClient
from lib.hermes_runner import execute_agent_turn

os.environ["COMPOSIO_API_KEY"] = "ak__AA-tGzwmasbekb-rhC-"
os.environ["TELEGRAM_BOT_TOKEN"] = "8916712872:AAGPR875g-RrxX-1iwKsLORjS0p2Oifg5jE"
os.environ["TELEGRAM_ALLOWED_USERS"] = "987654321,123456789"

print("==================================================")
print("  Iris (Hermes Agent) - Live Integration Test")
print("==================================================")

# 1. Telegram Connection Test
tg = TelegramClient()
bot_info = tg.get_me()
print(f"1. Telegram Bot Status: {'OK' if bot_info.get('ok') else 'FAIL'}")
if bot_info.get('ok'):
    bot_res = bot_info.get('result', {})
    print(f"   Name: {bot_res.get('first_name')} (@{bot_res.get('username')})")
    print(f"   ID:   {bot_res.get('id')}")

# 2. Composio Connection Test
cmp = ComposioClient()
accounts = cmp.get_connected_accounts()
print(f"\n2. Composio Status: Configured ({len(accounts)} Connected Apps)")
for acc in accounts:
    print(f"   - App: {acc.get('toolkit', {}).get('slug')} | Account ID: {acc.get('id')}")

# 3. Hermes Agent Turn Test (/status command simulation)
print("\n3. Testing Agent Turn Handling:")
res = execute_agent_turn(chat_id=987654321, user_message="/status", telegram_client=None)
print(f"   Turn Result Status: {res.get('status')}")
clean_reply = res.get('response', '').encode('ascii', 'replace').decode('ascii')
print(f"   Agent Reply:\n{clean_reply}")
print("==================================================")
