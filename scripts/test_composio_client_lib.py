import os
import json
from lib.composio_client import ComposioClient

os.environ["COMPOSIO_API_KEY"] = "ak__AA-tGzwmasbekb-rhC-"

client = ComposioClient()

print("=== 1. Checking Configuration ===")
print(f"Is Configured: {client.is_configured()}")

print("\n=== 2. Connected Accounts ===")
accounts = client.get_connected_accounts()
print(f"Found {len(accounts)} active connected accounts:")
for acc in accounts:
    print(f"  - App: {acc.get('toolkit', {}).get('slug')} | ID: {acc.get('id')} | User: {acc.get('user_id')}")

print("\n=== 3. Searching Gmail Tools ===")
gmail_tools = client.get_tools_for_query("gmail", limit=3)
schemas = [client.convert_tool_to_schema(t) for t in gmail_tools]
print(f"Converted OpenAI Schemas Sample:")
print(json.dumps(schemas, indent=2))

print("\n=== 4. Live Tool Execution via ComposioClient ===")
if accounts:
    acc = accounts[0] # googlecalendar or gmail
    toolkit_slug = acc.get('toolkit', {}).get('slug')
    acc_id = acc.get('id')
    user_id = acc.get('user_id')

    if toolkit_slug == 'gmail':
        res = client.execute_tool("GMAIL_FETCH_EMAILS", {"max_results": 1}, connected_account_id=acc_id, user_id=user_id)
        print(f"Gmail Execution Result Successful: {res.get('successful', False)}")
        print(f"Data snippet: {str(res.get('data'))[:300]}")
    elif toolkit_slug == 'googlecalendar':
        res = client.execute_tool("GOOGLECALENDAR_EVENTS_LIST", {"calendarId": "primary", "maxResults": 1}, connected_account_id=acc_id, user_id=user_id)
        print(f"Google Calendar Execution Result Successful: {res.get('successful', False)}")
        print(f"Summary: {res.get('data', {}).get('summary')}")
