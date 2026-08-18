import os
import json
from lib.image_gen_client import ImageGenClient
from lib.hermes_runner import execute_agent_turn

os.environ["COMPOSIO_API_KEY"] = "ak__AA-tGzwmasbekb-rhC-"
os.environ["TELEGRAM_BOT_TOKEN"] = "8916712872:AAGPR875g-RrxX-1iwKsLORjS0p2Oifg5jE"
os.environ["TELEGRAM_ALLOWED_USERS"] = "987654321,123456789"

print("==================================================")
print("  Iris (Hermes Agent) - Image Gen & OpenRouter Test")
print("==================================================")

# 1. Direct Image Generation Test (Pollinations AI - No OpenAI Key)
prompt = "A futuristic glowing crystal dragon hovering over a starry mountain lake, concept art"
img_client = ImageGenClient()
result = img_client.generate_image(prompt)

print(f"1. Pollinations Free Image Generation Result:")
print(f"   Success: {result.get('success')}")
print(f"   Provider: {result.get('provider')}")
print(f"   Image URL: {result.get('image_url')}")

# 2. Agent Turn Image Command Simulation
print("\n2. Testing /image Command via Agent Turn:")
turn_res = execute_agent_turn(chat_id=987654321, user_message="/image A majestic golden eagle soaring over snow covered mountains", telegram_client=None)
print(f"   Turn Status: {turn_res.get('status')}")
print(f"   Type: {turn_res.get('type')}")
print(f"   Generated Image URL: {turn_res.get('image_url')}")
print("==================================================")
