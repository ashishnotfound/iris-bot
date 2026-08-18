import os
import json
from lib.image_gen_client import ImageGenClient

print("=== Testing Fast Image Generation URL Builder ===")

client = ImageGenClient()
prompt = "A futuristic cybernetic owl floating over a neon city at night, concept art"
res = client.generate_image(prompt)

print(f"Success: {res.get('success')}")
print(f"Provider: {res.get('provider')}")
print(f"Generated Image URL:\n{res.get('image_url')}")
