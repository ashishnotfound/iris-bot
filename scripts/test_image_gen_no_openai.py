import os
import urllib.parse
import requests

def test_pollinations_image_gen(prompt="A futuristic neon cybernetic owl perched on a branch, digital art"):
    print("=== 1. Testing Pollinations.ai (100% Free, No API Key Required) ===")
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    print(f"Generated Image URL: {image_url}")
    try:
        r = requests.get(image_url, timeout=15)
        print(f"Status Code: {r.status_code}")
        print(f"Content Type: {r.headers.get('Content-Type')}")
        print(f"Content Size: {len(r.content)} bytes")
        if r.status_code == 200 and "image" in r.headers.get('Content-Type', ''):
            print("[SUCCESS] Pollinations.ai image generation is 100% working!")
            return image_url
    except Exception as e:
        print(f"Pollinations error: {e}")
    return None

def test_openrouter_models():
    print("\n=== 2. Testing OpenRouter Models API ===")
    url = "https://openrouter.ai/api/v1/models"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            models = r.json().get("data", [])
            print(f"[SUCCESS] Fetched {len(models)} OpenRouter models!")
            free_models = [m.get("id") for m in models if ":free" in m.get("id", "")]
            print(f"Sample Free OpenRouter Models ({len(free_models)} available):")
            for fm in free_models[:10]:
                print(f" - {fm}")
    except Exception as e:
        print(f"OpenRouter test error: {e}")

if __name__ == "__main__":
    test_pollinations_image_gen()
    test_openrouter_models()
