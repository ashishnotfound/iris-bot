import os

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"
os.environ["COMPOSIO_API_KEY"] = COMPOSIO_API_KEY

print("=== Testing COMPOSIO_BASE_URL Overrides ===")

urls_to_test = [
    "https://backend.composio.dev/api/v1",
    "https://backend.composio.dev/api/v2",
    "https://backend.composio.dev/api/v3",
    "https://backend.composio.dev/v1",
    "https://backend.composio.dev/v3"
]

for url in urls_to_test:
    os.environ["COMPOSIO_BASE_URL"] = url
    try:
        from composio import ComposioToolSet
        ts = ComposioToolSet(api_key=COMPOSIO_API_KEY, base_url=url)
        print(f"[SUCCESS] Connected with base_url: {url}")
        break
    except Exception as e:
        print(f"[FAIL] base_url: {url} -> {e}")
