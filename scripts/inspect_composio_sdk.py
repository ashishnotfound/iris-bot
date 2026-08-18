import os
import inspect
from composio import Composio, ComposioToolSet, Action, App

COMPOSIO_API_KEY = "ak__AA-tGzwmasbekb-rhC-"
os.environ["COMPOSIO_API_KEY"] = COMPOSIO_API_KEY

print("=== Inspecting Composio SDK ===")
print(f"Composio version: {getattr(Composio, '__version__', 'unknown')}")

toolset = ComposioToolSet(api_key=COMPOSIO_API_KEY)
print(f"ToolSet attributes: {[m for m in dir(toolset) if not m.startswith('_')]}")

# Test execute_action method
if hasattr(toolset, "execute_action"):
    print("\n`execute_action` method signature:")
    print(inspect.signature(toolset.execute_action))

# Test getting tools
try:
    print("\nGetting tools for App.GMAIL...")
    tools = toolset.get_tools(apps=[App.GMAIL])
    print(f"Retrieved {len(tools)} tools for Gmail!")
    print(f"Sample tool type: {type(tools[0])}")
    print(f"Sample tool: {tools[0]}")
except Exception as e:
    print(f"Error get_tools: {e}")
