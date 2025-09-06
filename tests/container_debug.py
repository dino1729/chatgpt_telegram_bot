#!/usr/bin/env python3
import traceback

print("=== Testing imports step by step ===")

# Test 1: config_manager module
print("\n1. Testing config_manager module import...")
try:
    from helper_functions.config_manager import get_config_manager
    print("✓ ConfigManager and get_config_manager imported successfully")
    cm = get_config_manager()
    print("✓ Config manager instance created successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 2: chatgpt_refactored import
print("\n2. Testing chatgpt_refactored import...")
try:
    print("✓ ChatGPT imported successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 3: Full openai_utils import
print("\n3. Testing full openai_utils import...")
try:
    import openai_utils
    print("✓ openai_utils imported successfully")
    print(f"ChatGPT class: {openai_utils.ChatGPT}")
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

print("\n=== Test complete ===")
