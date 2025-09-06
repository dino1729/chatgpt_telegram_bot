#!/usr/bin/env python3
"""
Debug script to test imports and identify the exact issue
"""
import sys
import os

# Add the bot directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot'))

print("=== DEBUGGING IMPORTS ===")

# Test 1: Basic config import
print("\n1. Testing config import...")
try:
    print("✓ config imported successfully")
except Exception as e:
    print(f"✗ config import failed: {e}")

# Test 2: Config manager import  
print("\n2. Testing config_manager import...")
try:
    from helper_functions.config_manager import get_config_manager
    print("✓ config_manager imported successfully")
    
    cm = get_config_manager()
    print("✓ config_manager instance created successfully")
except Exception as e:
    print(f"✗ config_manager import/creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: ChatGPT refactored import
print("\n3. Testing ChatGPT import...")
try:
    from helper_functions.chatgpt_refactored import ChatGPT
    print("✓ ChatGPT imported successfully")
    
    # Try to create an instance
    chat = ChatGPT(model="gpt-5-chat")
    print("✓ ChatGPT instance created successfully")
except Exception as e:
    print(f"✗ ChatGPT import/creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: openai_utils import
print("\n4. Testing openai_utils import...")
try:
    import openai_utils
    print("✓ openai_utils imported successfully")
    print(f"  ChatGPT available: {openai_utils.ChatGPT is not None}")
    
    if openai_utils.ChatGPT is not None:
        chat_instance = openai_utils.ChatGPT(model="gpt-5-chat")
        print("✓ ChatGPT instance created via openai_utils")
    else:
        print("✗ ChatGPT is None in openai_utils")
        
except Exception as e:
    print(f"✗ openai_utils import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")
