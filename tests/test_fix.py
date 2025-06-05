#!/usr/bin/env python3

"""
Test script to verify that our import fix works correctly
"""

import sys
import os

# Add the bot directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot'))

def test_imports():
    print("Testing imports...")
    
    try:
        # Test config manager import
        from helper_functions.config_manager import get_config_manager
        print("✅ Successfully imported get_config_manager")
        
        # Test if config manager can be instantiated
        config_manager = get_config_manager()
        print("✅ Successfully created config_manager instance")
        
        # Test ChatGPT import
        from helper_functions.chatgpt_refactored import ChatGPT
        print("✅ Successfully imported ChatGPT class")
        
        # Test if ChatGPT can be instantiated (should not be None now)
        chatgpt_instance = ChatGPT(model="gpt-3.5-turbo")
        print("✅ Successfully created ChatGPT instance")
        
        if chatgpt_instance is not None:
            print("✅ ChatGPT instance is not None - import fix successful!")
        else:
            print("❌ ChatGPT instance is still None - import fix failed")
            
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing the import fix for ChatGPT Telegram Bot")
    print("=" * 50)
    
    success = test_imports()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed! The import fix appears to be working.")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    print("=" * 50)
