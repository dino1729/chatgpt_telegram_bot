#!/usr/bin/env python3
"""
Test script to validate the refactored module imports and basic functionality
"""
import sys
import os

def test_basic_imports():
    """Test basic imports without external dependencies"""
    print("Testing basic module imports...")
    
    try:
        # Test relative imports in helper functions
        sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))
        
        # Test individual helper modules with minimal dependencies
        print("✓ Testing helper_functions package...")
        import importlib
        hf_module = importlib.import_module("bot.helper_functions")
        print("✓ helper_functions module loaded")
        required_symbols = [
            "ConfigManager", "TokenCounter", "MessageFormatter",
            "ModelProviders", "SearchUtils", "FileUtils"
        ]
        missing = [s for s in required_symbols if not hasattr(hf_module, s)]
        if missing:
            raise AssertionError(f"Missing symbols: {missing}")
        print("✓ All expected symbols present in helper_functions")
        
        print("✓ All basic imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_class_instantiation():
    """Test basic class instantiation without external API calls"""
    print("\nTesting class instantiation...")
    
    try:
        from bot.helper_functions import TokenCounter, MessageFormatter, FileUtils
        
        # Test token counter
        TokenCounter()  # noqa: F841
        print("✓ TokenCounter instantiated")
        
        # Test message formatter  
        MessageFormatter()  # noqa: F841
        print("✓ MessageFormatter instantiated")
        
        # Test file utils
        FileUtils()  # noqa: F841
        print("✓ FileUtils instantiated")

        return True
    except Exception as e:  # noqa: BLE001
        print(f"✗ Class instantiation error: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility of the master module"""
    print("\nTesting backward compatibility...")
    
    try:
        # This should work if our refactoring maintains compatibility
        import bot.openai_utils
        
        # Check if main exports are available
        required_exports = [
            'ChatGPT', 'transcribe_audio', 'generate_images',
            'text_to_speech', 'is_content_acceptable'
        ]
        
        for export in required_exports:
            if hasattr(bot.openai_utils, export):
                print(f"✓ {export} is available")
            else:
                print(f"✗ {export} is missing")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility error: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Refactored OpenAI Utils ===\n")
    
    success = True
    success &= test_basic_imports()
    success &= test_class_instantiation() 
    success &= test_backward_compatibility()
    
    print("\n=== Test Results ===")
    if success:
        print("🎉 All tests passed! Refactoring appears successful.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        
    sys.exit(0 if success else 1)
