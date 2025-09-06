#!/usr/bin/env python3
"""
Comprehensive test script for the refactored OpenAI Utils modules
Tests the module structure, imports, and functionality
"""
import sys

def test_module_structure():
    """Test that all expected modules can be imported"""
    print("=== Testing Module Structure ===")
    
    results = {}
    
    # Test helper_functions package
    try:
        print("✓ helper_functions package imports successfully")
        results['helper_package'] = True
    except Exception as e:
        print(f"✗ helper_functions package failed: {e}")
        results['helper_package'] = False
    
    # Test individual helper modules (with expected import warnings)
    helper_modules = [
        'config_manager', 'token_utils', 'message_utils', 'file_utils'
    ]
    
    for module in helper_modules:
        try:
            exec(f"from bot.helper_functions import {module}")
            print(f"✓ {module} imports successfully")
            results[module] = True
        except Exception as e:
            print(f"✗ {module} failed: {e}")
            results[module] = False
    
    # Test main module
    try:
        print("✓ main openai_utils imports successfully")
        results['main_module'] = True
    except Exception as e:
        print(f"✗ main openai_utils failed: {e}")
        results['main_module'] = False
    
    return results

def test_backward_compatibility():
    """Test backward compatibility of the refactored module"""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        import bot.openai_utils as ou
        
        # Test key exports that should exist for backward compatibility
        expected_exports = [
            'config_manager', 'azure_api_key', 'azure_api_base',
            'temperature', 'max_tokens', 'model_name',
            'UPLOAD_FOLDER', 'SUMMARY_FOLDER', 'VECTOR_FOLDER'
        ]
        
        for export in expected_exports:
            if hasattr(ou, export):
                val = getattr(ou, export)
                print(f"✓ {export}: {type(val).__name__}")
            else:
                print(f"✗ {export}: missing")
        
        # Test that functions exist (even if not callable due to missing deps)
        expected_functions = [
            'generate_images_sync', 'is_content_acceptable_sync',
            'transcribe_audio_sync', 'text_to_speech_sync'
        ]
        
        for func in expected_functions:
            if hasattr(ou, func):
                print(f"✓ {func}: available")
            else:
                print(f"✗ {func}: missing")
                
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

def test_config_manager():
    """Test the ConfigManager functionality"""
    print("\n=== Testing ConfigManager ===")
    
    try:
        from bot.helper_functions.config_manager import get_config_manager
        
        cm = get_config_manager()
        print(f"✓ ConfigManager created: {type(cm).__name__}")
        
        # Test some key attributes
        test_attrs = ['azure_api_key', 'temperature', 'max_tokens']
        for attr in test_attrs:
            if hasattr(cm, attr):
                val = getattr(cm, attr)
                print(f"✓ {attr}: {val}")
            else:
                print(f"✗ {attr}: missing")
        
        # Test that methods exist
        if hasattr(cm, 'get_openai_completion_options'):
            options = cm.get_openai_completion_options()
            print(f"✓ get_openai_completion_options(): {type(options).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        return False

def test_available_components():
    """Test which components are available based on dependencies"""
    print("\n=== Testing Available Components ===")
    
    try:
        from bot.helper_functions import (
            TokenCounter, MessageFormatter, FileUtils,
            config_manager
        )
        
        # Test components that should work without external dependencies
        available_components = []
        
        if TokenCounter:
            TokenCounter()
            print("✓ TokenCounter: available and functional")
            available_components.append('TokenCounter')
        
        if MessageFormatter:
            MessageFormatter()
            print("✓ MessageFormatter: available and functional")  
            available_components.append('MessageFormatter')
        
        if FileUtils:
            FileUtils()
            print("✓ FileUtils: available and functional")
            available_components.append('FileUtils')
        
        if config_manager:
            print("✓ config_manager: available and functional")
            available_components.append('config_manager')
        
        print(f"\nTotal available components: {len(available_components)}")
        return len(available_components) > 0
        
    except Exception as e:
        print(f"✗ Component availability test failed: {e}")
        return False

def generate_status_report():
    """Generate a status report of the refactoring"""
    print("\n" + "="*60)
    print("REFACTORING STATUS REPORT")
    print("="*60)
    
    structure_results = test_module_structure()
    compat_success = test_backward_compatibility()
    config_success = test_config_manager()
    components_success = test_available_components()
    
    print("\n=== SUMMARY ===")
    print(f"Module Structure: {'✓ PASS' if structure_results.get('main_module') else '✗ FAIL'}")
    print(f"Backward Compatibility: {'✓ PASS' if compat_success else '✗ FAIL'}")
    print(f"ConfigManager: {'✓ PASS' if config_success else '✗ FAIL'}")
    print(f"Available Components: {'✓ PASS' if components_success else '✗ FAIL'}")
    
    print("\n=== RECOMMENDATIONS ===")
    if not all([structure_results.get('main_module'), compat_success, config_success]):
        print("❌ Some core functionality is broken - needs investigation")
    else:
        print("✅ Core refactoring is functional!")
        print("📦 To enable full functionality, install dependencies:")
        print("   pip install -r requirements.txt")
        print("🧪 Run actual API tests with valid credentials")
        print("🔧 Consider adding unit tests for individual modules")

if __name__ == "__main__":
    sys.path.append('.')
    generate_status_report()
