"""
Helper functions package for ChatGPT Telegram Bot
Provides modular components for the refactored openai_utils
"""

# Core configuration and utilities
try:
    from .config_manager import ConfigManager, get_config_manager
    # Don't create the global instance here to avoid import errors
    # It will be created when first accessed
except ImportError as e:
    print(f"Warning: Could not import ConfigManager: {e}")
    ConfigManager = None
    get_config_manager = None

try:
    from .token_utils import TokenCounter
except ImportError as e:
    print(f"Warning: Could not import TokenCounter: {e}")
    TokenCounter = None

try:
    from .message_utils import MessageFormatter
except ImportError as e:
    print(f"Warning: Could not import MessageFormatter: {e}")
    MessageFormatter = None

try:
    from .model_providers import ModelProviders
except ImportError as e:
    print(f"Warning: Could not import ModelProviders: {e}")
    ModelProviders = None

try:
    from .search_utils import SearchUtils
except ImportError as e:
    print(f"Warning: Could not import SearchUtils: {e}")
    SearchUtils = None

try:
    from .index_utils import IndexUtils
except ImportError as e:
    print(f"Warning: Could not import IndexUtils: {e}")
    IndexUtils = None

try:
    from .file_utils import FileUtils
except ImportError as e:
    print(f"Warning: Could not import FileUtils: {e}")
    FileUtils = None

try:
    from .chatgpt_refactored import ChatGPT
except ImportError as e:
    print(f"Warning: Could not import ChatGPT: {e}")
    ChatGPT = None

# Audio functions
try:
    from .audio_utils import transcribe_audio, translate_text, text_to_speech, local_text_to_speech
except ImportError as e:
    print(f"Warning: Could not import audio utilities: {e}")
    transcribe_audio = None
    translate_text = None  
    text_to_speech = None
    local_text_to_speech = None

# Image functions
try:
    from .image_utils import generate_images, is_content_acceptable
except ImportError as e:
    print(f"Warning: Could not import image utilities: {e}")
    generate_images = None
    is_content_acceptable = None

# Export all available components
__all__ = [
    # Classes
    'ConfigManager', 'config_manager',
    'TokenCounter', 'MessageFormatter', 'ModelProviders', 
    'SearchUtils', 'IndexUtils', 'FileUtils', 'ChatGPT',
    # Functions
    'transcribe_audio', 'translate_text', 'text_to_speech', 'local_text_to_speech',
    'generate_images', 'is_content_acceptable'
]
