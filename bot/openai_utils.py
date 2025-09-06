"""Refactored OpenAI Utils master module (backward compatibility layer)."""

import asyncio  # Needed for sync wrappers (kept at top to satisfy style)

# Import all the helper modules with defensive importing
# Try relative imports first (when used as package), fall back to absolute imports
try:
    from .helper_functions.config_manager import get_config_manager
except ImportError:
    from helper_functions.config_manager import get_config_manager

try:
    try:
        from .helper_functions.chatgpt_refactored import ChatGPT
    except ImportError:
        from helper_functions.chatgpt_refactored import ChatGPT
except ImportError as e:
    print(f"Warning: ChatGPT class not available: {e}")
    ChatGPT = None

try:
    try:
        from .helper_functions.audio_utils import (
            transcribe_audio, 
            translate_text, 
            text_to_speech, 
            local_text_to_speech
        )
    except ImportError:
        from helper_functions.audio_utils import (
            transcribe_audio, 
            translate_text, 
            text_to_speech, 
            local_text_to_speech
        )
except ImportError as e:
    print(f"Warning: Audio utilities not available: {e}")
    transcribe_audio = None
    translate_text = None
    text_to_speech = None
    local_text_to_speech = None

try:
    try:
        from .helper_functions.image_utils import (
            generate_images, 
            generate_images_gpt_image_1,
            edit_image_gpt_image_1,
            is_content_acceptable
        )
    except ImportError:
        from helper_functions.image_utils import (
            generate_images, 
            generate_images_gpt_image_1,
            edit_image_gpt_image_1,
            is_content_acceptable
        )
except ImportError as e:
    print(f"Warning: Image utilities not available: {e}")
    generate_images = None
    generate_images_gpt_image_1 = None
    edit_image_gpt_image_1 = None
    is_content_acceptable = None

# Create global config manager instance
config_manager = get_config_manager()

# Export the main classes and functions for backward compatibility
__all__ = [
    'ChatGPT',
    'transcribe_audio',
    'translate_text', 
    'text_to_speech',
    'local_text_to_speech',
    'generate_images',
    'generate_images_gpt_image_1',
    'edit_image_gpt_image_1',
    'is_content_acceptable',
    'config_manager'
]

# Legacy imports for backward compatibility
cm = config_manager

smart_llm_azure_deployment = cm.smart_llm_azure_deployment
smart_llm_openai_model = cm.smart_llm_openai_model
thinking_llm_azure_deployment = cm.thinking_llm_azure_deployment
thinking_llm_openai_model = cm.thinking_llm_openai_model
fast_llm_azure_deployment = cm.fast_llm_azure_deployment
fast_llm_openai_model = cm.fast_llm_openai_model
superfast_llm_azure_deployment = cm.superfast_llm_azure_deployment
embedding_azure_deployment = cm.embedding_azure_deployment
embedding_openai_model = cm.embedding_openai_model

cohere_api_key = cm.cohere_api_key
google_api_key = cm.google_api_key
gemini_model_name = cm.gemini_model_name
groq_api_key = cm.groq_api_key

bing_api_key = cm.bing_api_key
bing_endpoint = cm.bing_endpoint
bing_news_endpoint = cm.bing_news_endpoint
openweather_api_key = cm.openweather_api_key

azurespeechkey = cm.azurespeechkey
azurespeechregion = cm.azurespeechregion
azuretexttranslatorkey = cm.azuretexttranslatorkey

rvctts_api_base = cm.rvctts_api_base
ollama_api_key = cm.ollama_api_key
ollama_api_base = cm.ollama_api_base

sum_template = cm.sum_template
eg_template = cm.eg_template
ques_template = cm.ques_template

temperature = cm.temperature
max_tokens = cm.max_tokens
model_name = cm.model_name
num_output = cm.num_output
max_chunk_overlap_ratio = cm.max_chunk_overlap_ratio
max_input_size = cm.max_input_size
context_window = cm.context_window
keywords = cm.keywords

lite_mode = cm.lite_mode

example_qs = []
summary = "No Summary available yet"
example_queries = cm.example_queries
summary_template = cm.summary_template
example_template = cm.example_template
qa_template = cm.qa_template

UPLOAD_FOLDER = cm.UPLOAD_FOLDER
SUMMARY_FOLDER = cm.SUMMARY_FOLDER
VECTOR_FOLDER = cm.VECTOR_FOLDER

OPENAI_COMPLETION_OPTIONS = cm.get_openai_completion_options()

# Wrapper functions for async functions to maintain compatibility

def generate_images_sync(prompt, n_images=4):
    """Synchronous wrapper for generate_images"""
    if generate_images is None:
        raise RuntimeError("Image generation not available - missing dependencies")
    return asyncio.run(generate_images(prompt, n_images, config_manager))

def is_content_acceptable_sync(prompt):
    """Synchronous wrapper for is_content_acceptable"""
    if is_content_acceptable is None:
        raise RuntimeError("Content moderation not available - missing dependencies")
    return asyncio.run(is_content_acceptable(prompt, config_manager))

def transcribe_audio_sync(audio_file):
    """Synchronous wrapper for transcribe_audio"""
    if transcribe_audio is None:
        raise RuntimeError("Audio transcription not available - missing dependencies")
    return asyncio.run(transcribe_audio(audio_file, config_manager))

def translate_text_sync(text, target_language):
    """Synchronous wrapper for translate_text"""
    if translate_text is None:
        raise RuntimeError("Text translation not available - missing dependencies")
    return asyncio.run(translate_text(text, target_language, config_manager))

def text_to_speech_sync(text, output_path, language):
    """Synchronous wrapper for text_to_speech"""
    if text_to_speech is None:
        raise RuntimeError("Text-to-speech not available - missing dependencies")
    return asyncio.run(text_to_speech(text, output_path, language, config_manager))

def local_text_to_speech_sync(text, output_path, model_name):
    """Synchronous wrapper for local_text_to_speech"""
    if local_text_to_speech is None:
        raise RuntimeError("Local text-to-speech not available - missing dependencies")
    return asyncio.run(local_text_to_speech(text, output_path, model_name, config_manager))

# Note: This is the new refactored openai_utils.py
# The original file has been backed up as openai_utils_backup.py
print("Loaded refactored OpenAI Utils with modular helper functions")
