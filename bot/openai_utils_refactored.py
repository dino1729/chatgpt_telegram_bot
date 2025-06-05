"""
Refactored OpenAI Utils - Master module that imports and exposes all functionality
from helper function modules for backward compatibility and clean API access.
"""

# Import all the helper modules
from .helper_functions.config_manager import get_config_manager
from .helper_functions.chatgpt_refactored import ChatGPT
from .helper_functions.audio_utils import (
    transcribe_audio, 
    translate_text, 
    text_to_speech, 
    local_text_to_speech
)
from .helper_functions.image_utils import (
    generate_images, 
    is_content_acceptable
)

# Export the main classes and functions for backward compatibility
__all__ = [
    'ChatGPT',
    'transcribe_audio',
    'translate_text', 
    'text_to_speech',
    'local_text_to_speech',
    'generate_images',
    'is_content_acceptable',
    'config_manager'
]

# Create global config manager instance
config_manager = get_config_manager()

# Legacy imports for backward compatibility
cm = config_manager

# Expose configuration settings at module level for backward compatibility
azure_api_key = cm.azure_api_key
azure_api_base = cm.azure_api_base
azure_embeddingapi_version = cm.azure_embeddingapi_version
azure_chatapi_version = cm.azure_chatapi_version
azure_gpt4_deploymentid = cm.azure_gpt4_deploymentid
openai_gpt4_modelname = cm.openai_gpt4_modelname
azure_gpt35_deploymentid = cm.azure_gpt35_deploymentid
openai_gpt35_modelname = cm.openai_gpt35_modelname
azure_embedding_deploymentid = cm.azure_embedding_deploymentid
openai_embedding_modelname = cm.openai_embedding_modelname

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
import asyncio

def generate_images_sync(prompt, n_images=4):
    """Synchronous wrapper for generate_images"""
    return asyncio.run(generate_images(prompt, n_images, config_manager))

def is_content_acceptable_sync(prompt):
    """Synchronous wrapper for is_content_acceptable"""
    return asyncio.run(is_content_acceptable(prompt, config_manager))

def transcribe_audio_sync(audio_file):
    """Synchronous wrapper for transcribe_audio"""
    return asyncio.run(transcribe_audio(audio_file, config_manager))

def translate_text_sync(text, target_language):
    """Synchronous wrapper for translate_text"""
    return asyncio.run(translate_text(text, target_language, config_manager))

def text_to_speech_sync(text, output_path, language):
    """Synchronous wrapper for text_to_speech"""
    return asyncio.run(text_to_speech(text, output_path, language, config_manager))

def local_text_to_speech_sync(text, output_path, model_name):
    """Synchronous wrapper for local_text_to_speech"""
    return asyncio.run(local_text_to_speech(text, output_path, model_name, config_manager))

# Note: This is the new refactored openai_utils.py
# The original file will be backed up and this will replace it
print("Loaded refactored OpenAI Utils with modular helper functions")
