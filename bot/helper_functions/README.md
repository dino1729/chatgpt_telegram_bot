# Helper Functions Documentation

This directory contains modular helper functions that have been refactored from the original monolithic `openai_utils.py` file. The code is now organized into logical modules for better maintainability, testing, and development.

## Module Structure

### 1. `config_manager.py`
- **Purpose**: Manages all configuration settings and initialization
- **Key Classes**: `ConfigManager`
- **Responsibilities**:
  - Load API keys and endpoints
  - Initialize LlamaIndex settings
  - Manage model parameters
  - Handle folder structure setup

### 2. `token_utils.py`
- **Purpose**: Token counting utilities for different language models
- **Key Classes**: `TokenCounter`
- **Responsibilities**:
  - Count tokens for chat completion models
  - Count tokens for completion models
  - Handle different model-specific token calculation rules

### 3. `message_utils.py`
- **Purpose**: Message formatting and prompt generation
- **Key Classes**: `MessageFormatter`
- **Responsibilities**:
  - Generate structured messages for API calls
  - Format vision messages with image support
  - Handle prompt templates and formatting
  - Post-process AI responses

### 4. `model_providers.py`
- **Purpose**: AI model provider implementations
- **Key Classes**: `ModelProviders`
- **Responsibilities**:
  - Handle Azure OpenAI integration
  - Manage Cohere, Groq, Google AI clients
  - Provide unified interface for different AI services
  - Support both streaming and non-streaming responses

### 5. `search_utils.py`
- **Purpose**: Web search and content extraction
- **Key Classes**: `SearchUtils`
- **Dependencies**: `file_utils.py`, `index_utils.py`
- **Responsibilities**:
  - Bing search integration
  - Web content extraction and scraping
  - Weather data retrieval
  - News search functionality

### 6. `index_utils.py`
- **Purpose**: LlamaIndex utilities for document processing
- **Key Classes**: `IndexUtils`
- **Responsibilities**:
  - Document indexing and querying
  - Summary generation
  - Vector search functionality
  - Query engine configuration

### 7. `audio_utils.py`
- **Purpose**: Audio processing and speech services
- **Key Functions**: Async functions for audio processing
- **Responsibilities**:
  - Speech-to-text transcription
  - Text-to-speech synthesis
  - Language translation
  - Local TTS integration

### 8. `image_utils.py`
- **Purpose**: Image generation and processing
- **Key Functions**: Image-related utilities
- **Responsibilities**:
  - DALL-E image generation
  - Content moderation
  - Image processing utilities

### 9. `file_utils.py`
- **Purpose**: File operations and management
- **Key Classes**: `FileUtils`
- **Responsibilities**:
  - File cleanup operations
  - Text file saving
  - Directory management

### 10. `chatgpt_refactored.py`
- **Purpose**: Main ChatGPT class using modular components
- **Key Classes**: `ChatGPT`
- **Responsibilities**:
  - Main chat interface
  - Coordinate between different modules
  - Maintain backward compatibility
  - Handle streaming and non-streaming responses

## Usage

### Direct Import
```python
from helper_functions.chatgpt_refactored import ChatGPT
from helper_functions.config_manager import config_manager
from helper_functions.audio_utils import transcribe_audio

# Use the refactored components
chatgpt = ChatGPT("gpt-4o")
response = await chatgpt.send_message("Hello!")
```

### Through Master Module
```python
# The main openai_utils.py imports everything for backward compatibility
from openai_utils import ChatGPT, transcribe_audio, generate_images

# Use exactly as before - no code changes needed
chatgpt = ChatGPT("gpt-4o")
response = await chatgpt.send_message("Hello!")
```

## Benefits of Refactoring

1. **Modularity**: Each module has a single responsibility
2. **Maintainability**: Easier to find and fix issues
3. **Testability**: Individual modules can be tested in isolation
4. **Reusability**: Helper functions can be reused across projects
5. **Backward Compatibility**: Existing code continues to work unchanged
6. **Performance**: Only load what you need
7. **Documentation**: Each module is self-documenting

## Migration Guide

### For Existing Code
No changes required! The main `openai_utils.py` file now imports everything from the helper modules, maintaining full backward compatibility.

### For New Development
Consider importing specific modules directly for better performance and clearer dependencies:

```python
# Instead of importing everything
from openai_utils import *

# Import only what you need
from helper_functions.chatgpt_refactored import ChatGPT
from helper_functions.search_utils import SearchUtils
```

## Testing

Each module can be tested independently:

```python
# Test configuration
from helper_functions.config_manager import config_manager
assert config_manager.azure_api_key is not None

# Test token counting
from helper_functions.token_utils import TokenCounter
counter = TokenCounter()
tokens = counter.count_tokens_from_messages(messages, response)

# Test search functionality
from helper_functions.search_utils import SearchUtils
search = SearchUtils(config_manager)
results = search.get_bing_results("test query")
```

## Future Enhancements

1. **Async Optimization**: Further optimize async/await patterns
2. **Caching**: Add intelligent caching for API responses
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Metrics**: Add performance and usage metrics
5. **Error Handling**: Enhanced error handling and retry logic
6. **Configuration Validation**: Validate configuration on startup

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Configuration Missing**: Check that config.py has all required settings
3. **API Key Issues**: Verify all API keys are properly configured
4. **Path Issues**: Ensure folder structure exists and is writable

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new functionality:

1. Choose the appropriate module or create a new one
2. Follow the existing patterns and naming conventions
3. Add proper documentation and type hints
4. Update this README if adding new modules
5. Ensure backward compatibility is maintained
