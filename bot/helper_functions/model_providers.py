"""
AI model provider implementations for different services
"""
import logging
import asyncio

# Azure OpenAI for text and image per models.yml
try:
    from openai import AzureOpenAI as OpenAIAzure
except ImportError:
    OpenAIAzure = None
    logging.warning("Azure OpenAI library not available")

class ModelProviders:
    """Providers for models: Azure OpenAI (gpt-5-chat, gpt-5, model-router) and Google Gemini (gemini-2.5-flash)."""
    def __init__(self, config_manager):
        self.config = config_manager

    def get_azure_openai_client(self):
        if not OpenAIAzure:
            raise ImportError("Azure OpenAI library not available")
        return OpenAIAzure(
            api_key=self.config.azure_api_key,
            azure_endpoint=self.config.azure_api_base,
            api_version=self.config.azure_chatapi_version,
        )

    async def send_azure_openai_message(self, model, messages, completion_options):
        # Retry logic with exponential backoff
        max_retries = getattr(self.config, 'max_api_retries', 1)
        backoff = getattr(self.config, 'api_retry_backoff', 1.0)
        for attempt in range(1, max_retries + 1):
            try:
                client = self.get_azure_openai_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **completion_options
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt >= max_retries:
                    logging.error(f"Azure OpenAI message failed after {attempt} attempts: {e}")
                    raise
                wait = backoff * (2 ** (attempt - 1))
                logging.warning(f"Azure OpenAI message error, retry {attempt}/{max_retries} in {wait:.1f}s: {e}")
                await asyncio.sleep(wait)

    async def send_azure_openai_stream(self, model, messages, completion_options):
        # Retry creation of streaming generator
        max_retries = getattr(self.config, 'max_api_retries', 1)
        backoff = getattr(self.config, 'api_retry_backoff', 1.0)
        for attempt in range(1, max_retries + 1):
            try:
                client = self.get_azure_openai_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **completion_options
                )
                return response
            except Exception as e:
                if attempt >= max_retries:
                    logging.error(f"Azure OpenAI stream failed after {attempt} attempts: {e}")
                    raise
                wait = backoff * (2 ** (attempt - 1))
                logging.warning(f"Azure OpenAI stream error, retry {attempt}/{max_retries} in {wait:.1f}s: {e}")
                await asyncio.sleep(wait)

    def configure_gemini(self):
        """Dynamically import and configure Google Gemini model using importlib"""
        import importlib
        try:
            genai = importlib.import_module('google.generativeai')
        except ImportError:
            raise ImportError("Google Generative AI library not available")
        genai.configure(api_key=self.config.google_api_key)
        generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": 0.9,
            "top_k": 1,
        }
        return genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config
        )

    async def send_gemini_message(self, messages):
        # Retry logic for Gemini
        max_retries = getattr(self.config, 'max_api_retries', 1)
        backoff = getattr(self.config, 'api_retry_backoff', 1.0)
        for attempt in range(1, max_retries + 1):
            try:
                gemini = self.configure_gemini()
                response = await gemini.generate_content_async(str(messages).replace("'", '"'))
                return response.text
            except Exception as e:
                if attempt >= max_retries:
                    logging.error(f"Gemini message failed after {attempt} attempts: {e}")
                    raise
                wait = backoff * (2 ** (attempt - 1))
                logging.warning(f"Gemini message error, retry {attempt}/{max_retries} in {wait:.1f}s: {e}")
                await asyncio.sleep(wait)
