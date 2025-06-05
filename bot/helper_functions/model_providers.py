"""
AI model provider implementations for different services
"""
import logging

# Azure OpenAI for text and image per models.yml
try:
    from openai import AzureOpenAI as OpenAIAzure
except ImportError:
    OpenAIAzure = None
    logging.warning("Azure OpenAI library not available")

class ModelProviders:
    """Providers for models: Azure OpenAI (gpt-4o, o4-mini, model-router) and Google Gemini"""
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
        client = self.get_azure_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **completion_options
        )
        return response.choices[0].message.content

    async def send_azure_openai_stream(self, model, messages, completion_options):
        client = self.get_azure_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **completion_options
        )
        return response

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
            model_name="gemini-2.5-flash-preview-05-20",
            generation_config=generation_config
        )

    async def send_gemini_message(self, messages):
        gemini = self.configure_gemini()
        response = await gemini.generate_content_async(str(messages).replace("'", '"'))
        return response.text
