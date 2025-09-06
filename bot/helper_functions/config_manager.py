"""Unified configuration manager using only generic LLM role names (legacy aliases removed)."""

import os
import sys
import logging

_QUIET = os.getenv("QUIET_MODE") or os.getenv("FIRECRAWL_QUIET")
_level = logging.ERROR if _QUIET else (os.getenv("LOG_LEVEL") or "INFO")
try:  # pragma: no cover
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, str(_level).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    if _QUIET:
        for noisy in ["urllib3", "requests", "openai", "llama_index", "httpx"]:
            logging.getLogger(noisy).setLevel(logging.CRITICAL)
except Exception:  # pragma: no cover
    pass

# Try relative then absolute import for config module
try:
    from .. import config  # type: ignore
except Exception:  # pragma: no cover
    try:
        import config  # type: ignore
    except Exception:  # pragma: no cover
        config = None  # type: ignore
        logging.warning("Config module not found; using defaults")

LLAMA_INDEX_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding  # type: ignore
    from llama_index.llms.azure_openai import AzureOpenAI  # type: ignore
    from llama_index.core.node_parser import SentenceSplitter  # type: ignore
    from llama_index.core import PromptHelper, PromptTemplate, Settings  # type: ignore
    LLAMA_INDEX_AVAILABLE = True
except Exception:  # pragma: no cover
    logging.debug("LlamaIndex not installed")

try:  # pragma: no cover
    from openai import AzureOpenAI as OpenAIAzure  # type: ignore
except Exception:  # pragma: no cover
    OpenAIAzure = None  # type: ignore
    logging.debug("OpenAI library (AzureOpenAI) not installed")


class ConfigManager:
    """Load configuration data (generic model roles only)."""

    def __init__(self):
        self._load_api_keys()
        self._load_model_settings()
        self._load_templates()
        self._load_folders()
        self._setup_llama_index()

    def _get(self, name, default=None):
        return getattr(config, name, default) if config else default

    def _load_api_keys(self):
        """Load API keys and model deployment identifiers from config module (if present)."""
        # Core Azure OpenAI settings
        self.azure_api_key = self._get('azure_api_key')
        self.azure_api_base = self._get('azure_api_base')
        self.azure_embeddingapi_version = self._get('azure_embeddingapi_version')
        self.azure_chatapi_version = self._get('azure_chatapi_version')

        # Generic model identifiers (no legacy fallbacks)
        self.smart_llm_azure_deployment = self._get('smart_llm_azure_deployment')
        self.smart_llm_openai_model = self._get('smart_llm_openai_model', 'gpt-5-chat')
        self.thinking_llm_azure_deployment = self._get('thinking_llm_azure_deployment')
        self.thinking_llm_openai_model = self._get('thinking_llm_openai_model', 'gpt-3.5-turbo')
        self.fast_llm_azure_deployment = self._get('fast_llm_azure_deployment')
        self.fast_llm_openai_model = self._get('fast_llm_openai_model')
        self.superfast_llm_azure_deployment = self._get('superfast_llm_azure_deployment')
        self.embedding_azure_deployment = self._get('embedding_azure_deployment')
        self.embedding_openai_model = self._get('embedding_openai_model', 'text-embedding-ada-002')

        # Service/API keys
        self.openai_api_key = self._get('openai_api_key')
        self.openai_api_base = self._get('openai_base_url')
        self.anthropic_api_key = self._get('anthropic_api_key')
        self.google_api_key = self._get('google_api_key')
        self.cohere_api_key = self._get('cohere_api_key')
        self.bing_api_key = self._get('bing_api_key')
        self.bing_endpoint = self._get('bing_endpoint')
        self.bing_news_endpoint = self._get('bing_news_endpoint')
        self.openweather_api_key = self._get('openweather_api_key')
        self.firecrawl_base_url = self._get('firecrawl_base_url')
        self.firecrawl_api_key = self._get('firecrawl_api_key')
        self.azurespeechkey = self._get('azurespeechkey')
        self.azurespeechregion = self._get('azurespeechregion')
        self.azuretexttranslatorkey = self._get('azuretexttranslatorkey')
        self.rvctts_api_base = self._get('rvctts_api_base')
        self.ollama_api_key = self._get('ollama_api_key')
        self.ollama_api_base = self._get('ollama_api_base')
        self.gemini_model_name = self._get('gemini_model_name')
        self.groq_api_key = self._get('groq_api_key')

    def _load_model_settings(self):
        self.temperature = self._get('temperature', 0.7)
        self.max_tokens = self._get('max_tokens', 1000)
        self.model_name = self._get('model_name', 'gpt-3.5-turbo')
        self.num_output = self._get('num_output', 1000)
        self.max_chunk_overlap_ratio = self._get('max_chunk_overlap_ratio', 0.1)
        self.max_input_size = self._get('max_input_size', 4000)
        self.context_window = self._get('context_window', 4000)
        self.keywords = self._get('keywords', [])
        self.lite_mode = self._get('lite_mode', False)

    def _load_templates(self):
        self.sum_template = self._get('sum_template', '')
        self.eg_template = self._get('eg_template', '')
        self.ques_template = self._get('ques_template', '')
        self.example_queries = self._get('example_queries', [])
        if LLAMA_INDEX_AVAILABLE:
            self.summary_template = PromptTemplate(self.sum_template)
            self.example_template = PromptTemplate(self.eg_template)
            self.qa_template = PromptTemplate(self.ques_template)
        else:
            self.summary_template = None
            self.example_template = None
            self.qa_template = None

    def _load_folders(self):
        self.UPLOAD_FOLDER = self._get('UPLOAD_FOLDER', 'data/uploads')
        self.SUMMARY_FOLDER = self._get('SUMMARY_FOLDER', 'data/summary_index')
        self.VECTOR_FOLDER = self._get('VECTOR_FOLDER', 'data/vector_index')
        for f in [self.UPLOAD_FOLDER, self.SUMMARY_FOLDER, self.VECTOR_FOLDER]:
            os.makedirs(f, exist_ok=True)

    def _setup_llama_index(self):  # pragma: no cover
        if not LLAMA_INDEX_AVAILABLE:
            return
        needed = [
            'azure_api_key', 'azure_api_base', 'azure_chatapi_version',
            'smart_llm_azure_deployment', 'embedding_azure_deployment',
            'azure_embeddingapi_version'
        ]
        if any(getattr(self, n, None) is None for n in needed):
            logging.debug("Skipping LlamaIndex initialization due to missing config")
            return
        try:
            if OpenAIAzure is None:
                logging.warning("AzureOpenAI class not available; LlamaIndex disabled")
                return
            Settings.client = OpenAIAzure(
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_api_base,
                api_version=self.azure_chatapi_version,
            )
            Settings.llm = AzureOpenAI(
                azure_deployment=self.smart_llm_azure_deployment,
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_api_base,
                api_version=self.azure_chatapi_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            Settings.embed_model = AzureOpenAIEmbedding(
                azure_deployment=self.embedding_azure_deployment,
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_api_base,
                api_version=self.azure_embeddingapi_version,
            )
            Settings.text_splitter = SentenceSplitter()
            Settings.prompt_helper = PromptHelper(
                self.max_input_size,
                self.num_output,
                self.max_chunk_overlap_ratio,
            )
        except Exception as e:  # pragma: no cover
            logging.error(f"Failed LlamaIndex setup: {e}")

    def get_openai_completion_options(self):
        return {"temperature": self.temperature, "max_completion_tokens": self.max_tokens}


_singleton = None


def get_config_manager():
    global _singleton
    if _singleton is None:
        _singleton = ConfigManager()
    return _singleton

