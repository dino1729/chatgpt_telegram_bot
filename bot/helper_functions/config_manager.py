"""
Configuration and Settings Module
Handles all configuration loading and settings initialization
"""
import os
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Conditional imports - only import when needed to avoid dependency issues
try:
    from .. import config
except ImportError:
    try:
        import config
    except ImportError:
        config = None
        logging.warning("Config module not available")

# Optional imports for LlamaIndex - will be imported when needed
LLAMA_INDEX_AVAILABLE = False
try:
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
    from llama_index.llms.azure_openai import AzureOpenAI
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import PromptHelper, PromptTemplate, Settings
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    logging.warning("LlamaIndex not available - some features may be limited")

# Optional OpenAI import
try:
    from openai import AzureOpenAI as OpenAIAzure  
except ImportError:
    logging.warning("OpenAI library not available")

class ConfigManager:
    """Manages all configuration settings for the application"""
    
    def __init__(self):
        self._load_api_keys()
        self._load_model_settings()
        self._load_templates()
        self._load_folders()
        self._setup_llama_index()
        
    def _load_api_keys(self):
        """Load all API keys and endpoints"""
        if config is None:
            logging.warning("Config not available, using default/empty values")
            # Set defaults when config is not available
            self.azure_api_key = None
            self.azure_api_base = None
            self.azure_embeddingapi_version = None
            self.azure_chatapi_version = None
            self.azure_gpt4_deploymentid = None
            self.openai_gpt4_modelname = "gpt-4o"
            self.azure_gpt35_deploymentid = None
            self.openai_gpt35_modelname = "gpt-3.5-turbo"
            self.azure_embedding_deploymentid = None
            self.openai_embedding_modelname = "text-embedding-ada-002"
            self.openai_api_key = None
            self.openai_api_base = None
            self.anthropic_api_key = None
            self.google_api_key = None
            self.cohere_api_key = None
            self.bing_search_key = None
            return
            
        # Azure OpenAI
        self.azure_api_key = config.azure_api_key
        self.azure_api_base = config.azure_api_base
        self.azure_embeddingapi_version = config.azure_embeddingapi_version
        self.azure_chatapi_version = config.azure_chatapi_version
        self.azure_gpt4_deploymentid = config.azure_gpt4_deploymentid
        self.openai_gpt4_modelname = config.openai_gpt4_modelname
        self.azure_gpt35_deploymentid = config.azure_gpt35_deploymentid
        self.openai_gpt35_modelname = config.openai_gpt35_modelname
        self.azure_embedding_deploymentid = config.azure_embedding_deploymentid
        self.openai_embedding_modelname = config.openai_embedding_modelname
        
        # OpenAI direct API
        self.openai_api_key = getattr(config, 'openai_api_key', None)
        self.openai_api_base = getattr(config, 'openai_base_url', None)  # Note: config uses openai_base_url
        
        # Other AI services
        self.cohere_api_key = getattr(config, 'cohere_api_key', None)
        self.google_api_key = getattr(config, 'google_api_key', None)
        self.gemini_model_name = getattr(config, 'gemini_model_name', None)
        self.groq_api_key = getattr(config, 'groq_api_key', None)
        
        # Search and tools
        self.bing_api_key = getattr(config, 'bing_api_key', None)
        self.bing_endpoint = getattr(config, 'bing_endpoint', None)
        self.bing_news_endpoint = getattr(config, 'bing_news_endpoint', None)
        self.openweather_api_key = getattr(config, 'openweather_api_key', None)
        
        # Azure services
        self.azurespeechkey = getattr(config, 'azurespeechkey', None)
        self.azurespeechregion = getattr(config, 'azurespeechregion', None)
        self.azuretexttranslatorkey = getattr(config, 'azuretexttranslatorkey', None)
        
        # Local services
        self.rvctts_api_base = getattr(config, 'rvctts_api_base', None)
        self.ollama_api_key = getattr(config, 'ollama_api_key', None)
        self.ollama_api_base = getattr(config, 'ollama_api_base', None)
        
    def _load_model_settings(self):
        """Load model-specific settings"""
        if config is None:
            self.temperature = 0.7
            self.max_tokens = 1000
            self.model_name = "gpt-3.5-turbo"
            self.num_output = 1000
            self.max_chunk_overlap_ratio = 0.1
            self.max_input_size = 4000
            self.context_window = 4000
            self.keywords = []
            self.lite_mode = False
            return
            
        self.temperature = getattr(config, 'temperature', 0.7)
        self.max_tokens = getattr(config, 'max_tokens', 1000)
        self.model_name = getattr(config, 'model_name', "gpt-3.5-turbo")
        self.num_output = getattr(config, 'num_output', 1000)
        self.max_chunk_overlap_ratio = getattr(config, 'max_chunk_overlap_ratio', 0.1)
        self.max_input_size = getattr(config, 'max_input_size', 4000)
        self.context_window = getattr(config, 'context_window', 4000)
        self.keywords = getattr(config, 'keywords', [])
        self.lite_mode = False
        
    def _load_templates(self):
        """Load prompt templates"""
        self.sum_template = getattr(config, 'sum_template', "")
        self.eg_template = getattr(config, 'eg_template', "")
        self.ques_template = getattr(config, 'ques_template', "")
        self.example_queries = getattr(config, 'example_queries', [])
        
        # Only create LlamaIndex templates if available
        if LLAMA_INDEX_AVAILABLE:
            self.summary_template = PromptTemplate(self.sum_template)
            self.example_template = PromptTemplate(self.eg_template)
            self.qa_template = PromptTemplate(self.ques_template)
        else:
            self.summary_template = None
            self.example_template = None
            self.qa_template = None
        
    def _load_folders(self):
        """Load and ensure folder structure exists"""
        if config is None:
            self.UPLOAD_FOLDER = "data/uploads"
            self.SUMMARY_FOLDER = "data/summary_index"
            self.VECTOR_FOLDER = "data/vector_index"
        else:
            self.UPLOAD_FOLDER = getattr(config, 'UPLOAD_FOLDER', "data/uploads")
            self.SUMMARY_FOLDER = getattr(config, 'SUMMARY_FOLDER', "data/summary_index")
            self.VECTOR_FOLDER = getattr(config, 'VECTOR_FOLDER', "data/vector_index")
        
        # Ensure folders exist
        for folder in [self.UPLOAD_FOLDER, self.SUMMARY_FOLDER, self.VECTOR_FOLDER]:
            os.makedirs(folder, exist_ok=True)
            
    def _setup_llama_index(self):
        """Setup LlamaIndex settings"""
        if not LLAMA_INDEX_AVAILABLE:
            logging.warning("LlamaIndex not available - skipping setup")
            return

        # Defensive checks for required classes and config values
        missing = []
        if 'OpenAIAzure' not in globals() or OpenAIAzure is None:
            missing.append('OpenAIAzure')
        if 'AzureOpenAI' not in globals() or AzureOpenAI is None:
            missing.append('AzureOpenAI')
        if 'AzureOpenAIEmbedding' not in globals() or AzureOpenAIEmbedding is None:
            missing.append('AzureOpenAIEmbedding')
        if 'SentenceSplitter' not in globals() or SentenceSplitter is None:
            missing.append('SentenceSplitter')
        if 'PromptHelper' not in globals() or PromptHelper is None:
            missing.append('PromptHelper')
        if 'Settings' not in globals() or Settings is None:
            missing.append('Settings')
        # Check for required config values
        required_config = [
            'azure_api_key', 'azure_api_base', 'azure_chatapi_version',
            'azure_gpt4_deploymentid', 'temperature', 'max_tokens',
            'azure_embedding_deploymentid', 'azure_embeddingapi_version',
            'max_input_size', 'num_output', 'max_chunk_overlap_ratio'
        ]
        for attr in required_config:
            if getattr(self, attr, None) is None:
                missing.append(attr)
        if missing:
            logging.error(f"LlamaIndex setup skipped due to missing: {missing}")
            return

        try:
            Settings.client = OpenAIAzure(
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_api_base,
                api_version=self.azure_chatapi_version,
            )
            Settings.llm = AzureOpenAI(
                azure_deployment=self.azure_gpt4_deploymentid,
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_api_base,
                api_version=self.azure_chatapi_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            Settings.embed_model = AzureOpenAIEmbedding(
                azure_deployment=self.azure_embedding_deploymentid,
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_api_base,
                api_version=self.azure_embeddingapi_version,
                max_retries=3,
                embed_batch_size=1,
            )
            text_splitter = SentenceSplitter()
            Settings.text_splitter = text_splitter
            Settings.prompt_helper = PromptHelper(
                self.max_input_size, 
                self.num_output, 
                self.max_chunk_overlap_ratio
            )
        except Exception as e:
            logging.error(f"Failed to setup LlamaIndex: {e}")
            # Log all relevant values for debugging
            logging.error(f"azure_api_key: {self.azure_api_key}")
            logging.error(f"azure_api_base: {self.azure_api_base}")
            logging.error(f"azure_chatapi_version: {self.azure_chatapi_version}")
            logging.error(f"azure_gpt4_deploymentid: {self.azure_gpt4_deploymentid}")
            logging.error(f"temperature: {self.temperature}")
            logging.error(f"max_tokens: {self.max_tokens}")
            logging.error(f"azure_embedding_deploymentid: {self.azure_embedding_deploymentid}")
            logging.error(f"azure_embeddingapi_version: {self.azure_embeddingapi_version}")
            logging.error(f"max_input_size: {self.max_input_size}")
            logging.error(f"num_output: {self.num_output}")
            logging.error(f"max_chunk_overlap_ratio: {self.max_chunk_overlap_ratio}")
            logging.error(f"LLAMA_INDEX_AVAILABLE: {LLAMA_INDEX_AVAILABLE}")
            logging.error(f"OpenAIAzure: {OpenAIAzure if 'OpenAIAzure' in globals() else 'not imported'}")
            logging.error(f"AzureOpenAI: {AzureOpenAI if 'AzureOpenAI' in globals() else 'not imported'}")
            logging.error(f"AzureOpenAIEmbedding: {AzureOpenAIEmbedding if 'AzureOpenAIEmbedding' in globals() else 'not imported'}")
            logging.error(f"SentenceSplitter: {SentenceSplitter if 'SentenceSplitter' in globals() else 'not imported'}")
            logging.error(f"PromptHelper: {PromptHelper if 'PromptHelper' in globals() else 'not imported'}")
            logging.error(f"Settings: {Settings if 'Settings' in globals() else 'not imported'}")
        
    def get_openai_completion_options(self):
        """Get OpenAI completion options"""
        return {
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

# Global instance - created lazily
_config_manager = None

def get_config_manager():
    """Get or create the global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
