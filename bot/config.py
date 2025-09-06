import yaml
import dotenv
from pathlib import Path

config_dir = Path(__file__).parent.parent.resolve() / "config"

# load yaml config
with open(config_dir / "config.yml", 'r') as f:
    config_yaml = yaml.safe_load(f)

# load .env config
config_env = dotenv.dotenv_values(config_dir / "config.env")

# config parameters
telegram_token = config_yaml["telegram_token"]
azure_api_base = config_yaml["azure_api_base"]
azure_api_key = config_yaml["azure_api_key"]
azure_chatapi_version = config_yaml["azure_chatapi_version"]
azure_embeddingapi_version = config_yaml["azure_embeddingapi_version"]

# Generic LLM identifiers (new)
smart_llm_azure_deployment = config_yaml.get("smart_llm_azure_deployment")
smart_llm_openai_model = config_yaml.get("smart_llm_openai_model")
thinking_llm_azure_deployment = config_yaml.get("thinking_llm_azure_deployment")
thinking_llm_openai_model = config_yaml.get("thinking_llm_openai_model")
fast_llm_azure_deployment = config_yaml.get("fast_llm_azure_deployment")
fast_llm_openai_model = config_yaml.get("fast_llm_openai_model")
superfast_llm_azure_deployment = config_yaml.get("superfast_llm_azure_deployment")
embedding_azure_deployment = config_yaml.get("embedding_azure_deployment")
embedding_openai_model = config_yaml.get("embedding_openai_model")

# Backward compatibility aliases (old variable names still expected elsewhere)

# OpenAI direct API
openai_api_key = config_yaml["openai_api_key"]
openai_base_url = config_yaml["openai_base_url"]

ollama_api_key = config_yaml["ollama_api_key"]
ollama_api_base = config_yaml["ollama_api_base"]

rvctts_api_base = config_yaml["rvctts_api_base"]

cohere_api_key = config_yaml["cohere_api_key"]
google_api_key = config_yaml["google_api_key"]
gemini_model_name = config_yaml["gemini_model_name"]
groq_api_key = config_yaml["groq_api_key"]

bing_api_key = config_yaml["bing_api_key"]
bing_endpoint = config_yaml["bing_endpoint"] + "/v7.0/search"
bing_news_endpoint = config_yaml["bing_endpoint"] + "/v7.0/news/search"

# Firecrawl (self-hosted) search service
firecrawl_base_url = config_yaml.get("firecrawl_base_url", "http://10.0.0.107:3002")
firecrawl_api_key = config_yaml.get("firecrawl_api_key", "fc-YOUR_API_KEY")

azurespeechkey = config_yaml["azurespeechkey"]
azurespeechregion = config_yaml["azurespeechregion"]
azuretexttranslatorkey = config_yaml["azuretexttranslatorkey"]

openweather_api_key = config_yaml["openweather_api_key"]

UPLOAD_FOLDER = config_yaml['paths']['UPLOAD_FOLDER']
BING_FOLDER = config_yaml['paths']['BING_FOLDER']
SUMMARY_FOLDER = config_yaml['paths']['SUMMARY_FOLDER']
VECTOR_FOLDER = config_yaml['paths']['VECTOR_FOLDER']

temperature = config_yaml['settings']['temperature']
max_tokens = config_yaml['settings']['max_tokens']
model_name = config_yaml['settings']['model_name']
num_output = config_yaml['settings']['num_output']
max_chunk_overlap_ratio = config_yaml['settings']['max_chunk_overlap_ratio']
max_input_size = config_yaml['settings']['max_input_size']
context_window = config_yaml['settings']['context_window']

use_chatgpt_api = config_yaml.get("use_chatgpt_api", True)
allowed_telegram_usernames = config_yaml["allowed_telegram_usernames"]
new_dialog_timeout = config_yaml["new_dialog_timeout"]
enable_message_streaming = config_yaml.get("enable_message_streaming", True)
return_n_generated_images = config_yaml.get("return_n_generated_images", 1)
n_chat_modes_per_page = config_yaml.get("n_chat_modes_per_page", 5)
mongodb_uri = f"mongodb://mongo:{config_env['MONGODB_PORT']}"

# chat_modes
with open(config_dir / "chat_modes.yml", 'r') as f:
    chat_modes = yaml.safe_load(f)

# models
with open(config_dir / "models.yml", 'r') as f:
    models = yaml.safe_load(f)

# files
help_group_chat_video_path = Path(__file__).parent.parent.resolve() / "static" / "help_group_chat.mp4"


# load prompts.yml config
with open(config_dir / "prompts.yml", 'r') as f:
    prompts_config = yaml.safe_load(f)

# Accessing the templates
sum_template = prompts_config["sum_template"]
eg_template = prompts_config["eg_template"]
ques_template = prompts_config["ques_template"]

system_prompt_content = prompts_config["system_prompt_content"]
system_prompt = [{
    "role": "system",
    "content": system_prompt_content
}]

example_queries = prompts_config['example_queries']
keywords = prompts_config['keywords']
