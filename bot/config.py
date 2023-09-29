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
openai_api_key = config_yaml["openai_api_key"]
openai_api_dallekey = config_yaml["openai_api_dallekey"]
openai_api_base = config_yaml["openai_api_base"]
openai_api_dallebase = config_yaml["openai_api_dallebase"]
openai_api_version = config_yaml["openai_api_version"]
openai_embeddingapi_version = config_yaml["openai_embeddingapi_version"]

llama2_api_key = config_yaml["llama2_api_key"]
llama2_api_base = config_yaml["llama2_api_base"]

cohere_api_key = config_yaml["cohere_api_key"]
google_palm_api_key = config_yaml["google_palm_api_key"]

bing_api_key = config_yaml["bing_api_key"]
bing_endpoint = config_yaml["bing_endpoint"] + "/v7.0/search"
bing_news_endpoint = config_yaml["bing_endpoint"] + "/v7.0/news/search"

azurespeechkey = config_yaml["azurespeechkey"]
azurespeechregion = config_yaml["azurespeechregion"]
azuretexttranslatorkey = config_yaml["azuretexttranslatorkey"]

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
