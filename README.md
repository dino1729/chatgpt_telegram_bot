## LLMs on Telegram!

This repo is a fork of https://github.com/karfly/chatgpt_telegram_bot

Following are the models supported:
Azure OpenAI:
gpt-35-turbo-16k
gpt-4

Palm
Cohere
LocalAI: Wizard LM

Bing search with Azure OpenaAI mode available to get the latest information!

## Bot commands
- `/retry` – Regenerate last bot answer
- `/new` – Start new dialog
- `/mode` – Select chat mode
- `/balance` – Show balance
- `/settings` – Show settings
- `/help` – Show help

## Setup
1. Get your [OpenAI API](https://openai.com/api/) key

2. Get your Telegram bot token from [@BotFather](https://t.me/BotFather)

3. Edit `config/config.example.yml` to set your tokens and run 2 commands below (*if you're advanced user, you can also edit* `config/config.example.env`):
    ```bash
    mv config/config.example.yml config/config.yml
    mv config/config.example.env config/config.env
    ```

4. 🔥 And now **run**:
    ```bash
    docker-compose --env-file config/config.env up --build
    ```

