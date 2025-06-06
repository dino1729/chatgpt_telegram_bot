## LLMs on Telegram!

This repo is a fork of https://github.com/karfly/chatgpt_telegram_bot

Following are the models supported:

Azure OpenAI:
gpt-35-turbo-16k
gpt-4o

Palm
Cohere

## Local Open-source LLM Models
Now with oobabooga's (https://github.com/oobabooga/text-generation-webui) openai endpoint, you can use any of the open-source models running locally on your machine.

LocalAI: Wizard LM

## Bing Search for latest information

One of the fundamental limitation of the gpt-35-turbo-16k model is that its data cutoff is in the year 2021. Now you can hook the gpt-35-turbo-16k model to the internet to get the latest and upto date information.

Bing search with Azure OpenaAI mode to get the latest information!

## Diffusion of LLMs!

Why should only the elite have access to LLMs? The non-zero sum nature of this technology means that it can be used by everyone to improve their lives and become more productive and thus add economic value to the society.

Adding support for voice conversation with LLMs! Just talk to the LLMs like you would talk to siri/google assistant/alexa. Added support for Hindi and Telugu languages to help with the diffusion of this powerful technology to the masses.

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
    docker compose up --build
    ```

