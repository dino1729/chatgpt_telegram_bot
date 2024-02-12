import config
import uuid, requests
import cohere
import tiktoken
import base64
from io import BytesIO
import os
import logging
import sys
import json
import google.generativeai as palm
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from mimetypes import guess_type
from openai import OpenAI
from openai import AzureOpenAI as OpenAIAzure
from newspaper import Article
from bs4 import BeautifulSoup
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    PromptHelper,
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
    set_global_service_context,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.agent import OpenAIAgent
from llama_hub.tools.weather.base import OpenWeatherMapToolSpec
from llama_hub.tools.bing_search.base import BingSearchToolSpec

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#UPLOAD_FOLDER = './data'  # set the upload folder path
UPLOAD_FOLDER = os.path.join(".", "data")
SUMMARY_FOLDER = os.path.join(UPLOAD_FOLDER, "summary_index")
VECTOR_FOLDER = os.path.join(UPLOAD_FOLDER, "vector_index")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SUMMARY_FOLDER ):
    os.makedirs(SUMMARY_FOLDER)
if not os.path.exists(VECTOR_FOLDER ):
    os.makedirs(VECTOR_FOLDER)


cohere_api_key = config.cohere_api_key
google_api_key = config.google_api_key

azure_api_type = "azure"
azure_api_base = config.azure_api_base
azure_api_key = config.azure_api_key
azure_chatapi_version = config.azure_chatapi_version
azure_embeddingapi_version = config.azure_embeddingapi_version

bing_api_key = config.bing_api_key
bing_endpoint = config.bing_endpoint
bing_news_endpoint = config.bing_news_endpoint

llama2_api_type = "open_ai"
llama2_api_key = config.llama2_api_key
llama2_api_base = config.llama2_api_base

rvctts_api_base = config.rvctts_api_base

azurespeechkey = config.azurespeechkey
azurespeechregion = config.azurespeechregion
azuretexttranslatorkey = config.azuretexttranslatorkey

openweather_api_key = config.openweather_api_key

num_output = 1024
max_chunk_overlap_ratio = 0.1
chunk_size = 256
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
# Check if user set the davinci model flag
client = OpenAIAzure(
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
)
# Check if user set the davinci model flag
gpt4_flag = True
if gpt4_flag:
    LLM_DEPLOYMENT_NAME = "gpt-4-turbo"
    LLM_MODEL_NAME = "gpt-4-turbo"
    max_input_size = 128000
    context_window = 128000
    print("Using gpt4 model.")
else:
    LLM_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
    LLM_MODEL_NAME = "gpt-35-turbo-16k"
    max_input_size = 48000
    context_window = 16000
    print("Using gpt-35-turbo-16k model.")

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

llm = AzureOpenAI(
    engine=LLM_DEPLOYMENT_NAME, 
    model=LLM_MODEL_NAME,
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_chatapi_version,
    temperature=0.25,
    max_tokens=num_output,
)
embedding_llm =AzureOpenAIEmbedding(
    model=EMBEDDINGS_DEPLOYMENT_NAME,
    azure_deployment=EMBEDDINGS_DEPLOYMENT_NAME,
    api_key=azure_api_key,
    azure_endpoint=azure_api_base,
    api_version=azure_embeddingapi_version,
    max_retries=3,
    embed_batch_size=1,
)

splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embedding_llm)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper,
    chunk_size=chunk_size,
    context_window=context_window,
    node_parser=splitter,
)
set_global_service_context(service_context)

sum_template = (
    "You are a world-class text summarizer connected to the internet. We have provided context information from the internet below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Based on the context provided, your task is to summarize the input context while effectively conveying the main points and relevant information. The summary should be presented in a numbered list of at least 10 key points and takeaways. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the latest context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
)
summary_template = PromptTemplate(sum_template)
ques_template = (
    "You are a world-class personal assistant connected to the internet. You will be provided snippets of information from the internet based on user's query. Here is the context:\n"
    "---------------------\n"
    "{context_str}\n"
    "\n---------------------\n"
    "Based on the context provided, your task is to answer the user's question to the best of your ability. It is important to refrain from directly copying word-for-word from the original context. Additionally, please ensure that the summary excludes any extraneous details such as discounts, promotions, sponsorships, or advertisements, and remains focused on the core message of the content.\n"
    "---------------------\n"
    "Using both the latest context information and also using your own knowledge, "
    "answer the question: {query_str}\n"
)
qa_template = PromptTemplate(ques_template)

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.5,
    "max_tokens": 1024,
    "top_p": 0.9,
    "frequency_penalty": 0.6,
    "presence_penalty": 0.1
}

class ChatGPT:
    def __init__(self, model="gpt-4-turbo"):
        assert model in {"gpt-4-turbo", "gpt-4", "gpt-35-turbo-16k", "cohere", "palm", "gemini", "mixtral8x7b"}, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        # Convert model names for token counting
        token_count_model = self.model
        if token_count_model == "gpt-4-turbo":
            token_count_model = "gpt-4"
        elif token_count_model == "gpt-35-turbo-16k":
            token_count_model = "gpt-3.5-turbo-16k"

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-4-turbo", "gpt-4", "gpt-35-turbo-16k"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    client = OpenAIAzure(
                        api_key=azure_api_key,
                        azure_endpoint=azure_api_base,
                        api_version=azure_chatapi_version,
                    )
                    r = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=token_count_model)
                elif self.model == "mixtral8x7b":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    local_client = OpenAI(
                        api_key = llama2_api_key,
                        api_base = llama2_api_base
                    )
                    r = local_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                elif self.model == "cohere":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    co = cohere.Client(cohere_api_key)
                    r = co.generate(
                        model='command-nightly',
                        prompt=str(messages).replace("'", '"'),
                        temperature=0.5,
                        max_tokens=1024,
                    )
                    answer = r.generations[0].text
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                elif self.model == "palm":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    palm.configure(api_key=google_api_key)
                    r = await palm.chat_async(
                        model="models/chat-bison-001",
                        messages=str(messages).replace("'", '"'),
                        temperature=0.5,
                    )
                    answer = r.last
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                elif self.model == "gemini":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    genai.configure(api_key=google_api_key)
                    gemini = genai.GenerativeModel('gemini-pro')
                    r = await gemini.generate_content_async(str(messages).replace("'", '"'))
                    answer = r.text
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
            # except openai.error.InvalidRequestError as e:  # too many tokens
            #     if len(dialog_messages) == 0:
            #         raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e
            except Exception as e:
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        # Convert model names for token counting
        token_count_model = self.model
        if token_count_model == "gpt-4-turbo":
            token_count_model = "gpt-4"
        elif token_count_model == "gpt-35-turbo-16k":
            token_count_model = "gpt-3.5-turbo-16k"

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                # Chat models
                if self.model in {"gpt-4-turbo", "gpt-4", "gpt-35-turbo-16k"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    client = OpenAIAzure(
                        api_key=azure_api_key,
                        azure_endpoint=azure_api_base,
                        api_version=azure_chatapi_version,
                    )                 
                    r_gen = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = ""
                    for r_item in r_gen:
                        if r_item.choices:
                            delta = r_item.choices[0].delta
                            if delta.content:
                                answer += delta.content
                                n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=token_count_model)
                                n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                                yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                elif self.model == "mixtral8x7b":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    local_client = OpenAI(
                        api_key = llama2_api_key,
                        api_base = llama2_api_base
                    )
                    r_gen = await local_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r_gen.choices[0].message["content"]
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                elif self.model == "cohere":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    co = cohere.Client(cohere_api_key)
                    r_gen = co.generate(
                        model='command-nightly',
                        prompt=str(messages).replace("'", '"'),
                        temperature=0.5,
                        max_tokens=1024,
                    )
                    answer = r_gen.generations[0].text
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                elif self.model == "palm":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    palm.configure(api_key=google_api_key)
                    r_gen = await palm.chat_async(
                        model="models/chat-bison-001",
                        messages=str(messages).replace("'", '"'),
                        temperature=0.5,
                    )
                    answer = r_gen.last
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                elif self.model == "gemini":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    genai.configure(api_key=google_api_key)
                    gemini = genai.GenerativeModel('gemini-pro')
                    r_gen = await gemini.generate_content_async(str(messages).replace("'", '"'))
                    answer = r_gen.text
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
            # except openai.error.InvalidRequestError as e:  # too many tokens
            #     if len(dialog_messages) == 0:
            #         raise e
            except Exception as e:
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    async def send_vision_message(self, message, dialog_messages=[], chat_mode="assistant", image_buffer: BytesIO = None):
            n_dialog_messages_before = len(dialog_messages)
            answer = None
            while answer is None:
                try:
                    if self.model == "gpt-4":
                        messages = self._generate_prompt_messages(message, dialog_messages, chat_mode, image_buffer)
                        image_client = OpenAIAzure(
                            api_key=azure_api_key,
                            azure_endpoint=azure_api_base,
                            api_version=azure_chatapi_version,
                        )
                        r = image_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **OPENAI_COMPLETION_OPTIONS
                        )
                        answer = r.choices[0].message.content
                    else:
                        raise ValueError(f"Unsupported model: {self.model}")

                    answer = self._postprocess_answer(answer)
                    n_input_tokens, n_output_tokens = (r.usage.prompt_tokens, r.usage.completion_tokens)
                # except openai.error.InvalidRequestError as e:  # too many tokens
                #     if len(dialog_messages) == 0:
                #         raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e
                except Exception as e:
                    if len(dialog_messages) == 0:
                        raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                    # forget first message in dialog_messages
                    dialog_messages = dialog_messages[1:]

            n_first_dialog_messages_removed = n_dialog_messages_before - len(
                dialog_messages
            )

            return (
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            )

    async def send_vision_message_stream(self, message, dialog_messages=[], chat_mode="assistant", image_buffer: BytesIO = None):
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model == "gpt-4":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode, image_buffer)
                    print("Prompt sent to GPT-4: ", messages)
                    image_client = OpenAIAzure(
                        api_key=azure_api_key,
                        azure_endpoint=azure_api_base,
                        api_version=azure_chatapi_version,
                    )
                    r_gen = image_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = ""
                    for r_item in r_gen:
                        if r_item.choices:
                            delta = r_item.choices[0].delta
                            if delta.content:
                                answer += delta.content
                                (n_input_tokens, n_output_tokens) = self._count_tokens_from_messages(messages, answer, model=self.model)
                                n_first_dialog_messages_removed = (n_dialog_messages_before - len(dialog_messages))
                                yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                answer = self._postprocess_answer(answer)

            # except openai.error.InvalidRequestError as e:  # too many tokens
            #     if len(dialog_messages) == 0:
            #         raise e
            except Exception as e:
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_internetmessage(self, message, dialog_messages=[], chat_mode="internet_connected_assistant"):

        if chat_mode not in config.chat_modes.keys():
                raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        # Convert model names for token counting
        token_count_model = self.model
        if token_count_model == "gpt-4-turbo":
            token_count_model = "gpt-4"
        elif token_count_model == "gpt-35-turbo-16k":
            token_count_model = "gpt-3.5-turbo-16k"

        client = OpenAIAzure(
            api_key=azure_api_key,
            azure_endpoint=azure_api_base,
            api_version=azure_chatapi_version,
        )
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                if "news" in message.lower():
                    answer = self._get_bing_news_results(message)
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                elif "weather" in message.lower():
                    answer = self._get_weather_data(message)
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                else:
                    answer = self._get_bing_results(message)
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo")
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                
                answer = self._postprocess_answer(answer)

            except Exception as e:
                if len(dialog_messages) == 0:
                    raise e
                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]
            
        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _encode_image(self, image_buffer: BytesIO) -> bytes:

        mime_type, _ = guess_type(image_buffer.name)
        if mime_type is None:
            mime_type = 'application/octet-stream' # Default MIME type if none is found

        base64_encoded_data = base64.b64encode(image_buffer.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode, image_buffer: BytesIO = None):
        
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        messages = [{"role": "system", "content": prompt}]

        if image_buffer is None:
            # Text-based interaction
            for dialog_message in dialog_messages:
                messages.append({"role": "user", "content": dialog_message["user"]})
                messages.append({"role": "assistant", "content": dialog_message["bot"]})
            messages.append({"role": "user", "content": message})
        else:
            # Image-based interaction
            for dialog_message in dialog_messages:
                messages.append({
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": dialog_message["user"]
                    }
                })
                messages.append({"role": "assistant", "content": dialog_message["bot"]})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_image(image_buffer)
                        }
                    }
                ]
            })

        return messages

    def _postprocess_answer(self, answer):
        
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        
        encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens
    
    def _clearallfiles(self):
        
        # Ensure the UPLOAD_FOLDER is empty
        for root, dirs, files in os.walk(UPLOAD_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)

    def _text_extractor(self, url):

        if url:
            # Extract the article
            article = Article(url)
            try:
                article.download()
                article.parse()
                #Check if the article text has atleast 75 words
                if len(article.text.split()) < 75:
                    raise Exception("Article is too short. Probably the article is behind a paywall.")
            except Exception as e:
                print("Failed to download and parse article from URL using newspaper package: %s. Error: %s", url, str(e))
                # Try an alternate method using requests and beautifulsoup
                try:
                    req = requests.get(url)
                    soup = BeautifulSoup(req.content, 'html.parser')
                    article.text = soup.get_text()
                except Exception as e:
                    print("Failed to download article using beautifulsoup method from URL: %s. Error: %s", url, str(e))
            return article.text
        else:
            return None

    def _saveextractedtext_to_file(self, text, filename):

        # Save the output to the article.txt file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, 'w') as file:
            file.write(text)

        return f"Text saved to {file_path}"

    def _get_bing_results(self, query, num=10):

        self._clearallfiles()
        # Construct a request
        mkt = 'en-US'
        params = { 'q': query, 'mkt': mkt, 'count': num, 'responseFilter': ['Webpages','News'] }
        headers = { 'Ocp-Apim-Subscription-Key': bing_api_key }
        response = requests.get(bing_endpoint, headers=headers, params=params)
        response_data = response.json()  # Parse the JSON response

        # Extract snippets and append them into a single text variable
        all_snippets = [result['snippet'] for result in response_data['webPages']['value']]
        combined_snippets = '\n'.join(all_snippets)
        
        # Format the results as a string
        output = f"Here is the context from Bing for the query: '{query}':\n"
        output += combined_snippets

        # Save the output to a file
        self._saveextractedtext_to_file(output, "bing_results.txt")
        # Query the results using llama-index
        answer = str(self._simple_query(UPLOAD_FOLDER, query)).strip()

        return answer

    def _get_bing_news_results(self, query, num=5):

        self._clearallfiles()
        # Construct a request
        mkt = 'en-US'
        params = { 'q': query, 'mkt': mkt, 'freshness': 'Day', 'count': num }
        headers = { 'Ocp-Apim-Subscription-Key': bing_api_key }
        response = requests.get(bing_news_endpoint, headers=headers, params=params)
        response_data = response.json()  # Parse the JSON response
        #pprint(response_data)

        # Extract text from the urls and append them into a single text variable
        all_urls = [result['url'] for result in response_data['value']]
        all_snippets = [self._text_extractor(url) for url in all_urls]

        # Combine snippets with titles and article names
        combined_output = ""
        for i, (snippet, result) in enumerate(zip(all_snippets, response_data['value'])):
            title = f"Article {i + 1}: {result['name']}"
            if len(snippet.split()) >= 75:  # Check if article has at least 75 words
                combined_output += f"\n{title}\n{snippet}\n"

        # Format the results as a string
        output = f"Here's scraped text from top {num} articles for: '{query}':\n"
        output += combined_output

        # Save the output to a file
        self._saveextractedtext_to_file(output, "bing_results.txt")
        # Summarize the bing search response
        bingsummary = str(self._summarize(UPLOAD_FOLDER)).strip()

        return bingsummary
    
    def _get_bing_agent(self, query):

        bing_tool = BingSearchToolSpec(
            api_key=bing_api_key,
        )
    
        agent = OpenAIAgent.from_tools(
            bing_tool.to_tool_list(),
            llm=llm,
            verbose=False,
        )
    
        return str(agent.chat(query))
    
    def _get_weather_data(self, query):

        # Initialize OpenWeatherMapToolSpec
        weather_tool = OpenWeatherMapToolSpec(
            key=openweather_api_key,
        )

        agent = OpenAIAgent.from_tools(
            weather_tool.to_tool_list(),
            llm=llm,
            verbose=False,
        )

        return str(agent.chat(query))

    def _summarize(self, data_folder):
          
        # Initialize a document
        documents = SimpleDirectoryReader(data_folder).load_data()
        #index = VectorStoreIndex.from_documents(documents)
        summary_index = SummaryIndex.from_documents(documents)
        # SummaryIndexRetriever
        retriever = summary_index.as_retriever(
            retriever_mode='default',
        )
        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            summary_template=summary_template,
        )
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        response = query_engine.query("Generate a summary of the input context. Be as verbose as possible, while keeping the summary concise and to the point.")

        return response

    def _simple_query(self, data_folder, query):
         
        # Initialize a document
        documents = SimpleDirectoryReader(data_folder).load_data()
        #index = VectorStoreIndex.from_documents(documents)
        vector_index = VectorStoreIndex.from_documents(documents)
        # configure retriever
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=6,
        )
        # # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_template,
        )
        # # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ],
        )
        response = query_engine.query(query)

        return response

async def transcribe_audio(audio_file):
    
    # Create an instance of a speech config with your subscription key and region
    # Currently the v2 endpoint is required. In a future SDK release you won't need to set it. 
    endpoint_string = "wss://{}.stt.speech.microsoft.com/speech/universal/v2".format(azurespeechregion)
    #speech_config = speechsdk.translation.SpeechTranslationConfig(subscription=azurespeechkey, endpoint=endpoint_string)
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_file))
    # set up translation parameters: source language and target languages
    # Currently the v2 endpoint is required. In a future SDK release you won't need to set it. 
    #endpoint_string = "wss://{}.stt.speech.microsoft.com/speech/universal/v2".format(service_region)
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=azurespeechkey,
        endpoint=endpoint_string,
        speech_recognition_language='en-US',
        target_languages=('en','hi','te'))
    #audio_config = speechsdk.audio.AudioConfig(filename=weatherfilename)
    # Specify the AutoDetectSourceLanguageConfig, which defines the number of possible languages
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "hi-IN", "te-IN"])
    # Creates a translation recognizer using and audio file as input.
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, 
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_source_language_config)
    result = recognizer.recognize_once()

    translated_result = format(result.translations['en'])
    detectedSrcLang = format(result.properties[speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult])

    return translated_result, detectedSrcLang

async def translate_text(text, target_language):
    
    # Add your key and endpoint
    key = azuretexttranslatorkey
    endpoint = "https://api.cognitive.microsofttranslator.com"
    # location, also known as region.
    location = azurespeechregion
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': [target_language]
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    # You can pass more than one object in body.
    body = [{
        'text': text
    }]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

async def text_to_speech(text, output_path, language):
    
    speech_config = speechsdk.SpeechConfig(subscription=azurespeechkey, region=azurespeechregion)
    # Set the voice based on the language
    if language == "te-IN":
        speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"
    elif language == "hi-IN":
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
    else:
        # Use a default voice if the language is not specified or unsupported
        speech_config.speech_synthesis_voice_name = "en-US-NancyNeural"
    # Use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Get the audio data from the result object
        audio_data = result.audio_data  
        # Save the audio data as a WAV file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)
            #print("Speech synthesized and saved to WAV file.")

async def local_text_to_speech(text, output_path, model_name):
    
    url = rvctts_api_base
    payload = json.dumps({
      "speaker_name": model_name,
      "input_text": text,
      "emotion": "Angry",
      "speed": 1.5
    })
    headers = {
      'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        audio_content = response.content
        # Save the audio to a file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_content)
    else:
        print("Error:", response.text)

async def generate_images(prompt, n_images=4):

    image_client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version="2023-12-01-preview"
    )
    r = image_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=n_images,
        size="1024x1024",
        )
    image_urls = [item.url for item in r.data]
    return image_urls

async def is_content_acceptable(prompt):
    
    r = await OpenAIAzure.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())
