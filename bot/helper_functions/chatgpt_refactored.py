"""
Main ChatGPT class using refactored helper modules
"""
import logging
from io import BytesIO
import config
from .config_manager import get_config_manager
from .token_utils import TokenCounter
from .message_utils import MessageFormatter
from .model_providers import ModelProviders
from .search_utils import SearchUtils

class ChatGPT:
    """Main ChatGPT class with modular helper functions"""
    
    def __init__(self, model="gpt-4o"):
        # Only models defined in config/models.yml are supported
        assert model in {"gpt-4o", "o4-mini", "model-router", "gemini-2.5-flash-preview-05-20"}, f"Unknown model: {model}"
        
        self.model = model
        self.config = get_config_manager()
        self.token_counter = TokenCounter()
        self.message_formatter = MessageFormatter()
        self.model_providers = ModelProviders(self.config)
        self.search_utils = SearchUtils(self.config)
        
    def _get_token_count_model(self):
        """Get the appropriate model name for token counting"""
        if self.model == "gpt-4o":
            return "gpt-4o"
        elif self.model == "gpt-3p5-turbo-16k":
            return "gpt-3.5-turbo-16k"
        return self.model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        """Send a message to the AI model"""
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        token_count_model = self._get_token_count_model()
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        completion_options = self.config.get_openai_completion_options()
        
        while answer is None:
            try:
                messages = self.message_formatter.generate_prompt_messages(
                    message, dialog_messages, chat_mode
                )
                
                # Route to provider based on supported models
                if self.model in {"gpt-4o", "o4-mini", "model-router"}:
                    answer = await self.model_providers.send_azure_openai_message(
                        self.model, messages, completion_options
                    )
                elif self.model == "gemini-2.5-flash-preview-05-20":
                    answer = await self.model_providers.send_gemini_message(messages)
                else:
                    raise ValueError(f"Unsupported model: {self.model}")

                answer = self.message_formatter.postprocess_answer(answer)
                n_input_tokens, n_output_tokens = self.token_counter.count_tokens_from_messages(
                    messages, answer, model=token_count_model
                )
                
            except Exception as e:
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e
                # Forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        """Send a streaming message to the AI model"""
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        token_count_model = self._get_token_count_model()
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        completion_options = self.config.get_openai_completion_options()
        
        while answer is None:
            try:
                messages = self.message_formatter.generate_prompt_messages(
                    message, dialog_messages, chat_mode
                )
                
                if self.model in {"gpt-4o", "o4-mini", "model-router"}:
                    r_gen = await self.model_providers.send_azure_openai_stream(
                        self.model, messages, completion_options
                    )
                    answer = ""
                    for r_item in r_gen:
                        if r_item.choices:
                            delta = r_item.choices[0].delta
                            if delta.content:
                                answer += delta.content
                                n_input_tokens, n_output_tokens = self.token_counter.count_tokens_from_messages(
                                    messages, answer, model=token_count_model
                                )
                                n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                                yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                                
                elif self.model == "gemini-2.5-flash-preview-05-20":
                    # no streaming support for gemini, use non-stream fallback
                    answer = await self.model_providers.send_gemini_message(messages)
                    n_input_tokens, n_output_tokens = self.token_counter.count_tokens_from_messages(
                        messages, answer, model=token_count_model
                    )
                    n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                    yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                else:
                    # unsupported or end of Azure stream falls here
                    raise ValueError(f"Unsupported or unstreamable model: {self.model}")
                    
            except Exception as e:
                if len(dialog_messages) == 0:
                    raise e
                # Forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def _send_non_streaming_message(self, messages, chat_mode):
        """Helper method for non-streaming models"""
        # Only fallback for gemini text model
        if self.model == "gemini-2.5-flash-preview-05-20":
            return await self.model_providers.send_gemini_message(messages)
        raise ValueError(f"Unsupported model for non-streaming: {self.model}")

    async def send_vision_message(self, message, dialog_messages=[], chat_mode="assistant", image_buffer: BytesIO = None):
        """Send a vision message with optional image"""
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        token_count_model = self.model
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        completion_options = self.config.get_openai_completion_options()
        
        if image_buffer is not None:
            image_buffer.seek(0)
            
        while answer is None:
            try:
                # For vision tasks, use gpt-4o as fallback if current model doesn't support vision
                vision_model = self.model if self.model in ["gpt-4o", "gemini-2.5-flash-preview-05-20"] else "gpt-4o"
                logging.info(f"Using vision model: {vision_model} for non-streaming (original model: {self.model})")
                
                messages = self.message_formatter.generate_vision_prompt_messages(
                    message, dialog_messages, chat_mode, image_buffer
                )
                answer = await self.model_providers.send_azure_openai_message(
                    vision_model, messages, completion_options
                )
                n_input_tokens, n_output_tokens = self.token_counter.count_tokens_from_messages(
                    messages, answer, model=token_count_model
                )

                answer = self.message_formatter.postprocess_answer(answer)
                
            except Exception as e:
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e
                # Forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
        return (answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed)

    async def send_vision_message_stream(self, message, dialog_messages=[], chat_mode="assistant", image_buffer: BytesIO = None):
        """Send a streaming vision message with optional image"""
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        token_count_model = self.model
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        completion_options = self.config.get_openai_completion_options()
        
        if image_buffer is not None:
            image_buffer.seek(0)
            
        while answer is None:
            try:
                # For vision tasks, use gpt-4o as fallback if current model doesn't support vision
                vision_model = self.model if self.model in ["gpt-4o", "gemini-2.5-flash-preview-05-20"] else "gpt-4o"
                logging.info(f"Using vision model: {vision_model} (original model: {self.model})")
                
                messages = self.message_formatter.generate_vision_prompt_messages(
                    message, dialog_messages, chat_mode, image_buffer
                )
                r_gen = await self.model_providers.send_azure_openai_stream(
                    vision_model, messages, completion_options
                )
                answer = ""
                for r_item in r_gen:
                    if r_item.choices:
                        delta = r_item.choices[0].delta
                        if delta.content:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self.token_counter.count_tokens_from_messages(
                                messages, answer, model=token_count_model
                            )
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            logging.debug(f"Yielding partial answer. Tokens: input={n_input_tokens}, output={n_output_tokens}, messages_removed={n_first_dialog_messages_removed}")
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            logging.debug(f"Yielding partial answer. Tokens: input={n_input_tokens}, output={n_output_tokens}, messages_removed={n_first_dialog_messages_removed}")
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                    
            except Exception as e:
                logging.error(f"Exception occurred: {e}")
                if len(dialog_messages) == 0:
                    raise e
                # Forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_internetmessage(self, message, dialog_messages=[], chat_mode="internet_connected_assistant"):
        """Send an internet-connected message with search capabilities"""
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        token_count_model = self._get_token_count_model()
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        
        while answer is None:
            try:
                messages = self.message_formatter.generate_prompt_messages(
                    message, dialog_messages, chat_mode
                )
                
                if "news" in message.lower():
                    answer = self.search_utils.get_bing_news_results(message)
                elif "weather" in message.lower():
                    answer = self.search_utils.get_weather_data(message)
                else:
                    answer = self.search_utils.get_bing_results(message)
                
                n_input_tokens, n_output_tokens = self.token_counter.count_tokens_from_messages(
                    messages, answer, model="gpt-3.5-turbo"
                )
                n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                
                answer = self.message_formatter.postprocess_answer(answer)

            except Exception as e:
                if len(dialog_messages) == 0:
                    raise e
                # Forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]
            
        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    # Legacy method compatibility
    def _generate_prompt(self, message, dialog_messages, chat_mode):
        """Legacy method for backward compatibility"""
        return self.message_formatter.generate_prompt(message, dialog_messages, chat_mode)
        
    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):
        """Legacy method for backward compatibility"""
        return self.message_formatter.generate_prompt_messages(message, dialog_messages, chat_mode)
        
    def _generate_vision_prompt_messages(self, message, dialog_messages, chat_mode, image_buffer: BytesIO = None):
        """Legacy method for backward compatibility"""
        return self.message_formatter.generate_vision_prompt_messages(message, dialog_messages, chat_mode, image_buffer)
        
    def _postprocess_answer(self, answer):
        """Legacy method for backward compatibility"""
        return self.message_formatter.postprocess_answer(answer)
        
    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        """Legacy method for backward compatibility"""
        return self.token_counter.count_tokens_from_messages(messages, answer, model)
        
    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        """Legacy method for backward compatibility"""
        return self.token_counter.count_tokens_from_prompt(prompt, answer, model)
        
    def _encode_image(self, image_buffer: BytesIO):
        """Legacy method for backward compatibility"""
        return self.message_formatter.encode_image(image_buffer)
        
    def _clearallfiles(self):
        """Legacy method for backward compatibility"""
        return self.search_utils.clear_all_files()
        
    def _text_extractor(self, url, debug=False):
        """Legacy method for backward compatibility"""
        return self.search_utils.text_extractor(url, debug)
        
    def _saveextractedtext_to_file(self, text, filename):
        """Legacy method for backward compatibility"""
        return self.search_utils.save_extracted_text_to_file(text, filename)
        
    def _get_bing_results(self, query, num=10):
        """Legacy method for backward compatibility"""
        return self.search_utils.get_bing_results(query, num)
        
    def _get_bing_news_results(self, query, num=5):
        """Legacy method for backward compatibility"""
        return self.search_utils.get_bing_news_results(query, num)
        
    def _get_bing_agent(self, query):
        """Legacy method for backward compatibility"""
        return self.search_utils.get_bing_agent(query)
        
    def _get_weather_data(self, query):
        """Legacy method for backward compatibility"""
        return self.search_utils.get_weather_data(query)
        
    def _summarize(self, data_folder):
        """Legacy method for backward compatibility"""
        from .index_utils import IndexUtils
        index_utils = IndexUtils(self.config)
        return index_utils.summarize(data_folder)
        
    def _simple_query(self, data_folder, query):
        """Legacy method for backward compatibility"""
        from .index_utils import IndexUtils
        index_utils = IndexUtils(self.config)
        return index_utils.simple_query(data_folder, query)
