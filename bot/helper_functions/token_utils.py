"""
Token counting utilities for various language models
"""
import tiktoken
import logging

class TokenCounter:
    """Handles token counting for different language models"""
    
    @staticmethod
    def count_tokens_from_messages(messages, answer, model="gpt-3.5-turbo"):
        """
        Count tokens from messages and answer for a given model
        
        Args:
            messages: List of message dictionaries
            answer: Response string
            model: Model name for token counting
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # If model not found, use default
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Define default token values
        default_n_input_tokens = 0
        default_n_output_tokens = 0

        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4o":
            tokens_per_message = 3
            tokens_per_name = -1
        else:
            # If the model is unknown, return default token values
            return default_n_input_tokens, default_n_output_tokens

        # Count input tokens
        n_input_tokens = 0
        try:
            for message in messages:
                n_input_tokens += tokens_per_message
                for key, value in message.items():
                    # Ensure that the value is a string or convert it if possible
                    if isinstance(value, list):
                        value = "\n".join(str(item) for item in value) if all(isinstance(item, str) for item in value) else ""
                    elif not isinstance(value, str):
                        value = str(value) if value is not None else ""
                    n_input_tokens += len(encoding.encode(value))
                    if key == "name":
                        n_input_tokens += tokens_per_name
            n_input_tokens += 2
            
            # Count output tokens
            n_output_tokens = 1 + len(encoding.encode(str(answer)))

        except Exception as e:
            # If there's an error during token counting, log the error and return default values
            logging.exception(f"An error occurred during token counting: {e}")
            return default_n_input_tokens, default_n_output_tokens

        return n_input_tokens, n_output_tokens

    @staticmethod
    def count_tokens_from_prompt(prompt, answer, model="text-davinci-003"):
        """
        Count tokens from prompt and answer for completion models
        
        Args:
            prompt: Input prompt string
            answer: Response string
            model: Model name for token counting
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            n_input_tokens = len(encoding.encode(prompt)) + 1
            n_output_tokens = len(encoding.encode(answer))
            return n_input_tokens, n_output_tokens
        except Exception as e:
            logging.exception(f"An error occurred during prompt token counting: {e}")
            return 0, 0
