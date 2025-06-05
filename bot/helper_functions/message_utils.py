"""
Utilities for generating prompts and formatting messages
"""
import base64
import logging
from io import BytesIO
from mimetypes import guess_type
import config

class MessageFormatter:
    """Handles message formatting and prompt generation"""
    
    @staticmethod
    def generate_prompt(message, dialog_messages, chat_mode):
        """
        Generate a simple text prompt from messages
        
        Args:
            message: Current user message
            dialog_messages: List of previous dialog messages
            chat_mode: Chat mode configuration key
            
        Returns:
            Formatted prompt string
        """
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # Add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # Current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    @staticmethod
    def generate_prompt_messages(message, dialog_messages, chat_mode):
        """
        Generate structured messages for chat completion API
        
        Args:
            message: Current user message
            dialog_messages: List of previous dialog messages
            chat_mode: Chat mode configuration key
            
        Returns:
            List of formatted message dictionaries
        """
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        messages = []

        messages.append({"role": "system", "content": prompt})
        
        # Add dialog history
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    @staticmethod
    def encode_image(image_buffer: BytesIO) -> str:
        """
        Encode image buffer to base64 data URL
        
        Args:
            image_buffer: BytesIO buffer containing image data
            
        Returns:
            Base64 encoded data URL string
        """
        image_buffer.seek(0)
        mime_type, _ = guess_type(image_buffer.name)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        base64_encoded_data = base64.b64encode(image_buffer.read()).decode("utf-8")
        return f"data:{mime_type};base64,{base64_encoded_data}"

    @staticmethod
    def generate_vision_prompt_messages(message, dialog_messages, chat_mode, image_buffer: BytesIO = None):
        """
        Generate structured messages for vision-enabled chat completion
        
        Args:
            message: Current user message
            dialog_messages: List of previous dialog messages
            chat_mode: Chat mode configuration key
            image_buffer: Optional image buffer for vision input
            
        Returns:
            List of formatted message dictionaries with vision support
        """
        logging.debug("Generating vision prompt messages.")
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        messages = []

        # Reset Buffer
        if image_buffer is not None:
            image_buffer.seek(0)
            logging.debug("Image buffer reset.")

        # System message
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": prompt}]
        })

        # Add dialog history
        for dialog_message in dialog_messages:
            if "user" in dialog_message and dialog_message["user"]:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": dialog_message["user"]}]
                })
            if "bot" in dialog_message and dialog_message["bot"]:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": dialog_message["bot"]}]
                })

        # Current user message with optional image
        user_message_content = [{"type": "text", "text": message}]

        if image_buffer is not None:
            encoded_image = MessageFormatter.encode_image(image_buffer)
            user_message_content.append({
                "type": "image_url",
                "image_url": {"url": encoded_image}
            })
            logging.debug("Encoded image added to user message content.")

        messages.append({
            "role": "user",
            "content": user_message_content
        })

        logging.debug("Vision prompt messages generated successfully.")
        return messages

    @staticmethod
    def postprocess_answer(answer):
        """
        Post-process the AI response
        
        Args:
            answer: Raw response string
            
        Returns:
            Cleaned response string
        """
        return answer.strip()
