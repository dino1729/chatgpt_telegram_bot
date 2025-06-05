"""
Image generation and processing utilities
"""
import base64
import json
import os
import subprocess
import tempfile
from io import BytesIO
from openai import OpenAI, AzureOpenAI as OpenAIAzure

class ImageUtils:
    """Utilities for image generation and processing"""
    
    def __init__(self, config_manager):
        self.config = config_manager

async def generate_images(prompt, n_images=4, config_manager=None):
    """
    Generate images using Azure OpenAI DALL-E
    
    Args:
        prompt: Text prompt for image generation
        n_images: Number of images to generate
        config_manager: Configuration manager instance
        
    Returns:
        List of image URLs
    """
    # Import config_manager if not provided
    if config_manager is None:
        try:
            from .config_manager import get_config_manager
            config_manager = get_config_manager()
        except ImportError:
            try:
                from config_manager import get_config_manager
                config_manager = get_config_manager()
            except ImportError:
                raise RuntimeError("Config manager not available - unable to import")
    
    if config_manager is None:
        raise RuntimeError("Config manager not available")
    
    image_client = OpenAIAzure(
        api_key=config_manager.azure_api_key,
        azure_endpoint=config_manager.azure_api_base,
        api_version="2023-12-01-preview"
    )
    
    response = image_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=n_images,
        size="1024x1024",
    )
    
    image_urls = [item.url for item in response.data]
    return image_urls

async def generate_images_gpt_image_1(prompt, n_images=1, size="1024x1024", config_manager=None):
    """
    Generate images using OpenAI's latest GPT-Image-1 model
    
    Args:
        prompt: Text prompt for image generation
        n_images: Number of images to generate (GPT-Image-1 typically supports 1)
        size: Image size ("1024x1024", "1792x1024", "1024x1792")
        config_manager: Configuration manager instance
        
    Returns:
        List of image URLs (as data URLs with base64)
    """
    import os
    
    # Import config_manager if not provided
    if config_manager is None:
        try:
            from .config_manager import get_config_manager
            config_manager = get_config_manager()
        except ImportError:
            try:
                from config_manager import get_config_manager
                config_manager = get_config_manager()
            except ImportError:
                raise RuntimeError("Config manager not available - unable to import")
    
    if config_manager is None:
        raise RuntimeError("Config manager not available")
    
    # Use regular OpenAI client for GPT-Image-1 generation (not Azure)
    api_key = config_manager.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = getattr(config_manager, 'openai_api_base', None) or os.getenv("OPENAI_API_BASE")
    
    image_client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        # Generate image using GPT-Image-1 model
        response = image_client.images.generate(
            model="gpt-image-1",  # Use the new GPT-Image-1 model
            prompt=prompt,
            n=1,  # GPT-Image-1 typically generates 1 image at a time
            size=size
        )
        
        # Extract base64 data and convert to data URLs
        image_urls = []
        for item in response.data:
            # Get base64 data from response
            image_bytes = base64.b64decode(item.b64_json)
            # Convert to data URL
            b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            data_url = f"data:image/png;base64,{b64_encoded}"
            image_urls.append(data_url)
        
        return image_urls
        
    except Exception as e:
        # Handle errors
        if "model" in str(e).lower() and ("not found" in str(e).lower() or "not available" in str(e).lower()):
            raise RuntimeError("GPT-Image-1 model not available")
        else:
            raise

async def edit_image_gpt_image_1(image, prompt, mask=None, n_images=1, size="1024x1024", config_manager=None):
    """
    Edit images using OpenAI's GPT-Image-1 model (not Azure - using regular OpenAI API)
    
    Args:
        image: Image file (BytesIO buffer or file path)
        prompt: Text prompt describing the desired edit
        mask: Optional mask image for inpainting (not used in this implementation)
        n_images: Number of edited images to generate
        size: Image size
        config_manager: Configuration manager instance
        
    Returns:
        List of edited image URLs (as data URLs with base64)
    """
    import os
    from PIL import Image as PILImage
    
    def _ensure_png(path: str) -> str:
        """Ensure the file is a PNG, converting if necessary"""
        if path.lower().endswith(".png"):
            return path
        tmp_path = tempfile.mktemp(suffix=".png")
        PILImage.open(path).save(tmp_path, format="PNG")
        return tmp_path
    
    # Import config_manager if not provided
    if config_manager is None:
        try:
            from .config_manager import get_config_manager
            config_manager = get_config_manager()
        except ImportError:
            try:
                from config_manager import get_config_manager
                config_manager = get_config_manager()
            except ImportError:
                raise RuntimeError("Config manager not available - unable to import")
    
    if config_manager is None:
        raise RuntimeError("Config manager not available")
    
    # Use regular OpenAI client for GPT-Image-1 editing (not Azure)
    api_key = config_manager.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = getattr(config_manager, 'openai_api_base', None) or os.getenv("OPENAI_API_BASE")
    
    if not api_key:
        raise RuntimeError("Missing OpenAI API key for GPT-Image-1 editing")
    
    # Handle image input - save to temporary file if it's a BytesIO buffer
    temp_image_path = None
    original_temp_path = None
    try:
        if isinstance(image, BytesIO):
            # Save BytesIO to temporary PNG file
            original_temp_path = tempfile.mktemp(suffix=".png")
            image.seek(0)
            with open(original_temp_path, "wb") as f:
                f.write(image.read())
            image_path = original_temp_path
        elif isinstance(image, str):
            image_path = image
        else:
            raise ValueError("Image must be either a file path or BytesIO buffer")
        
        # Ensure image is PNG format (required by API)
        png_image_path = _ensure_png(image_path)
        if png_image_path != image_path:
            temp_image_path = png_image_path  # Track the converted PNG for cleanup
        
        # Use OpenAI client for GPT-Image-1 editing
        image_client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Open the image file
        with open(png_image_path, "rb") as img_file:
            response = image_client.images.edit(
                model="gpt-image-1",  # Use the GPT-Image-1 model
                image=img_file,
                prompt=prompt,
                n=1,  # GPT-Image-1 typically generates 1 image at a time
                size=size
            )
        
        # Extract base64 data and convert to data URLs
        image_urls = []
        for item in response.data:
            # Get base64 data from response
            if hasattr(item, 'b64_json') and item.b64_json:
                data_url = f"data:image/png;base64,{item.b64_json}"
                image_urls.append(data_url)
            elif hasattr(item, 'url') and item.url:
                # If URL is provided instead of base64, we'll need to download it
                # For now, just use the URL directly
                image_urls.append(item.url)
        
        return image_urls
        
    except Exception as e:
        # Handle any API errors
        if "model" in str(e).lower() and ("not found" in str(e).lower() or "not available" in str(e).lower()):
            raise RuntimeError("GPT-Image-1 model not available for image editing")
        else:
            raise
    finally:
        # Clean up temporary files
        if original_temp_path and os.path.exists(original_temp_path):
            os.unlink(original_temp_path)
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

async def is_content_acceptable(prompt, config_manager=None):
    """
    Check if content is acceptable using OpenAI moderation
    
    Args:
        prompt: Content to check
        config_manager: Configuration manager instance
        
    Returns:
        Boolean indicating if content is acceptable
    """
    # Import config_manager if not provided
    if config_manager is None:
        try:
            from .config_manager import get_config_manager
            config_manager = get_config_manager()
        except ImportError:
            try:
                from config_manager import get_config_manager
                config_manager = get_config_manager()
            except ImportError:
                return True  # Default to acceptable if config not available
    
    if config_manager is None:
        return True  # Default to acceptable if config not available
    
    # Note: This function needs to be updated as the original implementation
    # uses deprecated API. Here's a placeholder for the corrected version.
    # The actual implementation would depend on the current OpenAI moderation API
    try:
        client = OpenAIAzure(
            api_key=config_manager.azure_api_key,
            azure_endpoint=config_manager.azure_api_base,
            api_version=config_manager.azure_chatapi_version,
        )
        # This would need to be updated to use the current moderation endpoint
        # r = client.moderations.create(input=prompt)
        # return not any(r.results[0].categories.values())
        return True  # Placeholder - implement proper moderation check
    except Exception:
        return True  # Default to acceptable if check fails
