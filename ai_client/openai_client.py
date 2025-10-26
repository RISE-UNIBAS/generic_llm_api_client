"""
OpenAI-specific implementation of the BaseAIClient.

This module provides the OpenAIClient class, which implements the BaseAIClient
interface specifically for OpenAI's API, supporting both text-only and
multimodal content (text + images).
"""
import base64
import time
from datetime import datetime, timezone
from typing import List, Tuple, Any, Optional

from openai import OpenAI

from .base_client import BaseAIClient


class OpenAIClient(BaseAIClient):
    """
    OpenAI-specific implementation of the BaseAIClient.
    
    This class implements the BaseAIClient interface for OpenAI's API,
    handling both text-only and multimodal (text + images) requests through
    the OpenAI Chat Completions API.
    
    Key features:
    - Full support for multimodal content via GPT-4 Vision models
    - Image encoding and formatting specific to OpenAI's requirements
    - Support for OpenAI-specific parameters like frequency_penalty, top_p, etc.
    """
    
    PROVIDER_ID = "openai"
    SUPPORTS_MULTIMODAL = True
    
    def _init_client(self):
        """Initialize the OpenAI client with the provided API key."""
        self.api_client = OpenAI(api_key=self.api_key)
    
    def _prepare_message_with_images(self, prompt: str) -> List[dict]:
        """
        Prepare an OpenAI message object with text and images.
        
        Args:
            prompt (str): The text prompt
            
        Returns:
            list: List of OpenAI message objects with content
        """
        # Prepare user content with text
        user_content = [{"type": "text", "text": prompt}]
        
        # Add images if any
        for resource in self.image_resources:
            if self.is_url(resource):
                # For URLs, directly use the URL in the prompt
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": resource}
                })
            else:
                # For local files, read and encode as base64
                try:
                    with open(resource, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    # Specify resize if included in settings
                    image_url_dict = {"url": f"data:image/jpeg;base64,{base64_image}"}
                    if 'image_resize' in self.settings and self.settings['image_resize'] not in ('', 'none'):
                        # Parse resize setting like '512x512'
                        try:
                            width, height = self.settings['image_resize'].split('x')
                            image_url_dict["detail"] = {"width": int(width), "height": int(height)}
                        except (ValueError, TypeError):
                            # If parsing fails, don't include resize details
                            pass
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": image_url_dict
                    })
                except Exception as e:
                    print(f"Error reading image file {resource}: {e}")
        
        # Return the complete message structure
        return [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
    
    def do_prompt(self, model: str, prompt: str) -> Any:
        """
        Send a prompt to the OpenAI model and get the response.
        
        Args:
            model (str): The OpenAI model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            prompt (str): The text prompt to send
            
        Returns:
            tuple: (response_object, elapsed_time_in_seconds)
        """
        
        # Prepare messages with any images
        messages = self._prepare_message_with_images(prompt)
        
        # Extract OpenAI-specific parameters from settings
        params = {
            'model': model,
            'messages': messages,
            'temperature': self.settings.get('temperature', 0.5),
        }
        
        # Add optional parameters if specified
        optional_params = [
            'max_tokens', 'top_p', 'frequency_penalty', 
            'presence_penalty', 'seed'
        ]
        for param in optional_params:
            if param in self.settings and self.settings[param] is not None:
                params[param] = self.settings[param]
        
        # Send the request to OpenAI
        response = self.api_client.chat.completions.create(**params)

        return response

    def transpose_response(self, response: Any) -> dict:
        return_value = self.get_empty_generic_response()
        return_value['text'] = response.choices[0].message.content
        return return_value

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from OpenAI.
        
        Returns:
            list: List of tuples (model_id, created_date)
        """
        if self.api_client is None:
            raise ValueError('OpenAI client is not initialized.')
        
        raw_list = self.api_client.models.list()
        model_list = []
        
        for model in raw_list:
            readable_date = datetime.fromtimestamp(model.created, tz=timezone.utc).strftime('%Y-%m-%d')
            model_list.append((model.id, readable_date))
        
        return model_list