"""
Mistral-specific implementation of the BaseAIClient.

This module provides the MistralClient class, which implements the BaseAIClient
interface specifically for Mistral's API. It currently supports text-only
interactions with the Mistral chat completion API.
"""
import time
from datetime import datetime, timezone
from typing import List, Tuple, Any, Optional

from mistralai import Mistral

from .base_client import BaseAIClient


class MistralClient(BaseAIClient):
    """
    Mistral-specific implementation of the BaseAIClient.
    
    This class implements the BaseAIClient interface for Mistral's API,
    currently supporting text-only requests via the chat completion API.
    
    Key features:
    - Integration with Mistral's chat completion API
    - Support for Mistral-specific parameters like random_seed, presence_penalty, etc.
    """
    
    PROVIDER_ID = "mistral"
    SUPPORTS_MULTIMODAL = False
    
    def _init_client(self):
        """Initialize the Mistral client with the provided API key."""
        self.api_client = Mistral(api_key=self.api_key)
    
    def do_prompt(self, model: str, prompt: str) -> Any:
        """
        Send a prompt to the Mistral model and get the response.
        
        Args:
            model (str): The Mistral model identifier (e.g., "mistral-medium", "mistral-small")
            prompt (str): The text prompt to send
            
        Returns:
            tuple: (response_object, elapsed_time_in_seconds)
        """
        
        # Prepare the base message structure
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Prepend system message if allowed by this Mistral model version
        # Some older Mistral models don't support system messages
        if model not in ['mistral-tiny', 'mistral-small-2402']:
            messages.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })
        
        # Extract Mistral-specific parameters from settings
        params = {
            'model': model,
            'messages': messages
        }
        
        # Add optional parameters if specified
        optional_params = [
            'temperature', 'max_tokens', 'random_seed', 'presence_penalty'
        ]
        
        for param in optional_params:
            if param in self.settings and self.settings[param] is not None:
                # Handle terminology differences
                if param == 'random_seed':
                    params['random_seed'] = self.settings[param]
                elif param == 'presence_penalty':
                    params['presence_penalty'] = self.settings[param]
                else:
                    params[param] = self.settings[param]
        
        # Send the request to Mistral
        response = self.api_client.chat.complete(**params)

        return response

    def transpose_response(self, response: Any) -> dict:
        return_value = self.get_empty_generic_response()
        return_value['text'] = response.content[0].text
        return return_value

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from Mistral.
        
        Returns:
            list: List of tuples (model_id, created_date)
        """
        if self.api_client is None:
            raise ValueError('Mistral client is not initialized.')
        
        model_list = []
        raw_list = self.api_client.models.list()
        
        for model in raw_list.data:
            # Only include models that support chat completion
            completion = getattr(model.capabilities, 'completion_chat', None)
            if completion:
                readable_date = datetime.fromtimestamp(
                    model.created, tz=timezone.utc
                ).strftime('%Y-%m-%d')
                model_list.append((model.id, readable_date))
        
        return model_list