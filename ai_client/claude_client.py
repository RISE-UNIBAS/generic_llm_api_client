"""
Anthropic Claude-specific implementation of the BaseAIClient.

This module provides the ClaudeClient class, which implements the BaseAIClient
interface specifically for Anthropic's Claude API. It currently supports text-only 
interactions with the potential for multimodal support as Claude's API evolves.
"""
import time
from datetime import datetime
from typing import List, Tuple, Any, Optional

from anthropic import Anthropic

from .base_client import BaseAIClient


class ClaudeClient(BaseAIClient):
    """
    Anthropic Claude-specific implementation of the BaseAIClient.
    
    This class implements the BaseAIClient interface for Anthropic's Claude API,
    currently supporting text-only requests with potential for multimodal support
    as Claude's API evolves.
    
    Key features:
    - Integration with Anthropic's Messages API
    - Support for Claude-specific parameters like top_p, top_k, etc.
    - Future-proofed for multimodal capabilities
    """
    
    PROVIDER_ID = "anthropic"
    # Set this to False until Claude multimodal support is implemented
    SUPPORTS_MULTIMODAL = False
    
    def _init_client(self):
        """Initialize the Anthropic client with the provided API key."""
        self.api_client = Anthropic(api_key=self.api_key)
    
    def do_prompt(self, model: str, prompt: str) -> Any:
        """
        Send a prompt to the Claude model and get the response.
        
        Args:
            model (str): The Claude model identifier (e.g., "claude-3-opus-20240229")
            prompt (str): The text prompt to send
            
        Returns:
            tuple: (response_object, elapsed_time_in_seconds)
        """
        # Extract Claude-specific parameters from settings
        params = {
            'model': model,
            'messages': [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            'system': self.system_prompt
        }
        
        # Add optional parameters if specified
        optional_params = [
            'max_tokens', 'temperature', 'top_p', 'top_k'
        ]
        
        for param in optional_params:
            if param in self.settings and self.settings[param] is not None:
                params[param] = self.settings[param]
        
        # Send the request to Anthropic
        response = self.api_client.messages.create(**params)
        return response

    def transpose_response(self, response: Any) -> dict:
        return_value = self.get_empty_generic_response()
        return_value['text'] = response.content[0].text
        return return_value

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from Claude.
        
        Returns:
            list: List of tuples (model_id, created_date)
        """
        if self.api_client is None:
            raise ValueError('Claude client is not initialized.')
        
        model_list = []
        raw_list = self.api_client.models.list()
        
        for model in raw_list:
            readable_date = datetime.fromtimestamp(
                datetime.fromisoformat(str(model.created_at)).timestamp()
            ).strftime('%Y-%m-%d')
            model_list.append((model.id, readable_date))
        
        return model_list