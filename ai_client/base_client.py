"""
Abstract base AI client defining the interface for all AI provider implementations.

This module provides an abstract base class that defines the common interface
for all AI provider client implementations. It handles the shared functionality
like timing, image resource management, and basic client lifecycle, while 
leaving provider-specific implementations to subclasses.
"""
import abc
import time
from typing import List, Tuple, Any, Optional


class BaseAIClient(abc.ABC):
    """
    Abstract base AI client defining the interface for all provider implementations.
    
    This abstract class defines the common methods and properties that all AI
    provider client implementations must support. It handles basic functionality
    like timing, client lifecycle, and image resource management, while leaving
    provider-specific implementations to subclasses.
    
    Attributes:
        PROVIDER_ID (str): The unique identifier for this AI provider
        SUPPORTS_MULTIMODAL (bool): Whether this provider supports multimodal content
        image_resources (list): List of image paths or URLs to include in prompts
        init_time (float): Time when the client was initialized
        end_time (float): Time when the client session ended
        api_key (str): The API key used for authentication
        settings (dict): Provider-specific settings like temperature, max_tokens, etc.
    """
    
    PROVIDER_ID = "base"
    SUPPORTS_MULTIMODAL = False
    
    def __init__(self, api_key: str, system_prompt: Optional[str] = None, **settings):
        """
        Initialize the AI client base.
        
        Args:
            api_key (str): API key for authentication with the provider
            system_prompt (str, optional): System prompt/role description for the AI
            **settings: Additional provider-specific settings like temperature, max_tokens, etc.
        """
        self.init_time = time.time()
        self.end_time = None
        self.api_key = api_key
        self.system_prompt = system_prompt or "A helpful assistant that provides accurate information."
        self.settings = settings
        self.image_resources: List[str] = []
        self.api_client = None
        
        # Initialize the client implementation
        self._init_client()
    
    @abc.abstractmethod
    def _init_client(self):
        """
        Initialize the provider-specific client.
        
        This method must be implemented by each provider to handle
        their specific client initialization requirements.
        """
        pass
    
    @property
    def elapsed_time(self) -> float:
        """
        Get the elapsed time since client initialization.
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.end_time is None:
            return time.time() - self.init_time
        return self.end_time - self.init_time
    
    def end_client(self):
        """
        End the client session and record the end time.
        
        This method cleans up the client object and records when the session ended.
        The client is considered inactive after this method is called.
        """
        self.api_client = None
        self.end_time = time.time()
    
    def add_image_resource(self, resource: str):
        """
        Add an image resource to be included in the next prompt.
        
        Args:
            resource (str): Either a local file path or a URL to an image
        """
        if not self.SUPPORTS_MULTIMODAL:
            # Silently accept but won't use if provider doesn't support images
            pass
        self.image_resources.append(resource)
    
    def clear_image_resources(self):
        """
        Clear all registered image resources.
        
        Call this after sending a prompt to clean up resources and 
        prevent unintentional inclusion in subsequent requests.
        """
        self.image_resources = []
    
    def is_url(self, resource: str) -> bool:
        """
        Check if a resource is a URL or a local file path.
        
        Args:
            resource (str): The resource string to check
            
        Returns:
            bool: True if the resource is a URL, False if it's a file path
        """
        return resource.startswith(('http://', 'https://'))

    def get_empty_generic_response(self) -> dict:
        """
        Convert the raw response from the provider into a generic format.

        Args:
            response (Any): The raw response from the provider

        Returns:
            dict: The transposed response in a generic format
        """
        ret_val = {
            "text": '',
            "json": {},
            "model": '',
            "duration": ''
        }
        return ret_val

    def prompt(self, model: str, prompt: str) -> Tuple[dict, float]:
        """
        Send a prompt to the AI model and get the response.
        
        Args:
            model (str): The model identifier to use
            prompt (str): The text prompt to send to the model
            
        Returns:
            tuple: (response_object, elapsed_time_in_seconds)
        """
        start_time = time.time()
        response = self.do_prompt(model, prompt)
        generic_response = self.transpose_response(response)
        elapsed_time = time.time() - start_time
        return generic_response, elapsed_time

    @abc.abstractmethod
    def do_prompt(self, model: str, prompt: str) -> Any:
        pass

    @abc.abstractmethod
    def transpose_response(self, response: Any) -> dict:
        """
        Convert the raw response from the provider into a generic format.

        Args:
            response (Any): The raw response from the provider

        Returns:
            Any: The transposed response in a generic format
        """
        pass

    @abc.abstractmethod
    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from the current provider.
        
        Returns:
            list: List of tuples containing (model_id, created_date)
                 The created_date may be None for some providers.
        """
        pass

    def has_multimodal_support(self) -> bool:
        """
        Check if the provider supports multimodal content.

        Returns:
            bool: True if the provider supports multimodal content, False otherwise
        """
        return self.SUPPORTS_MULTIMODAL

    def __str__(self):
        """
        String representation of the AI client.

        Returns:
            str: The string representation of the client
        """
        return self.PROVIDER_ID

def create_ai_client(provider: str, api_key: str, system_prompt: str = None, **settings) -> BaseAIClient:
    """
    Factory function to create an appropriate AI client for the given provider.
    
    Args:
        provider (str): The AI provider ID ('openai', 'gemini', 'claude', or 'mistral')
        api_key (str): API key for the provider
        system_prompt (str, optional): System prompt/role description
        **settings: Additional provider-specific settings
        
    Returns:
        BaseAIClient: An instance of the appropriate AI client implementation
        
    Raises:
        ValueError: If an unsupported provider is specified
    """
    from .openai_client import OpenAIClient
    from .gemini_client import GeminiClient
    from .claude_client import ClaudeClient
    from .mistral_client import MistralClient
    from .deepseek_client import DeepSeekClient
    from .qwen_client import QwenClient
    
    provider_map = {
        'openai': OpenAIClient,
        'genai': GeminiClient,
        'anthropic': ClaudeClient,
        'mistral': MistralClient,
        'deepseek': DeepSeekClient,
        'qwen': QwenClient
    }
    
    if provider not in provider_map:
        raise ValueError(f"Unsupported AI provider: {provider}. " +
                         f"Supported providers: {', '.join(provider_map.keys())}")
    
    client_class = provider_map[provider]
    return client_class(api_key, system_prompt, **settings)