"""
Abstract base AI client defining the interface for all AI provider implementations.

This module provides an abstract base class that defines the common interface
for all AI provider client implementations. It handles shared functionality
like timing and basic client lifecycle, while leaving provider-specific
implementations to subclasses.
"""

import abc
import time
import asyncio
from typing import List, Tuple, Any, Optional
from .response import LLMResponse, Usage
from .utils import (
    retry_with_exponential_backoff,
    read_text_files,
    resize_image_if_needed,
)


class BaseAIClient(abc.ABC):
    """
    Abstract base AI client defining the interface for all provider implementations.

    This abstract class defines the common methods and properties that all AI
    provider client implementations must support. It handles basic functionality
    like timing, retry logic, async support, and client lifecycle.

    Key features:
    - Images are passed per-request, avoiding stateful management bugs
    - Returns structured LLMResponse objects with detailed usage info
    - Optional async support for parallel processing
    - Built-in retry logic for rate limiting

    Attributes:
        PROVIDER_ID (str): The unique identifier for this AI provider
        SUPPORTS_MULTIMODAL (bool): Whether this provider supports multimodal content
        api_key (str): The API key used for authentication
        system_prompt (str): Default system prompt for requests
        settings (dict): Provider-specific settings like temperature, max_tokens, etc.
        base_url (str, optional): Custom base URL for API (e.g., for OpenRouter, sciCORE)
        max_image_size (int, optional): Maximum image dimension in pixels (None = no resize)
        image_quality (int): JPEG quality for resized images (1-100, default 85)
    """

    PROVIDER_ID = "base"
    SUPPORTS_MULTIMODAL = False

    def __init__(
        self,
        api_key: str,
        system_prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        max_image_size: Optional[int] = 2048,
        image_quality: int = 85,
        **settings,
    ):
        """
        Initialize the AI client base.

        Args:
            api_key: API key for authentication with the provider
            system_prompt: System prompt/role description for the AI
            base_url: Custom base URL for API endpoints (provider-specific)
            max_image_size: Maximum image dimension in pixels (None to disable resize, default 2048)
            image_quality: JPEG quality for resized images (1-100, default 85)
            **settings: Additional provider-specific settings like temperature, max_tokens, etc.
        """
        self.init_time = time.time()
        self.end_time = None
        self.api_key = api_key
        self.system_prompt = (
            system_prompt or "A helpful assistant that provides accurate information."
        )
        self.base_url = base_url
        self.max_image_size = max_image_size
        self.image_quality = image_quality
        self.settings = settings
        self.api_client = None

        # Track conversation state for multi-turn requests
        self.conversations = {}  # Format: {conv_id: {"messages": [...], "cache_ref": None}}

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
            Elapsed time in seconds
        """
        if self.end_time is None:
            return time.time() - self.init_time
        return self.end_time - self.init_time

    def end_client(self):
        """
        End the client session and record the end time.

        This method cleans up the client object and records when the session ended.
        """
        self.api_client = None
        self.end_time = time.time()

    @staticmethod
    def is_url(resource: str) -> bool:
        """
        Check if a resource is a URL or a local file path.

        Args:
            resource: The resource string to check

        Returns:
            True if the resource is a URL, False if it's a file path
        """
        return resource.startswith(("http://", "https://"))

    def prompt(
        self,
        model: str,
        prompt: str,
        images: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Any] = None,
        conversation_id: Optional[str] = None,
        cache: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a prompt to the AI model and get the response.

        Args:
            model: The model identifier to use
            prompt: The text prompt to send to the model
            images: Optional list of image paths/URLs to include (if provider supports multimodal)
            files: Optional list of text file paths to include in the prompt
            system_prompt: Optional system prompt to override the default
            response_format: Optional Pydantic model for structured output
            conversation_id: Continue an existing conversation by ID (generates new ID if None)
            cache: Enable caching for large prompts/files (provider-specific implementation)
            **kwargs: Additional provider-specific parameters
                Common: temperature, max_tokens, top_p
                OpenAI: prompt_cache_key, prompt_cache_retention
                Gemini: cache_id, ttl

        Returns:
            LLMResponse object with response and usage information

        Note:
            Images and files are NOT stored statefully. Pass them with each request.
            Images will be automatically resized if max_image_size is set.

            Caching behavior (when cache=True):
            - OpenAI: Automatic for 1024+ tokens (always active)
            - Claude: Adds cache_control blocks to file content
            - Gemini: Uses cache_id from kwargs if provided
        """
        start_time = time.time()

        # Use provided system prompt or fall back to default
        sys_prompt = system_prompt or self.system_prompt

        # Handle text files
        file_list = files or []
        file_content = ""
        if file_list:
            file_content = read_text_files(file_list)
            # For caching mode with Claude, keep content separate
            # Other providers append files to prompt
            if not (cache and self.PROVIDER_ID == "anthropic"):
                # Append to prompt as usual (backward-compatible)
                prompt = prompt + file_content

        # Handle conversation continuation
        messages = []
        existing_cache_ref = None

        if conversation_id and conversation_id in self.conversations:
            # Load existing conversation history
            conv_data = self.conversations[conversation_id]
            messages = conv_data["messages"].copy()
            existing_cache_ref = conv_data.get("cache_ref")

        # Add current prompt as user message
        messages.append({"role": "user", "content": prompt})

        # Handle multimodal content
        image_list = images or []
        if image_list and not self.SUPPORTS_MULTIMODAL:
            # Silently ignore images if provider doesn't support them
            image_list = []

        # Resize images if needed
        if image_list and self.max_image_size:
            image_list = [
                resize_image_if_needed(img, self.max_image_size, self.image_quality)
                for img in image_list
            ]

        # Call provider-specific implementation with retry logic
        try:
            response = self._do_prompt_with_retry(
                model=model,
                prompt=prompt,
                messages=messages if len(messages) > 1 else None,
                images=image_list,
                system_prompt=sys_prompt,
                response_format=response_format,
                cache=cache,
                file_content=file_content,
                **kwargs,
            )

            # Create or update conversation ID
            if conversation_id is None and len(messages) > 0:
                # Create new conversation ID
                import uuid

                conversation_id = str(uuid.uuid4())

            if conversation_id:
                # Store conversation history
                messages.append({"role": "assistant", "content": response.text})
                self.conversations[conversation_id] = {
                    "messages": messages,
                    "cache_ref": getattr(response, "cache_ref", existing_cache_ref),
                }
                response.conversation_id = conversation_id

        except Exception as e:
            # Create error response
            response = self._create_error_response(model, str(e))

        elapsed_time = time.time() - start_time
        response.duration = elapsed_time

        return response

    def _do_prompt_with_retry(self, **kwargs) -> LLMResponse:
        """
        Execute prompt with retry logic for rate limiting.

        This wraps the provider-specific _do_prompt method with retry logic.
        """
        # Configure retry for common rate limit errors
        retry_func = retry_with_exponential_backoff(
            self._do_prompt,
            max_retries=3,
            initial_delay=1.0,
            max_delay=60.0,
            retryable_exceptions=(Exception,),  # Provider-specific exceptions will be caught
        )

        return retry_func(**kwargs)

    @abc.abstractmethod
    def _do_prompt(
        self,
        model: str,
        prompt: str,
        messages: Optional[List[dict]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Any] = None,
        cache: bool = False,
        file_content: str = "",
        **kwargs,
    ) -> LLMResponse:
        """
        Provider-specific prompt implementation.

        Args:
            model: Model identifier
            prompt: Text prompt
            messages: Optional conversation history (None for single-turn)
            images: List of image paths/URLs
            system_prompt: System prompt to use
            response_format: Optional Pydantic model for structured output
            cache: Enable caching (provider interprets this flag)
            file_content: Content from files (for caching if cache=True)
            **kwargs: Provider-specific parameters
                OpenAI: prompt_cache_key, prompt_cache_retention
                Gemini: cache_id, ttl

        Returns:
            LLMResponse object with the provider's response
        """
        pass

    async def prompt_async(
        self,
        model: str,
        prompt: str,
        images: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Any] = None,
        conversation_id: Optional[str] = None,
        cache: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Async version of prompt() for parallel processing.

        Args:
            model: The model identifier to use
            prompt: The text prompt to send
            images: Optional list of image paths/URLs
            files: Optional list of text file paths to include
            system_prompt: Optional system prompt override
            response_format: Optional Pydantic model for structured output
            conversation_id: Continue an existing conversation by ID
            cache: Enable caching (provider-specific)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object with response and usage information
        """
        # Default implementation runs sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.prompt(
                model=model,
                prompt=prompt,
                images=images,
                files=files,
                system_prompt=system_prompt,
                response_format=response_format,
                conversation_id=conversation_id,
                cache=cache,
                **kwargs,
            ),
        )

    def _create_error_response(self, model: str, error_message: str) -> LLMResponse:
        """
        Create an error response when the request fails.

        Args:
            model: Model identifier
            error_message: Error message

        Returns:
            LLMResponse with error information
        """
        return LLMResponse(
            text="",
            model=model,
            provider=self.PROVIDER_ID,
            finish_reason="error",
            usage=Usage(),
            raw_response={"error": error_message},
            duration=0.0,
        )

    @abc.abstractmethod
    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from the current provider.

        Returns:
            List of tuples containing (model_id, created_date)
            The created_date may be None for some providers.
        """
        pass

    def has_multimodal_support(self) -> bool:
        """
        Check if the provider supports multimodal content.

        Returns:
            True if the provider supports multimodal content, False otherwise
        """
        return self.SUPPORTS_MULTIMODAL

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear a specific conversation from memory.

        Args:
            conversation_id: The conversation ID to clear
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def clear_all_conversations(self) -> None:
        """Clear all conversations from memory."""
        self.conversations.clear()

    def get_conversation_history(self, conversation_id: str) -> Optional[List[dict]]:
        """
        Get the message history for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            List of message dictionaries, or None if conversation doesn't exist
        """
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]["messages"].copy()
        return None

    def __str__(self):
        """String representation of the AI client."""
        return self.PROVIDER_ID


def create_ai_client(
    provider: str,
    api_key: str,
    system_prompt: Optional[str] = None,
    base_url: Optional[str] = None,
    max_image_size: Optional[int] = 2048,
    image_quality: int = 85,
    **settings,
) -> BaseAIClient:
    """
    Factory function to create an appropriate AI client for the given provider.

    Args:
        provider: The AI provider ID ('openai', 'genai', 'anthropic', 'mistral', etc.)
        api_key: API key for the provider
        system_prompt: System prompt/role description
        base_url: Custom base URL for API (for OpenRouter, sciCORE, etc.)
        max_image_size: Maximum image dimension in pixels (None to disable, default 2048)
        image_quality: JPEG quality for resized images (1-100, default 85)
        **settings: Additional provider-specific settings

    Returns:
        BaseAIClient: An instance of the appropriate AI client implementation

    Raises:
        ValueError: If an unsupported provider is specified

    Examples:
        >>> # Standard usage
        >>> client = create_ai_client('openai', api_key='sk-...')
        >>> response, duration = client.prompt('gpt-4', 'Hello!')

        >>> # With image auto-resize
        >>> client = create_ai_client('openai', api_key='sk-...', max_image_size=1024)
        >>> response, duration = client.prompt('gpt-4o', 'Describe',
        ...                                    images=['huge_image.jpg'])  # Auto-resized

        >>> # With text files
        >>> response, duration = client.prompt('gpt-4', 'Analyze these documents',
        ...                                    files=['doc1.txt', 'doc2.txt'])

        >>> # With both images and files
        >>> response, duration = client.prompt('gpt-4o', 'Compare image to description',
        ...                                    images=['photo.jpg'],
        ...                                    files=['description.txt'])
    """
    from .openai_client import OpenAIClient
    from .gemini_client import GeminiClient
    from .claude_client import ClaudeClient
    from .mistral_client import MistralClient
    from .deepseek_client import DeepSeekClient
    from .qwen_client import QwenClient
    from .cohere_client import CohereClient

    provider_map = {
        "openai": OpenAIClient,
        "genai": GeminiClient,
        "google": GeminiClient,
        "anthropic": ClaudeClient,
        "mistral": MistralClient,
        "deepseek": DeepSeekClient,
        "qwen": QwenClient,
        "cohere": CohereClient,
        "openrouter": OpenAIClient,  # Uses OpenAI-compatible API
        "scicore": OpenAIClient,  # Uses OpenAI-compatible API
    }

    if provider not in provider_map:
        raise ValueError(
            f"Unsupported AI provider: {provider}. "
            f"Supported providers: {', '.join(provider_map.keys())}"
        )

    # Set default base_url for providers that need it (BEFORE client instantiation)
    if provider == "openrouter" and base_url is None:
        base_url = "https://openrouter.ai/api/v1"
    elif provider == "scicore" and base_url is None:
        base_url = "https://llm-api-h200.ceda.unibas.ch/litellm"

    client_class = provider_map[provider]
    return client_class(api_key, system_prompt, base_url, max_image_size, image_quality, **settings)
