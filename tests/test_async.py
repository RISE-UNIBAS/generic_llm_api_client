"""
Tests for async functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from ai_client import create_ai_client
from ai_client.response import LLMResponse


class TestAsyncFunctionality:
    """Tests for async prompt functionality."""

    @pytest.mark.asyncio
    async def test_prompt_async_basic(self, mock_openai_response):
        """Test basic async prompt."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")
            response = await client.prompt_async("gpt-4", "Hello!")

            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm an AI assistant."
            assert response.duration >= 0

    @pytest.mark.asyncio
    async def test_prompt_async_parallel(self, mock_openai_response):
        """Test parallel async prompts."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")

            # Create multiple async tasks
            tasks = [client.prompt_async("gpt-4", f"Prompt {i}") for i in range(5)]

            # Run them in parallel
            results = await asyncio.gather(*tasks)

            # Check all results
            assert len(results) == 5
            for response in results:
                assert isinstance(response, LLMResponse)
                assert response.duration >= 0

    @pytest.mark.asyncio
    async def test_prompt_async_with_images(self, mock_openai_response, sample_image_path):
        """Test async prompt with images."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")
            response = await client.prompt_async(
                "gpt-4o", "Describe this", images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_prompt_async_with_custom_params(self, mock_openai_response):
        """Test async prompt with custom parameters."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")
            response = await client.prompt_async("gpt-4", "Hello", temperature=0.9, max_tokens=100)

            assert isinstance(response, LLMResponse)

            # Verify parameters were passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.9
            assert call_args.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_prompt_async_error_handling(self):
        """Test async prompt error handling."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            client = create_ai_client("openai", api_key="test-key")
            response = await client.prompt_async("gpt-4", "Hello")

            # Should return error response, not raise
            assert isinstance(response, LLMResponse)
            assert response.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_prompt_async_benchmark_simulation(self, mock_openai_response, sample_image_path):
        """Simulate a benchmark workflow with async processing."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")

            # Simulate processing multiple images in batches
            images = [sample_image_path] * 10

            async def process_image(img_path):
                return await client.prompt_async(
                    "gpt-4o", "Transcribe this image", images=[img_path]
                )

            # Process in batches of 3
            results = []
            for i in range(0, len(images), 3):
                batch = images[i : i + 3]
                batch_tasks = [process_image(img) for img in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)

            assert len(results) == 10
            for response in results:
                assert isinstance(response, LLMResponse)
