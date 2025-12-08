"""
Tests for prompt caching functionality across providers.

This test suite validates the caching implementation for OpenAI, Claude, and Gemini.
"""

import pytest
from ai_client import create_ai_client
from ai_client.response import Usage, LLMResponse


class TestUsageDataclass:
    """Test the Usage dataclass with caching fields."""

    def test_usage_with_no_caching(self):
        """Test usage object with no caching."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_tokens is None
        assert usage.cache_creation_tokens == 0
        assert usage.cache_read_tokens == 0

    def test_usage_with_openai_caching(self):
        """Test usage object with OpenAI-style caching."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=80,
        )

        assert usage.cached_tokens == 80
        assert usage.get_cache_savings() == pytest.approx(0.8)  # 80/100

    def test_usage_with_claude_caching(self):
        """Test usage object with Claude-style caching."""
        usage = Usage(
            input_tokens=20,  # Tokens after cache breakpoint
            output_tokens=50,
            cache_creation_tokens=100,  # Tokens written to cache
            cache_read_tokens=100,  # Tokens read from cache
        )

        assert usage.cache_creation_tokens == 100
        assert usage.cache_read_tokens == 100
        assert usage.get_total_input_tokens() == 220  # 20 + 100 + 100
        assert usage.get_cache_savings() == pytest.approx(100 / 220)  # Only read tokens are savings

    def test_usage_to_dict(self):
        """Test usage serialization."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=80,
            cache_creation_tokens=10,
            cache_read_tokens=5,
        )

        usage_dict = usage.to_dict()

        assert usage_dict["input_tokens"] == 100
        assert usage_dict["output_tokens"] == 50
        assert usage_dict["cached_tokens"] == 80
        assert usage_dict["cache_creation_tokens"] == 10
        assert usage_dict["cache_read_tokens"] == 5


class TestLLMResponseDataclass:
    """Test the LLMResponse dataclass with conversation tracking."""

    def test_response_with_conversation_id(self):
        """Test response object with conversation ID."""
        response = LLMResponse(
            text="Hello",
            model="gpt-4o",
            provider="openai",
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5),
            raw_response={},
            conversation_id="conv_123",
        )

        assert response.conversation_id == "conv_123"
        assert response.cache_ref is None

    def test_response_with_cache_ref(self):
        """Test response object with cache reference (Gemini)."""
        response = LLMResponse(
            text="Hello",
            model="gemini-2.0-flash",
            provider="genai",
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5),
            raw_response={},
            cache_ref="cache_abc",
        )

        assert response.cache_ref == "cache_abc"
        assert response.conversation_id is None

    def test_response_to_dict_with_caching(self):
        """Test response serialization with caching fields."""
        response = LLMResponse(
            text="Hello",
            model="claude-3-sonnet",
            provider="anthropic",
            finish_reason="stop",
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=100,
            ),
            raw_response={},
            conversation_id="conv_456",
        )

        response_dict = response.to_dict()

        assert response_dict["conversation_id"] == "conv_456"
        assert response_dict["usage"]["cache_read_tokens"] == 100


class TestConversationTracking:
    """Test conversation tracking functionality."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create a mock client for testing."""
        # Mock the OpenAI client to avoid actual API calls
        mocker.patch("openai.OpenAI")
        client = create_ai_client("openai", api_key="test_key")

        # Mock the _do_prompt method to return test responses
        def mock_do_prompt(*args, **kwargs):
            messages = kwargs.get("messages")

            # Simulate responses based on conversation history
            if messages and len(messages) > 1:
                text = f"Response to message {len(messages)}"
            else:
                text = "First response"

            return LLMResponse(
                text=text,
                model="gpt-4o",
                provider="openai",
                finish_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5),
                raw_response={},
            )

        mocker.patch.object(client, "_do_prompt", side_effect=mock_do_prompt)
        return client

    def test_conversation_id_generation(self, mock_client):
        """Test that conversation IDs are generated automatically."""
        response = mock_client.prompt(
            model="gpt-4o",
            prompt="Hello",
        )

        assert response.conversation_id is not None
        assert isinstance(response.conversation_id, str)
        assert len(response.conversation_id) > 0

    def test_conversation_continuation(self, mock_client):
        """Test that conversations can be continued."""
        # First message
        response1 = mock_client.prompt(
            model="gpt-4o",
            prompt="What is 2+2?",
        )
        conv_id = response1.conversation_id

        # Continue conversation
        response2 = mock_client.prompt(
            model="gpt-4o",
            prompt="What about 3+3?",
            conversation_id=conv_id,
        )

        assert response2.conversation_id == conv_id
        # After first call: 2 messages stored (user + assistant)
        # Second call loads 2 + adds new user message = 3 messages total
        assert "message 3" in response2.text.lower()

    def test_conversation_history_storage(self, mock_client):
        """Test that conversation history is stored correctly."""
        response1 = mock_client.prompt(
            model="gpt-4o",
            prompt="First message",
        )
        conv_id = response1.conversation_id

        mock_client.prompt(
            model="gpt-4o",
            prompt="Second message",
            conversation_id=conv_id,
        )

        # Get conversation history
        history = mock_client.get_conversation_history(conv_id)

        assert history is not None
        assert len(history) == 4  # 2 user + 2 assistant messages
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "First message"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "Second message"
        assert history[3]["role"] == "assistant"

    def test_clear_conversation(self, mock_client):
        """Test clearing a specific conversation."""
        response = mock_client.prompt(
            model="gpt-4o",
            prompt="Test",
        )
        conv_id = response.conversation_id

        # Verify conversation exists
        assert mock_client.get_conversation_history(conv_id) is not None

        # Clear conversation
        mock_client.clear_conversation(conv_id)

        # Verify conversation is gone
        assert mock_client.get_conversation_history(conv_id) is None

    def test_clear_all_conversations(self, mock_client):
        """Test clearing all conversations."""
        # Create multiple conversations
        response1 = mock_client.prompt(model="gpt-4o", prompt="Test 1")
        response2 = mock_client.prompt(model="gpt-4o", prompt="Test 2")

        conv_id1 = response1.conversation_id
        conv_id2 = response2.conversation_id

        # Clear all
        mock_client.clear_all_conversations()

        # Verify all are gone
        assert mock_client.get_conversation_history(conv_id1) is None
        assert mock_client.get_conversation_history(conv_id2) is None


class TestOpenAICaching:
    """Test OpenAI automatic caching functionality."""

    @pytest.fixture
    def mock_openai_client(self, mocker):
        """Create a mock OpenAI client."""
        # Patch where OpenAI is imported (in openai_client.py)
        mock_openai = mocker.patch("ai_client.openai_client.OpenAI")

        # Create a mock response with cached tokens
        mock_response = mocker.Mock()
        mock_response.model = "gpt-4o"
        mock_response.choices = [
            mocker.Mock(
                message=mocker.Mock(content="Test response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = mocker.Mock(
            prompt_tokens=2000,
            completion_tokens=100,
            total_tokens=2100,
            prompt_tokens_details=mocker.Mock(cached_tokens=1920),
        )

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        client = create_ai_client("openai", api_key="test_key")
        return client

    def test_openai_cached_tokens_extraction(self, mock_openai_client):
        """Test that cached tokens are extracted correctly from OpenAI responses."""
        response = mock_openai_client.prompt(
            model="gpt-4o",
            prompt="x" * 2000,  # Long prompt to trigger caching
        )

        assert response.usage.cached_tokens == 1920
        assert response.usage.input_tokens == 2000
        assert response.usage.get_cache_savings() > 0.9  # 90%+ cached

    def test_openai_cache_parameters(self, mock_openai_client, mocker):
        """Test that OpenAI caching parameters are passed correctly via kwargs."""
        spy = mocker.spy(mock_openai_client.api_client.chat.completions, "create")

        mock_openai_client.prompt(
            model="gpt-4o",
            prompt="Test prompt",
            cache=True,  # Generic flag
            prompt_cache_key="test_key_123",  # Via kwargs
            prompt_cache_retention="24h",  # Via kwargs
        )

        # Verify parameters were passed
        call_kwargs = spy.call_args[1]
        assert call_kwargs["prompt_cache_key"] == "test_key_123"
        assert call_kwargs["prompt_cache_retention"] == "24h"


class TestClaudeCaching:
    """Test Claude cache_control blocks functionality."""

    @pytest.fixture
    def mock_claude_client(self, mocker):
        """Create a mock Claude client."""
        # Patch where Anthropic is imported (in claude_client.py)
        mock_anthropic = mocker.patch("ai_client.claude_client.Anthropic")

        # Create a mock response with cache metrics
        mock_response = mocker.Mock()
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.content = [mocker.Mock(type="text", text="Test response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = mocker.Mock(
            input_tokens=50,
            output_tokens=100,
            cache_creation_input_tokens=2000,
            cache_read_input_tokens=0,
        )

        mock_anthropic.return_value.messages.create.return_value = mock_response

        client = create_ai_client("anthropic", api_key="test_key")
        return client

    def test_claude_cache_creation_tokens(self, mock_claude_client):
        """Test that cache creation tokens are extracted correctly."""
        response = mock_claude_client.prompt(
            model="claude-3-5-sonnet-20241022",
            prompt="Analyze this document",
            files=["test_file.txt"],
            cache=True,  # Generic caching flag
        )

        assert response.usage.cache_creation_tokens == 2000
        assert response.usage.cache_read_tokens == 0

    def test_claude_cache_control_blocks(self, mock_claude_client, mocker, tmp_path):
        """Test that cache_control blocks are added when cache=True."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("x" * 5000)  # Long content

        spy = mocker.spy(mock_claude_client.api_client.messages, "create")

        mock_claude_client.prompt(
            model="claude-3-5-sonnet-20241022",
            prompt="Analyze this",
            files=[str(test_file)],
            cache=True,  # Generic caching flag
        )

        # Verify cache_control blocks were added
        call_kwargs = spy.call_args[1]
        system = call_kwargs["system"]

        assert isinstance(system, list)
        assert len(system) > 0
        assert "cache_control" in system[0]
        assert system[0]["cache_control"]["type"] == "ephemeral"

    def test_claude_cache_read_tokens(self, mock_claude_client, mocker):
        """Test that cache read tokens are extracted on subsequent requests."""
        # Mock a response with cache hits
        mock_response = mocker.Mock()
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.content = [mocker.Mock(type="text", text="Cached response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = mocker.Mock(
            input_tokens=50,
            output_tokens=100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2000,  # Cache hit!
        )

        mock_claude_client.api_client.messages.create.return_value = mock_response

        response = mock_claude_client.prompt(
            model="claude-3-5-sonnet-20241022",
            prompt="Another question",
        )

        assert response.usage.cache_read_tokens == 2000
        assert response.usage.cache_creation_tokens == 0
        assert response.usage.get_cache_savings() > 0  # Some savings from cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
