"""Tests for Claude requests, structured output, and fallback handling."""

import json
from unittest.mock import Mock, patch
from ai_client import create_ai_client, ClaudeClient
from ai_client.response import LLMResponse


class TestClaudeClient:
    """Tests for ClaudeClient."""

    def test_claude_client_initialization(self):
        """Test Claude client initialization."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic:
            client = create_ai_client("anthropic", api_key="test-key")

            assert isinstance(client, ClaudeClient)
            assert client.PROVIDER_ID == "anthropic"
            assert client.SUPPORTS_MULTIMODAL is True
            mock_anthropic.assert_called_once()

    def test_prompt_text_only(self, mock_claude_response):
        """Test text-only prompt."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_client.messages.create.return_value = mock_claude_response

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt("claude-3-5-sonnet-20241022", "Hello!")

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Claude."
            assert response.provider == "anthropic"
            assert response.finish_reason == "end_turn"
            assert response.usage.input_tokens == 15
            assert response.usage.output_tokens == 25
            assert response.duration >= 0

            # Check API call
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"
            assert "messages" in call_args.kwargs
            assert "system" in call_args.kwargs

    def test_prompt_with_images(self, mock_claude_response, sample_image_path):
        """Test prompt with images."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_client.messages.create.return_value = mock_claude_response

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-3-5-sonnet-20241022", "Describe this image", images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)

            # Check that images were included
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            content = messages[0]["content"]
            assert isinstance(content, list)
            assert any(item["type"] == "image" for item in content)

    def test_prompt_with_structured_output(self, mock_pydantic_model):
        """Test prompt with structured output via tools."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # Mock tool-based response
            mock_response = Mock()
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.content = [Mock()]
            mock_response.content[0].type = "tool_use"
            mock_response.content[0].name = "extract_structured_data"
            mock_response.content[0].input = {"name": "test", "value": 42}
            mock_response.stop_reason = "tool_use"
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 15
            mock_response.usage.output_tokens = 25

            mock_client.messages.create.return_value = mock_response

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-3-5-sonnet-20241022", "Extract data", response_format=mock_pydantic_model
            )

            # Check that tools were used
            call_args = mock_client.messages.create.call_args
            assert "tools" in call_args.kwargs
            assert "tool_choice" in call_args.kwargs

            # Check response contains validated JSON
            assert isinstance(response, LLMResponse)
            data = json.loads(response.text)
            assert data["name"] == "test"
            assert data["value"] == 42

            # Check that parsed field is populated
            assert response.parsed is not None
            assert isinstance(response.parsed, dict)
            assert response.parsed["name"] == "test"
            assert response.parsed["value"] == 42

    def test_max_tokens_varies_by_model(self):
        """Test that max_tokens defaults vary by model."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].type = "text"
            mock_response.content[0].text = "test"
            mock_response.stop_reason = "end_turn"
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 10
            mock_client.messages.create.return_value = mock_response

            client = create_ai_client("anthropic", api_key="test-key")

            # Test opus model
            client.prompt("claude-3-opus-20240229", "test")
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["max_tokens"] == 4096

            # Test sonnet model
            client.prompt("claude-3-5-sonnet-20241022", "test")
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["max_tokens"] == 8192


class TestTemperatureParameter:
    """Temperature handling for supported and rejecting models."""

    def test_retired_model_sends_no_temperature(self, mock_claude_response):
        """Omit temperature for known rejecting models."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = mock_claude_response

            client = create_ai_client("anthropic", api_key="test-key")
            client.prompt("claude-sonnet-5", "Hello", temperature=0.5)

        assert "temperature" not in api.messages.create.call_args.kwargs

    def test_ordinary_model_still_sends_temperature(self, mock_claude_response):
        """Preserve temperature for models that support it."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = mock_claude_response

            client = create_ai_client("anthropic", api_key="test-key")
            client.prompt("claude-3-5-sonnet-20241022", "Hello", temperature=0.5)

        assert api.messages.create.call_args.kwargs["temperature"] == 0.5

    def test_unlisted_model_recovers_from_the_rejection(self, mock_claude_response):
        """Retry without temperature after an unlisted model rejects it."""
        rejection = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': '`temperature` is deprecated for this model.'}}"
        )
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = [rejection, mock_claude_response]

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt("claude-brand-new", "Hello", temperature=0.5)

        assert response.finish_reason != "error"
        assert "temperature" not in api.messages.create.call_args.kwargs


def _json_text_response(payload: dict) -> Mock:
    response = Mock()
    response.content = [Mock()]
    response.content[0].type = "text"
    response.content[0].text = json.dumps(payload)
    response.stop_reason = "end_turn"
    response.usage = Mock()
    response.usage.input_tokens = 15
    response.usage.output_tokens = 25
    return response


FORCED_TOOL_REJECTION = Exception(
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'tool_choice: type \"tool\" and \"any\" are not supported for this model.'}}"
)


class TestForcedToolRefusal:
    """Structured-output fallbacks after forced tool use is rejected."""

    def test_listed_model_is_offered_an_optional_tool(self, mock_pydantic_model):
        """Send the original schema as an optional tool for known rejecting models."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = _tool_response({"name": "a", "value": 1})

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-sonnet-5-5", "Extract", response_format=mock_pydantic_model
            )

        api.messages.create.assert_called_once()
        kwargs = api.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["tools"][0]["input_schema"] == mock_pydantic_model.model_json_schema()
        assert "output_config" not in kwargs
        assert response.parsed == {"name": "a", "value": 1}

    def test_optional_tool_says_it_is_the_final_answer(self, mock_pydantic_model):
        """Apply the final-answer instruction only to the optional tool."""
        from ai_client.claude_client import OPTIONAL_TOOL_DESCRIPTION

        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = _tool_response({"name": "a", "value": 1})

            client = create_ai_client("anthropic", api_key="test-key")
            client.prompt("claude-sonnet-5-5", "Extract", response_format=mock_pydantic_model)
            optional = api.messages.create.call_args.kwargs["tools"][0]["description"]
            client.prompt("claude-opus-4-8", "Extract", response_format=mock_pydantic_model)
            forced = api.messages.create.call_args.kwargs["tools"][0]["description"]

        assert optional == OPTIONAL_TOOL_DESCRIPTION
        assert forced == "Extract structured data according to the provided schema"

    def test_unlisted_model_recovers_from_the_rejection(self, mock_pydantic_model):
        """Retry with an optional tool after an unlisted model rejects forced use."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = [
                FORCED_TOOL_REJECTION,
                _tool_response({"name": "b", "value": 2}),
            ]

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-brand-new", "Extract", response_format=mock_pydantic_model
            )

        kwargs = api.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert "output_config" not in kwargs
        assert response.parsed == {"name": "b", "value": 2}
        assert response.usage.attempts == 1  # The rejected request reports no usage.

    def test_plain_text_answer_to_optional_tool_is_accepted(self, mock_pydantic_model):
        """Accept a JSON text response without retrying the optional tool."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = _json_text_response({"name": "t", "value": 5})

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-sonnet-5-5", "Extract", response_format=mock_pydantic_model
            )

        api.messages.create.assert_called_once()
        assert response.parsed == {"name": "t", "value": 5}

    def test_unknown_tool_key_retries_as_text(self, mock_pydantic_model):
        """Retry unknown wrapper keys as text and retain discarded usage."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = [
                _tool_response({"schema": {"name": None, "value": None}}),
                _json_text_response({"name": "c", "value": 3}),
            ]

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-sonnet-5-5", "Extract", response_format=mock_pydantic_model
            )

        assert api.messages.create.call_count == 2
        retry = api.messages.create.call_args.kwargs
        assert "tools" not in retry
        assert "tool_choice" not in retry
        assert "output_config" not in retry
        assert response.parsed == {"name": "c", "value": 3}
        assert response.usage.attempts == 2
        assert response.usage.discarded_output_tokens == 25

    def test_known_keys_only_are_not_retried(self, mock_pydantic_model):
        """Accept known keys without requiring every field to be present."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = _tool_response({"name": "only"})

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-sonnet-5-5", "Extract", response_format=mock_pydantic_model
            )

        api.messages.create.assert_called_once()
        assert response.parsed == {"name": "only"}

    def test_failed_optional_tool_falls_back_to_text(self, mock_pydantic_model):
        """Remove tool parameters before the text fallback."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = [
                Exception("overloaded"),
                _json_text_response({"name": "d", "value": 4}),
            ]

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-opus-5-5", "Extract", response_format=mock_pydantic_model
            )

        assert api.messages.create.call_count == 2
        calls = [call.kwargs for call in api.messages.create.call_args_list]
        assert calls[0]["tool_choice"] == {"type": "auto"}
        assert "tools" not in calls[1] and "tool_choice" not in calls[1]
        assert response.parsed == {"name": "d", "value": 4}

    def test_other_tool_failure_still_falls_back_to_text(self, mock_pydantic_model):
        """Preserve the text fallback for unrelated forced-tool failures."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = [
                Exception("overloaded"),
                _json_text_response({"name": "c", "value": 3}),
            ]

            client = create_ai_client("anthropic", api_key="test-key")
            client.prompt("claude-opus-4-8", "Extract", response_format=mock_pydantic_model)

        kwargs = api.messages.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "output_config" not in kwargs


def _tool_response(payload: dict) -> Mock:
    response = Mock()
    response.content = [Mock()]
    response.content[0].type = "tool_use"
    response.content[0].name = "extract_structured_data"
    response.content[0].input = payload
    response.stop_reason = "tool_use"
    response.usage = Mock()
    response.usage.input_tokens = 15
    response.usage.output_tokens = 25
    return response


class TestOptionalToolKeyCheck:
    """Allowed properties and wrapper detection in optional-tool responses."""

    def test_extra_allow_model_keeps_its_extra_keys(self):
        """Accept explicitly allowed extra keys without retrying."""
        from pydantic import BaseModel, ConfigDict

        class Extensible(BaseModel):
            model_config = ConfigDict(extra="allow")
            name: str

        payload = {"name": "Alice", "score": 7}
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = _tool_response(payload)

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt("claude-sonnet-5-5", "Extract", response_format=Extensible)

        api.messages.create.assert_called_once()
        assert response.parsed == payload
        assert response.usage.attempts == 1
        assert response.usage.discarded_output_tokens == 0

    def test_wrapped_recursive_model_is_retried(self):
        """Resolve the root reference before rejecting wrapper keys."""
        from pydantic import BaseModel, Field

        class Node(BaseModel):
            name: str | None = None
            children: list["Node"] = Field(default_factory=list)

        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = [
                _tool_response({"schema": {"name": "Alice", "children": []}}),
                _json_text_response({"name": "Alice", "children": []}),
            ]

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt("claude-sonnet-5-5", "Extract", response_format=Node)

        assert api.messages.create.call_count == 2
        retry = api.messages.create.call_args.kwargs
        # Rejected optional-tool responses fall back to text.
        assert "output_config" not in retry and "tools" not in retry
        assert response.parsed == {"name": "Alice", "children": []}
        assert response.usage.attempts == 2
        assert response.usage.discarded_output_tokens == 25

    def test_pattern_and_schema_valued_additional_properties(self):
        """Skip key checks for pattern properties or schema-valued additional properties."""
        from ai_client.claude_client import _unknown_keys

        patterned = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "patternProperties": {"^x_": {"type": "integer"}},
        }
        assert _unknown_keys(patterned, {"name": "a", "x_score": 1}) == []
        assert _unknown_keys(patterned, {"name": "a", "schema": {}}) == []

        typed_extra = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        }
        assert _unknown_keys(typed_extra, {"name": "a", "score": 1}) == []

    def test_closed_default_model_still_rejects_a_wrapper(self, mock_pydantic_model):
        """Treat unspecified extra keys as unknown for wrapper detection."""
        from ai_client.claude_client import _unknown_keys

        schema = mock_pydantic_model.model_json_schema()
        assert "additionalProperties" not in schema
        assert _unknown_keys(schema, {"schema": {"name": None}}) == ["schema"]
        assert _unknown_keys(schema, {"name": "a"}) == []
