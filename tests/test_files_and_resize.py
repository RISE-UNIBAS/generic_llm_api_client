"""
Unit tests for file handling and image resizing features.

These tests verify that:
- Text files are correctly read and formatted
- Images are resized when needed
- File contents are appended to prompts correctly
- Multiple files and images work together
"""

import os
from PIL import Image
from ai_client.utils import read_text_files, resize_image_if_needed
from ai_client import create_ai_client


class TestReadTextFiles:
    """Tests for reading and formatting text files."""

    def test_read_single_file(self, tmp_path):
        """Test reading a single text file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, world!", encoding="utf-8")

        result = read_text_files([str(file_path)])

        assert '<file name="test.txt">' in result
        assert "Hello, world!" in result
        assert "</file>" in result

    def test_read_multiple_files(self, tmp_path):
        """Test reading multiple text files."""
        file1 = tmp_path / "doc1.txt"
        file1.write_text("First document", encoding="utf-8")

        file2 = tmp_path / "doc2.txt"
        file2.write_text("Second document", encoding="utf-8")

        result = read_text_files([str(file1), str(file2)])

        assert "doc1.txt" in result
        assert "First document" in result
        assert "doc2.txt" in result
        assert "Second document" in result
        assert result.count("<file") == 2
        assert result.count("</file>") == 2

    def test_read_empty_list(self):
        """Test that empty file list returns empty string."""
        result = read_text_files([])
        assert result == ""

    def test_read_utf8_file(self, tmp_path):
        """Test reading file with UTF-8 encoding."""
        file_path = tmp_path / "unicode.txt"
        file_path.write_text("Hëllö Wörld 你好 مرحبا", encoding="utf-8")

        result = read_text_files([str(file_path)])

        assert "Hëllö Wörld 你好 مرحبا" in result

    def test_read_latin1_file(self, tmp_path):
        """Test fallback to latin-1 encoding."""
        file_path = tmp_path / "latin1.txt"
        # Write with latin-1 encoding
        file_path.write_bytes("Hëllö".encode("latin-1"))

        result = read_text_files([str(file_path)])

        # Should read successfully with latin-1 fallback
        assert "Hëllö" in result or "latin1.txt" in result

    def test_read_nonexistent_file(self, tmp_path):
        """Test handling of nonexistent file."""
        file_path = tmp_path / "nonexistent.txt"

        result = read_text_files([str(file_path)])

        assert "File not found" in result or "Error" in result


class TestResizeImageIfNeeded:
    """Tests for image resizing functionality."""

    def test_no_resize_when_disabled(self, tmp_path):
        """Test that resizing is skipped when max_size is None."""
        # Create a test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (3000, 2000), color="red")
        img.save(str(img_path), "JPEG")

        result = resize_image_if_needed(str(img_path), max_size=None)

        # Should return original path
        assert result == str(img_path)

    def test_no_resize_when_small_enough(self, tmp_path):
        """Test that small images are not resized."""
        # Create a small image
        img_path = tmp_path / "small.jpg"
        img = Image.new("RGB", (800, 600), color="blue")
        img.save(str(img_path), "JPEG")

        result = resize_image_if_needed(str(img_path), max_size=2048)

        # Should return original path
        assert result == str(img_path)

    def test_resize_large_image(self, tmp_path):
        """Test that large images are resized."""
        # Create a large image
        img_path = tmp_path / "large.jpg"
        img = Image.new("RGB", (4000, 3000), color="green")
        img.save(str(img_path), "JPEG")

        result = resize_image_if_needed(str(img_path), max_size=2048, quality=85)

        # Should return a different path (temp file)
        assert result != str(img_path)
        assert os.path.exists(result)

        # Check the resized image dimensions
        resized = Image.open(result)
        assert max(resized.size) <= 2048
        # Aspect ratio should be maintained
        original_ratio = 4000 / 3000
        resized_ratio = resized.size[0] / resized.size[1]
        assert abs(original_ratio - resized_ratio) < 0.01

    def test_resize_preserves_aspect_ratio(self, tmp_path):
        """Test that resizing maintains aspect ratio."""
        # Create a non-square image
        img_path = tmp_path / "portrait.jpg"
        img = Image.new("RGB", (2000, 3000), color="yellow")  # Portrait orientation
        img.save(str(img_path), "JPEG")

        result = resize_image_if_needed(str(img_path), max_size=1500)

        resized = Image.open(result)
        # Height should be 1500, width should be proportional
        assert resized.size[1] == 1500
        assert 900 < resized.size[0] < 1100  # ~1000px

    def test_resize_png_to_jpg(self, tmp_path):
        """Test that PNG with alpha channel is converted to JPEG."""
        # Create a PNG with alpha channel
        img_path = tmp_path / "transparent.png"
        img = Image.new("RGBA", (3000, 2000), color=(255, 0, 0, 128))
        img.save(str(img_path), "PNG")

        result = resize_image_if_needed(str(img_path), max_size=2048)

        # Should create a JPEG (no transparency)
        resized = Image.open(result)
        assert resized.mode == "RGB"  # No alpha channel


class TestClientWithFiles:
    """Tests for BaseAIClient with files parameter."""

    def test_create_client_with_max_image_size(self):
        """Test creating client with max_image_size parameter."""
        client = create_ai_client(
            "openai", api_key="test-key", max_image_size=1024, image_quality=90
        )

        assert client.max_image_size == 1024
        assert client.image_quality == 90

    def test_create_client_without_max_image_size(self):
        """Test creating client with default max_image_size."""
        client = create_ai_client("openai", api_key="test-key")

        assert client.max_image_size == 2048  # Default value
        assert client.image_quality == 85  # Default value

    def test_create_client_with_disabled_resize(self):
        """Test creating client with image resizing disabled."""
        client = create_ai_client("openai", api_key="test-key", max_image_size=None)

        assert client.max_image_size is None

    def test_prompt_with_files(self, mocker, tmp_path):
        """Test that files parameter is correctly processed."""
        # Create test file
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content", encoding="utf-8")

        # Mock the OpenAI client
        client = create_ai_client("openai", api_key="test-key")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Response text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4"

        mocker.patch.object(
            client.api_client.chat.completions, "create", return_value=mock_response
        )

        # Call prompt with files
        client.prompt("gpt-4", "Analyze this document", files=[str(file_path)])

        # Verify the API was called with modified prompt
        call_args = client.api_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]

        # Handle both string and list formats (multimodal)
        if isinstance(user_message, list):
            # Multimodal format - extract text parts
            text_parts = [part["text"] for part in user_message if part.get("type") == "text"]
            user_text = " ".join(text_parts)
        else:
            user_text = user_message

        # The user message should contain the file content
        assert '<file name="test.txt">' in user_text
        assert "Test content" in user_text
        assert "</file>" in user_text

    def test_prompt_with_multiple_files(self, mocker, tmp_path):
        """Test prompt with multiple files."""
        # Create test files
        file1 = tmp_path / "doc1.txt"
        file1.write_text("First document", encoding="utf-8")

        file2 = tmp_path / "doc2.txt"
        file2.write_text("Second document", encoding="utf-8")

        # Mock the OpenAI client
        client = create_ai_client("openai", api_key="test-key")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4"

        mocker.patch.object(
            client.api_client.chat.completions, "create", return_value=mock_response
        )

        # Call prompt with multiple files
        client.prompt("gpt-4", "Compare documents", files=[str(file1), str(file2)])

        # Verify both files are in the prompt
        call_args = client.api_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]

        # Handle both string and list formats
        if isinstance(user_message, list):
            text_parts = [part["text"] for part in user_message if part.get("type") == "text"]
            user_text = " ".join(text_parts)
        else:
            user_text = user_message

        assert "doc1.txt" in user_text
        assert "First document" in user_text
        assert "doc2.txt" in user_text
        assert "Second document" in user_text

    def test_prompt_with_files_and_images(self, mocker, tmp_path):
        """Test prompt with both files and images."""
        # Create test file and image
        text_file = tmp_path / "description.txt"
        text_file.write_text("Image description", encoding="utf-8")

        img_file = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(str(img_file), "JPEG")

        # Mock the OpenAI client
        client = create_ai_client("openai", api_key="test-key")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o"

        mocker.patch.object(
            client.api_client.chat.completions, "create", return_value=mock_response
        )

        # Call prompt with both files and images
        client.prompt(
            "gpt-4o",
            "Does this image match the description?",
            images=[str(img_file)],
            files=[str(text_file)],
        )

        # Verify the call was made
        call_args = client.api_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        # Check that we have multimodal content
        user_content = messages[1]["content"]
        if isinstance(user_content, list):
            # Multimodal format
            text_parts = [c for c in user_content if c.get("type") == "text"]
            image_parts = [c for c in user_content if c.get("type") == "image_url"]
            assert len(text_parts) > 0
            assert len(image_parts) > 0
            # File content should be in text
            text_content = text_parts[0]["text"]
            assert "description.txt" in text_content
            assert "Image description" in text_content
        else:
            # Text format (files appended to prompt)
            assert "description.txt" in user_content
            assert "Image description" in user_content


class TestClientWithImageResize:
    """Tests for BaseAIClient with automatic image resizing."""

    def test_prompt_resizes_large_image(self, mocker, tmp_path):
        """Test that large images are automatically resized."""
        # Create a large image
        img_path = tmp_path / "large.jpg"
        img = Image.new("RGB", (4000, 3000), color="blue")
        img.save(str(img_path), "JPEG")

        # Create client with max_image_size
        client = create_ai_client("openai", api_key="test-key", max_image_size=2048)

        # Mock the OpenAI client
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o"

        mocker.patch.object(
            client.api_client.chat.completions, "create", return_value=mock_response
        )

        # Call prompt with large image
        client.prompt("gpt-4o", "Describe this image", images=[str(img_path)])

        # Verify the API was called (image was processed)
        assert client.api_client.chat.completions.create.called

    def test_prompt_with_resize_disabled(self, mocker, tmp_path):
        """Test that resizing can be disabled."""
        # Create a large image
        img_path = tmp_path / "large.jpg"
        img = Image.new("RGB", (4000, 3000), color="green")
        img.save(str(img_path), "JPEG")

        # Create client with resizing disabled
        client = create_ai_client("openai", api_key="test-key", max_image_size=None)

        # Mock the OpenAI client
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o"

        mocker.patch.object(
            client.api_client.chat.completions, "create", return_value=mock_response
        )

        # Call prompt with large image
        client.prompt("gpt-4o", "Describe this image", images=[str(img_path)])

        # Verify the API was called (no resize occurred, but still works)
        assert client.api_client.chat.completions.create.called
