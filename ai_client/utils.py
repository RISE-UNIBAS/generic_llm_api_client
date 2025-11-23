"""
Utility functions for AI client operations.

This module provides common utilities like retry logic, rate limiting,
and error handling for LLM API interactions.
"""

import time
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class APIError(Exception):
    """Base exception for API errors."""

    pass


def retry_with_exponential_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[..., T]:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exceptions that should trigger a retry

    Returns:
        Wrapped function with retry logic
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e

                if attempt == max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                    raise

                # Calculate delay with exponential backoff
                delay = min(initial_delay * (exponential_base**attempt), max_delay)

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception

    return wrapper


def is_rate_limit_error(exception: Exception) -> bool:
    """
    Check if an exception is a rate limit error.

    This handles various provider-specific rate limit exceptions.
    """
    error_message = str(exception).lower()
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "quota exceeded",
        "resource_exhausted",
    ]

    return any(indicator in error_message for indicator in rate_limit_indicators)


def get_retry_delay_from_error(exception: Exception) -> Optional[float]:
    """
    Extract retry delay from error message if available.

    Some providers include a retry-after header or message.
    """
    error_message = str(exception).lower()

    # Try to extract "retry after X seconds" patterns
    import re

    patterns = [r"retry after (\d+)", r"retry in (\d+)", r"wait (\d+) seconds"]

    for pattern in patterns:
        match = re.search(pattern, error_message)
        if match:
            return float(match.group(1))

    return None


def detect_image_mime_type(file_path: str) -> str:
    """
    Detect MIME type of an image from its file extension.

    Args:
        file_path: Path to the image file or URL

    Returns:
        MIME type string (e.g., "image/png", "image/jpeg")
        Defaults to "image/jpeg" if extension is not recognized
    """
    import os

    ext = os.path.splitext(file_path)[1].lower()

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }

    return mime_types.get(ext, "image/jpeg")


def read_text_files(file_paths: list[str]) -> str:
    """
    Read text files and format them for inclusion in a prompt.

    Args:
        file_paths: List of paths to text files

    Returns:
        Formatted string with file contents wrapped in XML-like tags

    Example:
        >>> files = read_text_files(['doc1.txt', 'doc2.txt'])
        >>> # Returns:
        >>> # <file name="doc1.txt">
        >>> # content of doc1...
        >>> # </file>
        >>> #
        >>> # <file name="doc2.txt">
        >>> # content of doc2...
        >>> # </file>
    """
    import os

    if not file_paths:
        return ""

    file_sections = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)

        try:
            # Try UTF-8 first (most common)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 for older documents
            try:
                with open(filepath, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read file {filepath}: {e}")
                content = f"[Error reading file: {e}]"
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            content = f"[File not found: {filepath}]"
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            content = f"[Error: {e}]"

        file_sections.append(f'<file name="{filename}">\n{content}\n</file>')

    return "\n\n" + "\n\n".join(file_sections)


def resize_image_if_needed(
    image_path: str,
    max_size: Optional[int] = None,
    quality: int = 85,
) -> str:
    """
    Resize an image if it exceeds the maximum dimensions.

    Creates a temporary resized copy if resizing is needed.
    Original file is never modified.

    Args:
        image_path: Path to the image file
        max_size: Maximum width or height in pixels (None = no resize)
        quality: JPEG quality for resized image (1-100)

    Returns:
        Path to the image to use (original or resized temp file)

    Example:
        >>> # Image is 4000x3000, max_size=2048
        >>> resized_path = resize_image_if_needed('large.jpg', max_size=2048)
        >>> # Returns path to temp file with image resized to 2048x1536
    """
    if max_size is None:
        return image_path

    try:
        from PIL import Image
        import tempfile
        import os

        # Open and check size
        img = Image.open(image_path)

        # No resize needed if within bounds
        if max(img.size) <= max_size:
            logger.debug(f"Image {image_path} is within size limit ({img.size})")
            return image_path

        # Calculate new size maintaining aspect ratio
        original_size = img.size
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Create temporary file
        suffix = os.path.splitext(image_path)[1] or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

        # Save resized image
        # Convert RGBA to RGB for JPEG
        if img.mode in ("RGBA", "LA", "P"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = rgb_img

        img.save(temp_file.name, "JPEG", quality=quality, optimize=True)
        temp_file.close()

        logger.info(
            f"Resized image {os.path.basename(image_path)} from {original_size} to {img.size}"
        )

        return temp_file.name

    except ImportError:
        logger.warning(
            "Pillow not installed - cannot resize images. Install with: pip install Pillow"
        )
        return image_path
    except Exception as e:
        logger.warning(f"Failed to resize image {image_path}: {e}. Using original.")
        return image_path
