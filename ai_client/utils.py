"""
Utility functions for AI client operations.

This module provides common utilities like retry logic, rate limiting,
and error handling for LLM API interactions.
"""
import time
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

T = TypeVar('T')

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
    retryable_exceptions: tuple = (Exception,)
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
                delay = min(initial_delay * (exponential_base ** attempt), max_delay)

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
        'rate limit',
        'rate_limit',
        'too many requests',
        '429',
        'quota exceeded',
        'resource_exhausted'
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
    patterns = [
        r'retry after (\d+)',
        r'retry in (\d+)',
        r'wait (\d+) seconds'
    ]

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
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
    }

    return mime_types.get(ext, 'image/jpeg')
