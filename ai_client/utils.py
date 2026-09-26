"""
Utility functions for AI client operations.

This module provides common utilities like retry logic, rate limiting,
and error handling for LLM API interactions.
"""

import json
import logging
import math
import time
from typing import Callable, Optional, Tuple, TypeVar
from functools import wraps

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class APIError(Exception):
    """Base exception for API errors."""

    pass


class ToolNotSupportedError(APIError):
    """Raised when tool calling is requested but not supported by provider/model."""

    pass


class ToolRegistryError(APIError):
    """Raised when tool registry cannot be loaded or tool not found."""

    pass


class ToolExecutionError(APIError):
    """Raised when tool execution fails."""

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


# Providers name the same three counts differently: OpenAI, Anthropic and genai in turn.
_INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count")
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens", "candidates_token_count")
_TOTAL_TOKEN_KEYS = ("total_tokens", "total_token_count")

# Where a response-shaped payload carries its usage block.
_USAGE_KEYS = ("usage", "usage_metadata")

# Cohere reports its counts one level below the usage block.
_NESTED_USAGE_KEYS = ("tokens", "billed_units")

# Discarded-attempt totals travel on the exception itself when a fallback also fails, so
# the error response can still report what was billed. Per-exception, so thread-safe.
DISCARDED_ATTRIBUTE = "_ai_client_discarded"

# A response body attached to an exception that arrived without one, for the same reason.
PAYLOAD_ATTRIBUTE = "_ai_client_payload"


def _member(source, key):
    """Read a key from a mapping or an attribute from an object."""
    return source.get(key) if isinstance(source, dict) else getattr(source, key, None)


def _read_token_count(source, keys) -> Optional[int]:
    """Read the first of keys present on a mapping or object as a plain integer."""
    for key in keys:
        value = _member(source, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        # Cohere reports whole numbers as floats.
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _counts_holder(source):
    """Return whichever object actually holds the counts, one level down if needed."""
    if source is None:
        return None
    if _read_token_count(source, _INPUT_TOKEN_KEYS + _OUTPUT_TOKEN_KEYS) is not None:
        return source

    for key in _NESTED_USAGE_KEYS:
        nested = _member(source, key)
        if nested is not None and (
            _read_token_count(nested, _INPUT_TOKEN_KEYS + _OUTPUT_TOKEN_KEYS) is not None
        ):
            return nested
    return None


def output_reported(source) -> bool:
    """Return True if the provider actually reported a completion count."""
    return _read_token_count(_counts_holder(source), _OUTPUT_TOKEN_KEYS) is not None


def usage_counts(source) -> Optional[tuple]:
    """
    Read token counts from a provider usage object or mapping.

    Providers name the fields differently: OpenAI reports prompt_tokens and
    completion_tokens, Anthropic input_tokens and output_tokens. Both are accepted.

    Args:
        source: A usage object or mapping from any provider

    Returns:
        (input_tokens, output_tokens, total_tokens), or None if no counts were found
    """
    if source is None:
        return None

    source = _counts_holder(source)
    if source is None:
        return None

    input_tokens = _read_token_count(source, _INPUT_TOKEN_KEYS)
    output_tokens = _read_token_count(source, _OUTPUT_TOKEN_KEYS)

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    total = _read_token_count(source, _TOTAL_TOKEN_KEYS)
    return (input_tokens, output_tokens, input_tokens + output_tokens if total is None else total)


def rejects_parameter(exception: Exception, parameter: str) -> bool:
    """
    Return True when a provider refused a named request parameter.

    Providers word the refusal differently -- OpenAI with "Unsupported value: 'temperature'
    does not support 0.2 with this model", Anthropic with "`temperature` is deprecated for
    this model" -- so match the parameter name against any of the ways they say no.

    Args:
        exception: Exception raised by a provider call
        parameter: Request parameter to look for

    Returns:
        True if the provider rejected that parameter
    """
    message = str(exception).lower()
    if parameter.lower() not in message:
        return False

    return any(
        phrase in message
        for phrase in ("unsupported", "not supported", "does not support", "deprecated")
    )


def billed_cost(source) -> Optional[float]:
    """
    Return a cost the provider reported charging, if it gave a usable number.

    Checked by type rather than for None: the attribute exists on any mock object, and a
    cost of None is not a billed total.

    Args:
        source: A usage object or mapping from any provider

    Returns:
        The billed cost in USD, or None if the provider reported none
    """
    if source is None:
        return None

    return _number(_field(source, "cost"))


def billed_components(source) -> Optional[Tuple[float, float]]:
    """
    Return the input and output parts of a billed cost, if the provider itemised it.

    OpenRouter reports the upstream prompt and completion costs beside its total. They are
    taken only when they account for the total exactly: a per-request fee (an image or
    request charge) belongs to neither, and splitting it between them would invent an
    attribution the provider did not make. A bring-your-own-key route is skipped too,
    since its total is OpenRouter's fee alone and the upstream figures describe a bill
    paid elsewhere.

    Args:
        source: A usage object or mapping from any provider

    Returns:
        (input_cost_usd, output_cost_usd), or None if the total cannot be split
    """
    if source is None:
        return None

    total = billed_cost(source)
    if total is None or _field(source, "is_byok") is True:
        return None

    details = _field(source, "cost_details")
    prompt_cost = _number(_field(details, "upstream_inference_prompt_cost"))
    completion_cost = _number(_field(details, "upstream_inference_completions_cost"))
    if prompt_cost is None or completion_cost is None:
        return None

    if not math.isclose(prompt_cost + completion_cost, total, rel_tol=1e-9, abs_tol=1e-12):
        return None
    return prompt_cost, completion_cost


def _field(source, name):
    """Read a field from a mapping or an object, whichever the provider returned."""
    if source is None:
        return None
    return source.get(name) if isinstance(source, dict) else getattr(source, name, None)


def _number(value) -> Optional[float]:
    """
    Return value as a float if it is a real number, else None.

    Checked by type rather than for None: the attribute exists on any mock object.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


# Keywords that make an object a JSON Schema rather than an instance of one.
_SCHEMA_KEYWORDS = ("$defs", "definitions", "$schema", "properties")


def schema_instruction(schema: dict) -> str:
    """
    Return the prompt text asking for JSON that conforms to a schema.

    Worded to ask for an instance filled with values: asked for JSON "matching this
    schema", some models answer with the schema itself, which parses and validates
    against a model whose fields are optional while carrying no data at all.

    Args:
        schema: JSON Schema of the expected response

    Returns:
        Text to append to the user's prompt
    """
    return (
        "\n\nRespond with a single JSON object containing your answer. The object must be "
        "an instance of the JSON Schema below: fill in the actual values, and do not "
        f"repeat the schema itself.\nJSON Schema: {json.dumps(schema)}"
    )


def json_schema_of(response_format) -> Optional[dict]:
    """Return the JSON Schema of a Pydantic model class, v2 or v1, or None."""
    for method in ("model_json_schema", "schema"):
        if hasattr(response_format, method):
            try:
                schema = getattr(response_format, method)()
            except Exception:
                return None
            return schema if isinstance(schema, dict) else None
    return None


def rejects_schema_echo(data, response_format):
    """
    Return data, or None if it is the requested schema echoed back rather than an answer.

    Such a response parses as JSON and validates against a model whose fields are
    optional, so it would otherwise pass as an answer carrying no data.

    Args:
        data: Parsed JSON response
        response_format: Pydantic model class that was requested

    Returns:
        data unchanged, or None if it was a schema echo
    """
    if data and response_format and is_schema_echo(data, json_schema_of(response_format)):
        logger.warning(
            "Response is the requested JSON Schema echoed back, not an instance of it; "
            "treating it as a failed parse."
        )
        return None
    return data


def is_schema_echo(data, schema: Optional[dict]) -> bool:
    """
    Tell whether a parsed response is the requested schema echoed back, not an answer.

    Recognised by schema keywords at the top level that the schema does not declare as
    fields of its own, in any branch of a union.

    Args:
        data: Parsed JSON response
        schema: JSON Schema that was requested

    Returns:
        True if the response looks like a schema rather than an instance of it
    """
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return False

    declared = _declared_fields(schema, schema, set())
    return any(key in data and key not in declared for key in _SCHEMA_KEYWORDS)


def _declared_fields(node, root: dict, seen: set) -> set:
    """
    Collect the top-level field names a schema allows, wherever it declares them.

    Pydantic puts a recursive model's fields behind a $ref, and a union's behind
    anyOf/oneOf branches (allOf for combined ones), so each is followed. A ref that
    cannot be resolved, or has already been visited, contributes nothing.

    Args:
        node: Schema node to read
        root: Whole schema, against which local refs ("#/$defs/Name") resolve
        seen: Refs already followed, guarding against cycles

    Returns:
        Names of the fields the node declares
    """
    if not isinstance(node, dict):
        return set()

    fields = node.get("properties")
    declared = set(fields) if isinstance(fields, dict) else set()

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/") and ref not in seen:
        seen.add(ref)
        target = root
        for part in ref[2:].split("/"):
            target = target.get(part) if isinstance(target, dict) else None
        declared |= _declared_fields(target, root, seen)

    for combinator in ("anyOf", "oneOf", "allOf"):
        branches = node.get(combinator)
        if isinstance(branches, list):
            for branch in branches:
                declared |= _declared_fields(branch, root, seen)
    return declared


def usage_of(payload):
    """Return the usage object held by a response-shaped payload, whatever it calls it."""
    if payload is None:
        return None

    for key in _USAGE_KEYS:
        usage = payload.get(key) if isinstance(payload, dict) else getattr(payload, key, None)
        if usage_counts(usage) is not None:
            return usage
    return None


def error_payload(exception: Exception):
    """
    Return the response-shaped payload a failed request carried, if any.

    A request can fail after the provider generated and charged for tokens: structured
    output that would not validate, or a response stopped by a content filter. The SDKs
    expose that response in different places, so each is tried in turn. The payload is
    returned whole rather than as bare counts, so its cost and reasoning details survive
    alongside them.

    Args:
        exception: Exception raised by a provider call

    Returns:
        The payload holding a usage block, or None if the failure carried none
    """
    attached = getattr(exception, PAYLOAD_ATTRIBUTE, None)
    if usage_counts(usage_of(attached)) is not None:
        return attached

    completion = getattr(exception, "completion", None)
    if usage_counts(getattr(completion, "usage", None)) is not None:
        return completion

    body = getattr(exception, "body", None)
    if isinstance(body, dict) and usage_counts(body.get("usage")) is not None:
        return body

    response = getattr(exception, "response", None)
    if response is not None:
        try:
            parsed = response.json()
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and usage_counts(parsed.get("usage")) is not None:
            return parsed

    return None


def usage_from_error(exception: Exception) -> Optional[tuple]:
    """
    Recover the token counts a failed request was still billed for.

    Args:
        exception: Exception raised by a provider call

    Returns:
        (input_tokens, output_tokens, total_tokens), or None if the failure carried none
    """
    return usage_counts(usage_of(error_payload(exception)))


def attach_discarded(exception: Exception, discarded) -> Exception:
    """
    Carry discarded-attempt totals on an exception so an error response can report them.

    Args:
        exception: Exception about to be raised
        discarded: DiscardedAttempts accumulated before the failure

    Returns:
        The same exception
    """
    try:
        setattr(exception, DISCARDED_ATTRIBUTE, discarded)
    except AttributeError:  # exceptions defined with __slots__
        pass
    return exception


def attach_payload(exception: Exception, payload) -> Exception:
    """
    Carry a response body on an exception that arrived without one.

    An SDK validates after the request returns, and some of its failures raise exceptions
    holding nothing -- a content filter, or output that does not match the schema. The
    tokens were billed regardless, so the caller attaches the body it already has.

    Args:
        exception: Exception about to be raised
        payload: Response body, as a mapping

    Returns:
        The same exception
    """
    if payload is not None:
        try:
            setattr(exception, PAYLOAD_ATTRIBUTE, payload)
        except AttributeError:  # exceptions defined with __slots__
            pass
    return exception


def discarded_from_error(exception: Exception):
    """
    Return the discarded-attempt totals attached to an exception, if any.

    Args:
        exception: Exception caught from a provider call

    Returns:
        The DiscardedAttempts carried by the exception, or None
    """
    return getattr(exception, DISCARDED_ATTRIBUTE, None)


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

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

    if not file_paths:
        return ""

    file_sections = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        if ext in _IMAGE_EXTENSIONS:
            logger.warning(
                f"'{filename}' looks like an image file but was passed via files=. "
                f"Use images= instead. Skipping."
            )
            continue

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


def extract_json_from_text(text: str) -> Optional[dict | list]:
    """
    Extract and parse JSON from text that may contain JSON in code blocks.

    Handles various formats:
    - "Here is the json: ```json {...}```"
    - "```json {...}```"
    - "``` {...} ```"
    - Plain JSON (as fallback)

    Args:
        text: The text potentially containing JSON

    Returns:
        Parsed JSON as dict or list if found and valid, None otherwise

    Example:
        >>> text = "Here is the data:\\n```json\\n{\"key\": \"value\"}\\n```"
        >>> extract_json_from_text(text)
        {'key': 'value'}
    """
    import json
    import re

    if not text or not isinstance(text, str):
        return None

    content = text.strip()

    try:
        # Try to extract JSON from code blocks
        if "```json" in content:
            # Match ```json ... ```
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            if json_match:
                content = json_match.group(1).strip()
        elif "```" in content:
            # Match generic code blocks ``` ... ```
            json_match = re.search(r"```\s*([\s\S]*?)\s*```", content)
            if json_match:
                content = json_match.group(1).strip()

        # Try to parse as JSON
        json_data = json.loads(content)

        # Validate it's a dict or list (not just a primitive)
        if isinstance(json_data, (dict, list)):
            return json_data

    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        # Not valid JSON or couldn't extract
        logger.debug(f"Could not extract JSON from text: {e}")

    return None
