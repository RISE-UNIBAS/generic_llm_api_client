"""
Content ordering policy for request assembly.

This module defines the ContentOrder enum, which controls how prompt text and
attachments (images, text files, etc.) are arranged when building provider-specific
request payloads.

The policy applies only to request assembly. It does not affect response
parsing, tool calling, schema validation, or any other logic.

Usage
-----
Simple policy (prompt vs. all attachments):

    client = create_ai_client('openai', api_key=...,
                              content_order='attachments_before_prompt')

Fine-grained ordering using named content slot strings:

    client = create_ai_client('openai', api_key=...,
                              content_order=['images', 'prompt', 'files'])

Per-request override:

    response = client.prompt(model=..., prompt=..., images=[...], files=[...],
                             content_order=['files', 'images', 'prompt'])

Known slot names
----------------
    "prompt"  — the main text prompt
    "images"  — image attachments (passed via the ``images`` parameter)
    "files"   — text-file attachments (passed via the ``files`` parameter)

Any slot not present in the supplied order list is appended at the end in
natural order, so unknown future attachment types degrade gracefully.
"""

from enum import Enum

# Named content slot identifiers.
# These are plain strings; the constants are provided for type-safe use.
SLOT_PROMPT = "prompt"
SLOT_IMAGES = "images"
SLOT_FILES = "files"


class ContentOrder(str, Enum):
    """
    High-level content-ordering policy shortcuts.

    These are conveniences that expand to canonical slot lists internally.
    For full control over individual slot positions, pass a list of slot
    name strings directly instead of a ContentOrder value.

    Values
    ------
    DEFAULT
        Preserve the provider's natural ordering:
        prompt → files → images.
    PROMPT_BEFORE_ATTACHMENTS
        Explicitly place the prompt before all attachments:
        prompt → images → files.
    ATTACHMENTS_BEFORE_PROMPT
        Place all attachments before the prompt:
        images → files → prompt.
    """

    DEFAULT = "default"
    PROMPT_BEFORE_ATTACHMENTS = "prompt_before_attachments"
    ATTACHMENTS_BEFORE_PROMPT = "attachments_before_prompt"