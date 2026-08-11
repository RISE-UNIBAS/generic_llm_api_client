"""
HuggingFace-specific client using OpenAI-compatible API.

HuggingFace exposes two OpenAI-compatible surfaces, both served by this client:

1. Inference Providers router (default): https://router.huggingface.co/v1
   A single HF token gives access to the open-weight models in the Inference
   Providers catalog, executed by partner providers (Groq, Together, Cerebras,
   Novita, ...). Model ids are Hub repo ids, optionally with a routing suffix:
       ":fastest"   highest throughput provider (the default)
       ":cheapest"  lowest price per output token
       ":preferred" your provider preference order from Hub settings
       ":<name>"    pin one provider, e.g. ":groq"

2. Dedicated Inference Endpoints: pass base_url explicitly, e.g.
   "https://<id>.<region>.<cloud>.endpoints.huggingface.cloud/v1". This is the
   route for any Hub model the router does not serve. Note that such endpoints
   take the *endpoint name* as the model argument, not the Hub repo id, and are
   billed per hardware-hour rather than per token.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .openai_client import OpenAIClient


class HuggingFaceClient(OpenAIClient):
    """
    HuggingFace client using the OpenAI-compatible Inference Providers API.

    Extends OpenAIClient with HuggingFace's router base URL. Supply an explicit
    base_url to target a dedicated Inference Endpoint instead.
    """

    PROVIDER_ID = "huggingface"
    SUPPORTS_MULTIMODAL = True

    ROUTER_BASE_URL = "https://router.huggingface.co/v1"

    # Some HuggingFace partner providers sit behind a WAF (Cloudflare) that blocks the OpenAI
    # SDK's default "OpenAI/Python" User-Agent with HTTP 403 "Your request was blocked."
    # Observed on the publicai and scaleway providers. A neutral User-Agent is accepted by
    # every provider tested, so default to one. The caller can still override it by passing
    # their own User-Agent in default_headers.
    DEFAULT_USER_AGENT = "generic-llm-api-client"

    def _init_client(self):
        """Initialize the client with HuggingFace's router base URL and a WAF-safe User-Agent."""
        if not self.base_url:
            self.base_url = self.ROUTER_BASE_URL

        headers = dict(self.settings.get("default_headers") or {})
        if not any(k.lower() == "user-agent" for k in headers):
            headers["User-Agent"] = self.DEFAULT_USER_AGENT
        self.settings["default_headers"] = headers

        super()._init_client()

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models.

        Overrides the OpenAI implementation because dedicated Inference Endpoints
        may omit the `created` timestamp that the base class assumes is present.

        Returns:
            List of tuples (model_id, created_date), where created_date is None
            if the endpoint does not report one
        """
        if self.api_client is None:
            raise ValueError("HuggingFace client is not initialized.")

        model_list = []

        for model in self.api_client.models.list():
            created = getattr(model, "created", None)
            readable_date = None
            if isinstance(created, (int, float)):
                readable_date = datetime.fromtimestamp(created, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                )
            model_list.append((model.id, readable_date))

        return model_list
