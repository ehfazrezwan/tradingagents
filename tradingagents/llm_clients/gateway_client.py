import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_DEFAULT_GATEWAY_URL = "http://svc-utility-belt.optimizely.com/llm-gateway/v1"

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "callbacks",
    "http_client", "http_async_client",
)


class NormalizedGatewayChatOpenAI(ChatOpenAI):
    """ChatOpenAI for the LLM Gateway with normalized content output.

    The gateway proxies multiple providers whose responses may contain
    block-structured content. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class GatewayClient(BaseLLMClient):
    """Client for the OpenAI-compatible LLM Gateway.

    Connects to an internal gateway that exposes multiple providers
    (Gemini, Claude) via a unified OpenAI-compatible API.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance pointing at the gateway."""
        self.warn_if_unknown_model()

        gateway_url = (
            self.base_url
            or os.environ.get("LLM_GATEWAY_URL")
            or _DEFAULT_GATEWAY_URL
        )
        api_key = (
            self.kwargs.get("api_key")
            or os.environ.get("LLM_GATEWAY_API_KEY", "")
        )

        llm_kwargs = {
            "model": self.model,
            "base_url": gateway_url,
            "api_key": api_key,
        }

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedGatewayChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the gateway."""
        return validate_model("gateway", self.model)
