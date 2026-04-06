import unittest
import warnings
from unittest.mock import patch

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.gateway_client import GatewayClient
from tradingagents.llm_clients.model_catalog import get_known_models
from tradingagents.llm_clients.validators import validate_model


class TestGatewayModelValidation(unittest.TestCase):
    """Verify gateway models are in the catalog and pass validation."""

    def test_gateway_models_in_catalog(self):
        known = get_known_models()
        self.assertIn("gateway", known)
        self.assertGreater(len(known["gateway"]), 0)

    def test_known_gateway_models_pass_validation(self):
        for model in get_known_models()["gateway"]:
            with self.subTest(model=model):
                self.assertTrue(validate_model("gateway", model))

    def test_unknown_gateway_model_fails_validation(self):
        self.assertFalse(validate_model("gateway", "not-a-real-model"))


class TestGatewayClientInstantiation(unittest.TestCase):
    """Verify GatewayClient can be created via constructor and factory."""

    def test_direct_instantiation(self):
        client = GatewayClient("google-vertex/claude-sonnet-4-6")
        self.assertEqual(client.model, "google-vertex/claude-sonnet-4-6")

    def test_factory_returns_gateway_client(self):
        client = create_llm_client("gateway", "google-vertex/claude-sonnet-4-6")
        self.assertIsInstance(client, GatewayClient)

    def test_factory_case_insensitive(self):
        client = create_llm_client("Gateway", "google-vertex/claude-sonnet-4-6")
        self.assertIsInstance(client, GatewayClient)


class TestGatewayClientGetLLM(unittest.TestCase):
    """Verify get_llm() passes correct parameters to ChatOpenAI."""

    @patch("tradingagents.llm_clients.gateway_client.NormalizedGatewayChatOpenAI")
    def test_default_gateway_url(self, mock_chat):
        client = GatewayClient("google-vertex/claude-sonnet-4-6")
        with patch.dict("os.environ", {}, clear=True):
            client.get_llm()

        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "google-vertex/claude-sonnet-4-6")
        self.assertEqual(
            call_kwargs["base_url"],
            "http://svc-utility-belt.optimizely.com/llm-gateway/v1",
        )

    @patch("tradingagents.llm_clients.gateway_client.NormalizedGatewayChatOpenAI")
    def test_env_var_overrides_default_url(self, mock_chat):
        client = GatewayClient("google-vertex/claude-sonnet-4-6")
        with patch.dict("os.environ", {"LLM_GATEWAY_URL": "http://custom-gw/v1"}):
            client.get_llm()

        self.assertEqual(mock_chat.call_args[1]["base_url"], "http://custom-gw/v1")

    @patch("tradingagents.llm_clients.gateway_client.NormalizedGatewayChatOpenAI")
    def test_explicit_base_url_takes_precedence(self, mock_chat):
        client = GatewayClient(
            "google-vertex/claude-sonnet-4-6",
            base_url="http://explicit/v1",
        )
        with patch.dict("os.environ", {"LLM_GATEWAY_URL": "http://env-gw/v1"}):
            client.get_llm()

        self.assertEqual(mock_chat.call_args[1]["base_url"], "http://explicit/v1")

    @patch("tradingagents.llm_clients.gateway_client.NormalizedGatewayChatOpenAI")
    def test_api_key_from_env(self, mock_chat):
        client = GatewayClient("google-vertex/claude-sonnet-4-6")
        with patch.dict("os.environ", {"LLM_GATEWAY_API_KEY": "test-key-123"}, clear=False):
            client.get_llm()

        self.assertEqual(mock_chat.call_args[1]["api_key"], "test-key-123")

    @patch("tradingagents.llm_clients.gateway_client.NormalizedGatewayChatOpenAI")
    def test_api_key_kwarg_takes_precedence(self, mock_chat):
        client = GatewayClient(
            "google-vertex/claude-sonnet-4-6",
            api_key="kwarg-key",
        )
        with patch.dict("os.environ", {"LLM_GATEWAY_API_KEY": "env-key"}):
            client.get_llm()

        self.assertEqual(mock_chat.call_args[1]["api_key"], "kwarg-key")

    def test_unknown_model_emits_warning(self):
        client = GatewayClient("not-a-gateway-model")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("tradingagents.llm_clients.gateway_client.NormalizedGatewayChatOpenAI"):
                client.get_llm()

        self.assertEqual(len(caught), 1)
        self.assertIn("not-a-gateway-model", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
