import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AnalystCache:
    """Caches propagate() outputs keyed by (ticker, date, config_hash).

    Stores results as JSON files to avoid redundant LLM calls when:
    - Re-running a backtest after a crash
    - Running multiple trials with the same configuration
    - Iterating on downstream logic without re-analyzing
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, date: str, config_hash: str) -> Path:
        ticker_dir = self.cache_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        return ticker_dir / f"{date}_{config_hash}.json"

    def get(
        self, ticker: str, date: str, config_hash: str
    ) -> Optional[Tuple[Dict, str]]:
        """Retrieve cached (state, signal) if available."""
        path = self._cache_path(ticker, date, config_hash)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["state"], data["signal"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt cache entry at {path}: {e}")
            return None

    def put(
        self, ticker: str, date: str, config_hash: str, state: Dict, signal: str
    ) -> None:
        """Store a (state, signal) result in the cache."""
        path = self._cache_path(ticker, date, config_hash)

        # Extract serializable subset of state (skip non-JSON-serializable objects)
        serializable_state = _extract_serializable_state(state)

        data = {"state": serializable_state, "signal": signal}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def config_hash(config: Dict) -> str:
        """Generate a hash from the config that affects analysis output.

        Hashes LLM provider/model settings and analyst configuration
        so that different configs get different cache entries.
        """
        relevant_keys = [
            "llm_provider",
            "deep_think_llm",
            "quick_think_llm",
            "max_debate_rounds",
            "max_risk_discuss_rounds",
        ]
        hash_input = {k: config.get(k, "") for k in relevant_keys}
        raw = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _extract_serializable_state(state: Dict) -> Dict:
    """Extract JSON-serializable fields from a TradingAgentsGraph state."""
    serializable = {}
    keys_to_extract = [
        "company_of_interest",
        "trade_date",
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_plan",
        "trader_investment_plan",
        "final_trade_decision",
    ]
    for key in keys_to_extract:
        if key in state:
            serializable[key] = state[key]

    # Extract debate states if present
    for debate_key in ["investment_debate_state", "risk_debate_state"]:
        if debate_key in state and isinstance(state[debate_key], dict):
            serializable[debate_key] = {
                k: v
                for k, v in state[debate_key].items()
                if isinstance(v, (str, list, dict, int, float, bool, type(None)))
            }

    return serializable
