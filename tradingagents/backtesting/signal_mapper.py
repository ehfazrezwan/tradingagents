import re
from typing import Dict, Optional


# Canonical signals and common aliases
_SIGNAL_ALIASES = {
    "BUY": "BUY",
    "STRONG BUY": "BUY",
    "STRONGBUY": "BUY",
    "OVERWEIGHT": "OVERWEIGHT",
    "OVER WEIGHT": "OVERWEIGHT",
    "HOLD": "HOLD",
    "NEUTRAL": "HOLD",
    "UNDERWEIGHT": "UNDERWEIGHT",
    "UNDER WEIGHT": "UNDERWEIGHT",
    "SELL": "SELL",
    "STRONG SELL": "SELL",
    "STRONGSELL": "SELL",
}

VALID_SIGNALS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}


class SignalMapper:
    """Maps trading signals to target portfolio allocation percentages."""

    def __init__(self, allocation_map: Dict[str, Optional[float]]):
        self.allocation_map = allocation_map

    def normalize_signal(self, raw_signal: str) -> str:
        """Normalize a raw signal string to one of the 5 canonical signals.

        Handles whitespace, casing, and common aliases. Falls back to HOLD
        if the signal cannot be parsed.
        """
        cleaned = re.sub(r"[^A-Z\s]", "", raw_signal.upper().strip())
        cleaned = " ".join(cleaned.split())  # collapse whitespace

        if cleaned in _SIGNAL_ALIASES:
            return _SIGNAL_ALIASES[cleaned]

        # Try to find a canonical signal embedded in the text
        for valid in VALID_SIGNALS:
            if valid in cleaned:
                return valid

        return "HOLD"

    def get_target_allocation(self, signal: str) -> Optional[float]:
        """Return the target allocation for a normalized signal.

        Returns None for HOLD (no position change).
        """
        normalized = self.normalize_signal(signal)
        return self.allocation_map.get(normalized)
