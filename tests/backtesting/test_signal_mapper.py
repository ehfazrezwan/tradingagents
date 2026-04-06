"""Tests for the SignalMapper class."""

import unittest

from tradingagents.backtesting.signal_mapper import SignalMapper, VALID_SIGNALS


class TestSignalMapper(unittest.TestCase):
    def setUp(self):
        self.allocation_map = {
            "BUY": 1.0,
            "OVERWEIGHT": 0.75,
            "HOLD": None,
            "UNDERWEIGHT": 0.25,
            "SELL": 0.0,
        }
        self.mapper = SignalMapper(self.allocation_map)

    def test_canonical_signals(self):
        for signal in VALID_SIGNALS:
            normalized = self.mapper.normalize_signal(signal)
            self.assertIn(normalized, VALID_SIGNALS)

    def test_case_insensitive(self):
        self.assertEqual(self.mapper.normalize_signal("buy"), "BUY")
        self.assertEqual(self.mapper.normalize_signal("Sell"), "SELL")
        self.assertEqual(self.mapper.normalize_signal("HOLD"), "HOLD")

    def test_whitespace_handling(self):
        self.assertEqual(self.mapper.normalize_signal("  BUY  "), "BUY")
        self.assertEqual(self.mapper.normalize_signal("OVER WEIGHT"), "OVERWEIGHT")
        self.assertEqual(self.mapper.normalize_signal("UNDER WEIGHT"), "UNDERWEIGHT")

    def test_aliases(self):
        self.assertEqual(self.mapper.normalize_signal("STRONG BUY"), "BUY")
        self.assertEqual(self.mapper.normalize_signal("STRONG SELL"), "SELL")
        self.assertEqual(self.mapper.normalize_signal("NEUTRAL"), "HOLD")

    def test_embedded_signal(self):
        self.assertEqual(
            self.mapper.normalize_signal("The recommendation is BUY based on analysis"),
            "BUY",
        )

    def test_unknown_defaults_to_hold(self):
        self.assertEqual(self.mapper.normalize_signal("MAYBE"), "HOLD")
        self.assertEqual(self.mapper.normalize_signal(""), "HOLD")
        self.assertEqual(self.mapper.normalize_signal("???"), "HOLD")

    def test_allocation_mapping(self):
        self.assertEqual(self.mapper.get_target_allocation("BUY"), 1.0)
        self.assertEqual(self.mapper.get_target_allocation("OVERWEIGHT"), 0.75)
        self.assertIsNone(self.mapper.get_target_allocation("HOLD"))
        self.assertEqual(self.mapper.get_target_allocation("UNDERWEIGHT"), 0.25)
        self.assertEqual(self.mapper.get_target_allocation("SELL"), 0.0)

    def test_allocation_normalizes_first(self):
        self.assertEqual(self.mapper.get_target_allocation("strong buy"), 1.0)
        self.assertIsNone(self.mapper.get_target_allocation("neutral"))


if __name__ == "__main__":
    unittest.main()
