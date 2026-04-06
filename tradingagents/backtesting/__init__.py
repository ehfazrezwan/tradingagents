def __getattr__(name):
    if name == "BacktestConfig":
        from .config import BacktestConfig
        return BacktestConfig
    if name == "BacktestRunner":
        from .runner import BacktestRunner
        return BacktestRunner
    if name == "BacktestResult":
        from .results import BacktestResult
        return BacktestResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BacktestConfig", "BacktestRunner", "BacktestResult"]
