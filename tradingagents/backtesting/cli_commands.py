"""Interactive CLI prompts and progress display for backtesting."""

import questionary
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box

from .config import BacktestConfig
from .runner import BacktestRunner
from .report import BacktestReport

console = Console()

STYLE = questionary.Style(
    [
        ("text", "fg:green"),
        ("highlighted", "noinherit"),
    ]
)


def get_backtest_date(prompt_text: str) -> str:
    """Prompt for a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        prompt_text,
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=STYLE,
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def get_backtest_config() -> BacktestConfig:
    """Interactively gather backtest configuration from user."""
    console.print("\n[bold cyan]Backtest Configuration[/bold cyan]\n")

    # Ticker
    ticker = questionary.text(
        "Enter ticker symbol to backtest (e.g., AAPL, NVDA, SPY):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a ticker.",
        style=STYLE,
    ).ask()
    if not ticker:
        console.print("\n[red]No ticker provided. Exiting...[/red]")
        exit(1)
    ticker = ticker.strip().upper()

    # Date range
    start_date = get_backtest_date("Enter backtest start date (YYYY-MM-DD):")
    end_date = get_backtest_date("Enter backtest end date (YYYY-MM-DD):")

    # Initial cash
    cash_str = questionary.text(
        "Initial cash (default: 100000):",
        default="100000",
        style=STYLE,
    ).ask()
    initial_cash = float(cash_str) if cash_str else 100_000.0

    # Analysts
    analyst_choices = questionary.checkbox(
        "Select analysts to include:",
        choices=[
            questionary.Choice("Market Analyst", value="market", checked=True),
            questionary.Choice("Social Media Analyst", value="social", checked=True),
            questionary.Choice("News Analyst", value="news", checked=True),
            questionary.Choice("Fundamentals Analyst", value="fundamentals", checked=True),
        ],
        style=STYLE,
    ).ask()
    if not analyst_choices:
        analyst_choices = ["market", "social", "news", "fundamentals"]

    # Reflection
    enable_reflection = questionary.confirm(
        "Enable reflection/learning between trading days?",
        default=True,
        style=STYLE,
    ).ask()

    # Caching
    enable_cache = questionary.confirm(
        "Enable analyst output caching? (reduces cost on re-runs)",
        default=True,
        style=STYLE,
    ).ask()

    config = BacktestConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        selected_analysts=analyst_choices,
        enable_reflection=enable_reflection if enable_reflection is not None else True,
        cache_analyst_outputs=enable_cache if enable_cache is not None else True,
    )

    # Show summary
    _display_config_summary(config)

    confirm = questionary.confirm(
        "Proceed with this configuration?",
        default=True,
        style=STYLE,
    ).ask()
    if not confirm:
        console.print("[yellow]Backtest cancelled.[/yellow]")
        exit(0)

    return config


def _display_config_summary(config: BacktestConfig) -> None:
    """Show a summary table of the backtest config."""
    table = Table(title="Backtest Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Ticker", config.ticker)
    table.add_row("Start Date", config.start_date)
    table.add_row("End Date", config.end_date)
    table.add_row("Initial Cash", f"${config.initial_cash:,.2f}")
    table.add_row("Analysts", ", ".join(config.selected_analysts))
    table.add_row("Reflection", str(config.enable_reflection))
    table.add_row("Caching", str(config.cache_analyst_outputs))
    table.add_row("Tx Cost", f"{config.transaction_cost_bps} bps")
    table.add_row("Slippage", f"{config.slippage_bps} bps")

    console.print()
    console.print(table)
    console.print()


def run_backtest_cli() -> None:
    """Main entry point for the backtest CLI command."""
    config = get_backtest_config()

    console.print(f"\n[bold green]Starting backtest: {config.ticker}[/bold green]")
    console.print(f"Period: {config.start_date} to {config.end_date}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running backtest...", total=None)

        def on_progress(day_idx, total_days, date, signal):
            progress.update(
                task,
                total=total_days,
                completed=day_idx + 1,
                description=f"Day {day_idx + 1}/{total_days}: {date} -> {signal}",
            )

        runner = BacktestRunner(config)
        result = runner.run(progress_callback=on_progress)

    # Display results
    console.print()
    _display_results_table(result)

    console.print(f"\n[dim]Results saved to: {config.results_dir}/[/dim]")


def _display_results_table(result) -> None:
    """Display a rich table with backtest results."""
    table = Table(title="Backtest Results", box=box.DOUBLE_EDGE)
    table.add_column("Metric", style="cyan")
    table.add_column("Strategy", style="green", justify="right")
    table.add_column("Buy & Hold", style="yellow", justify="right")

    def fmt(val, key):
        if isinstance(val, float):
            if "return" in key or "drawdown" in key or "win_rate" in key:
                return f"{val:.2%}"
            return f"{val:.4f}"
        return str(val)

    for key in result.metrics:
        strat_val = fmt(result.metrics.get(key, "N/A"), key)
        bench_val = fmt(result.benchmark_metrics.get(key, "N/A"), key) if result.benchmark_metrics else "N/A"
        table.add_row(key, strat_val, bench_val)

    console.print(table)

    # Trade summary
    if result.daily_snapshots:
        final = result.daily_snapshots[-1]
        pnl = final.equity - result.initial_cash
        color = "green" if pnl >= 0 else "red"
        console.print(f"\nFinal Equity: [bold]${final.equity:,.2f}[/bold]")
        console.print(f"Total P&L: [{color}]${pnl:,.2f}[/{color}]")
        console.print(f"Total Trades: {len(result.trades)}")
        console.print(f"Failed Dates: {len(result.failed_dates)}")
        console.print(f"Elapsed: {result.elapsed_seconds:.1f}s")
