"""
Generator raportu wynikowego z backtestu.

Parsuje obiekt stats z backtesting.py i formatuje profesjonalny raport
z kluczowymi metrykami: Sharpe, Drawdown, Win Rate, Profit Factor.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def print_backtest_report(
    stats,
    initial_cash: float = 10_000.0,
    output_path: Path | None = None,
) -> str:
    """
    Formatuje i wyświetla profesjonalny raport z backtestu.

    Args:
        stats: Obiekt stats zwrócony przez Backtest.run().
        initial_cash: Kapitał początkowy (backtesting.py nie eksportuje go w stats).
        output_path: Opcjonalna ścieżka do zapisu raportu (.txt).

    Returns:
        Sformatowany raport jako string.
    """
    # Ekstrakcja metryk z obiektu stats (klucze backtesting.py 0.6.x)
    start_value = stats.get("Start", "N/A")
    end_value = stats.get("End", "N/A")
    duration = stats.get("Duration", "N/A")
    final_equity = stats.get("Equity Final [$]", initial_cash)
    equity_peak = stats.get("Equity Peak [$]", initial_cash)

    # Zwroty
    net_profit = final_equity - initial_cash
    net_profit_pct = stats.get("Return [%]", 0)
    buy_hold_return = stats.get("Buy & Hold Return [%]", 0)
    cagr = stats.get("CAGR [%]", 0)

    # Drawdown
    max_drawdown_pct = stats.get("Max. Drawdown [%]", 0)
    avg_drawdown_pct = stats.get("Avg. Drawdown [%]", 0)

    # Wskaźniki ryzyka
    sharpe = stats.get("Sharpe Ratio", None)
    sortino = stats.get("Sortino Ratio", None)
    calmar = stats.get("Calmar Ratio", None)

    # Transakcje
    total_trades = stats.get("# Trades", 0)
    win_rate = stats.get("Win Rate [%]", 0)
    best_trade = stats.get("Best Trade [%]", 0)
    worst_trade = stats.get("Worst Trade [%]", 0)
    avg_trade = stats.get("Avg. Trade [%]", 0)
    profit_factor = stats.get("Profit Factor", None)
    sqn = stats.get("SQN", None)
    expectancy = stats.get("Expectancy [%]", None)
    exposure = stats.get("Exposure Time [%]", 0)

    # Pomocnicza funkcja formatowania NaN-safe
    def fmt(val, template="{:.4f}"):
        """Formatuje wartość lub zwraca N/A gdy NaN/None."""
        if val is None:
            return "N/A"
        try:
            import math
            if math.isnan(float(val)):
                return "N/A"
        except (TypeError, ValueError):
            return "N/A"
        return template.format(val)

    # Formatowanie raportu
    separator = "=" * 60
    report_lines = [
        "",
        separator,
        "  RAPORT BACKTESTU -- XAUUSD M5",
        separator,
        "",
        f"  Okres:           {start_value} -> {end_value}",
        f"  Czas trwania:    {duration}",
        f"  Ekspozycja:      {exposure:.1f}%",
        "",
        "  --- KAPITAL ---",
        f"  Kapital poczatkowy:   ${initial_cash:,.2f}",
        f"  Kapital koncowy:      ${final_equity:,.2f}",
        f"  Szczyt equity:        ${equity_peak:,.2f}",
        "",
        "  --- ZWROTY ---",
        f"  Zysk/Strata netto:    ${net_profit:,.2f} ({net_profit_pct:+.2f}%)",
        f"  Buy & Hold:           {buy_hold_return:+.2f}%",
        f"  CAGR:                 {cagr:+.2f}%",
        "",
        "  --- RYZYKO ---",
        f"  Max Drawdown:         {max_drawdown_pct:.2f}%",
        f"  Avg Drawdown:         {fmt(avg_drawdown_pct, '{:.2f}%')}",
        f"  Sharpe Ratio:         {fmt(sharpe)}",
        f"  Sortino Ratio:        {fmt(sortino)}",
        f"  Calmar Ratio:         {fmt(calmar)}",
        "",
        "  --- TRANSAKCJE ---",
        f"  Liczba transakcji:    {total_trades}",
        f"  Win Rate:             {fmt(win_rate, '{:.1f}%')}",
        f"  Avg Trade:            {fmt(avg_trade, '{:+.2f}%')}",
        f"  Best Trade:           {fmt(best_trade, '{:+.2f}%')}",
        f"  Worst Trade:          {fmt(worst_trade, '{:+.2f}%')}",
        f"  Profit Factor:        {fmt(profit_factor)}",
        f"  SQN:                  {fmt(sqn)}",
        f"  Expectancy:           {fmt(expectancy, '{:+.2f}%')}",
        "",
        separator,
    ]

    report = "\n".join(report_lines)

    # Wyświetlenie w loggerze
    for line in report_lines:
        if line.strip():
            logger.info(line)

    # Opcjonalny zapis do pliku
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info("Raport zapisany do: %s", output_path)

    return report


def extract_trade_log(stats) -> list[dict]:
    """
    Wyciąga listę indywidualnych transakcji z obiektu stats.

    Args:
        stats: Obiekt stats zwrócony przez Backtest.run().

    Returns:
        Lista słowników z danymi każdej transakcji.
    """
    trades = stats.get("_trades", None)
    if trades is None:
        logger.warning("Brak danych o indywidualnych transakcjach w stats.")
        return []

    trade_list = []
    for _, trade in trades.iterrows():
        trade_list.append({
            "entry_time": str(trade.get("EntryTime", "")),
            "exit_time": str(trade.get("ExitTime", "")),
            "entry_price": trade.get("EntryPrice", 0),
            "exit_price": trade.get("ExitPrice", 0),
            "pnl": trade.get("PnL", 0),
            "return_pct": trade.get("ReturnPct", 0),
            "size": trade.get("Size", 0),
            "duration": str(trade.get("Duration", "")),
        })

    logger.info("Wyeksportowano %d transakcji z logu.", len(trade_list))
    return trade_list
