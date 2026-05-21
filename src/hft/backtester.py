"""
Wektorowy Tick Backtester — symulacja strategii na danych tikowych.

Uzywa danych z gold_test.csv + predykcji modelu do obliczenia:
- PnL (Profit and Loss) z uwzglednieniem kosztow spreadu i prowizji
- Win Rate, Profit Factor, sredni trade
- Max Drawdown

Roznica vs backtesting.py:
- Operuje na tikach (nie swieckach OHLCV)
- Symuluje SL/TP na poziomie tiku
- Brak biblioteki backtrader — czysta implementacja numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import HFTConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Wyniki backtestingu tikowego."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    final_equity: float = 0.0

    def __str__(self) -> str:
        return (
            f"\n{'=' * 50}\n"
            f"  TICK BACKTEST RESULTS\n"
            f"{'=' * 50}\n"
            f"  Total Trades:   {self.total_trades}\n"
            f"  Win Rate:       {self.win_rate:.1f}%\n"
            f"  Profit Factor:  {self.profit_factor:.2f}\n"
            f"  Total PnL:      ${self.total_pnl:.2f}\n"
            f"  Avg Trade PnL:  ${self.avg_trade_pnl:.2f}\n"
            f"  Gross Profit:   ${self.gross_profit:.2f}\n"
            f"  Gross Loss:     ${self.gross_loss:.2f}\n"
            f"  Max Drawdown:   ${self.max_drawdown:.2f}\n"
            f"  Final Equity:   ${self.final_equity:.2f}\n"
            f"{'=' * 50}"
        )


class TickBacktester:
    """
    Wektorowy backtester na danych tikowych.

    Symuluje strategie BUY-only z SL/TP na kazdym tikowym sygnale:
    - Sygnal 1 -> otworz BUY
    - SL hit -> zamknij ze strata
    - TP hit -> zamknij z zyskiem
    - Timeout (max_holding_ticks) -> zamknij po biezacej cenie

    Koszty: spread + prowizja per lot.
    """

    def __init__(self, config: HFTConfig | None = None) -> None:
        self.config = config or HFTConfig()

    def run(
        self,
        mid_prices: np.ndarray,
        predictions: np.ndarray,
        volume: float = 0.01,
    ) -> BacktestResult:
        """
        Uruchamia backtest na danych tikowych.

        Args:
            mid_prices: Tablica mid_price (float64).
            predictions: Predykcje modelu (0 lub 1).
            volume: Wolumen pozycji w lotach.

        Returns:
            BacktestResult z metrykami.
        """
        n = len(mid_prices)

        if len(predictions) != n:
            raise ValueError(
                f"Niezgodnosc dlugosci: mid_prices={n}, predictions={len(predictions)}"
            )

        tp_delta = self.config.tp_pips * self.config.pip_size
        sl_delta = self.config.sl_pips * self.config.pip_size
        max_hold = self.config.max_holding_ticks

        # Koszt spreadu w $ per lot (XAUUSD: 1 lot = 100 oz)
        spread_cost = self.config.spread_cost_pips * self.config.pip_size * 100 * volume
        commission = self.config.commission_per_lot * volume

        logger.info(
            "Backtest: %d tikow, TP=%.2f pips, SL=%.2f pips, "
            "spread_cost=$%.4f, commission=$%.4f per trade",
            n, self.config.tp_pips, self.config.sl_pips,
            spread_cost, commission,
        )

        # Symulacja trade-by-trade
        trades_pnl: list[float] = []
        equity = self.config.initial_cash
        peak_equity = equity
        max_dd = 0.0
        i = 0

        while i < n - max_hold:
            if predictions[i] != 1:
                i += 1
                continue

            # Otworz pozycje BUY
            entry_price = mid_prices[i]
            tp_price = entry_price + tp_delta
            sl_price = entry_price - sl_delta

            # Szukaj wyjscia w oknie max_hold
            exit_price = entry_price  # domyslnie timeout
            for j in range(1, max_hold + 1):
                idx = i + j
                if idx >= n:
                    break

                price = mid_prices[idx]

                if price >= tp_price:
                    exit_price = tp_price
                    break
                elif price <= sl_price:
                    exit_price = sl_price
                    break
            else:
                # Timeout — zamknij po aktualnej cenie
                exit_price = mid_prices[min(i + max_hold, n - 1)]

            # PnL w dolarach (XAUUSD: 1 lot = 100 oz, wartosc punktu = $1 per 0.01 per lot)
            # Dla 0.01 lota: wartosc pipsa = $0.01
            price_diff = exit_price - entry_price
            trade_pnl = price_diff * 100 * volume - spread_cost - commission

            trades_pnl.append(trade_pnl)
            equity += trade_pnl

            # Drawdown tracking
            peak_equity = max(peak_equity, equity)
            dd = peak_equity - equity
            max_dd = max(max_dd, dd)

            # Przeskocz poza czas trwania pozycji (nie otwieramy nowej w trakcie)
            i = min(i + max_hold, idx + 1) if 'idx' in dir() else i + max_hold

        # Oblicz metryki
        result = self._compute_metrics(trades_pnl, max_dd)
        result.final_equity = equity

        logger.info(str(result))
        return result

    def _compute_metrics(
        self,
        trades_pnl: list[float],
        max_dd: float,
    ) -> BacktestResult:
        """Oblicza metryki backtestu z listy PnL poszczegolnych transakcji."""
        if not trades_pnl:
            return BacktestResult()

        pnl_arr = np.array(trades_pnl)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]

        return BacktestResult(
            total_trades=len(pnl_arr),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=float(pnl_arr.sum()),
            gross_profit=float(wins.sum()) if len(wins) > 0 else 0.0,
            gross_loss=float(losses.sum()) if len(losses) > 0 else 0.0,
            max_drawdown=max_dd,
            avg_trade_pnl=float(pnl_arr.mean()),
            win_rate=len(wins) / len(pnl_arr) * 100 if len(pnl_arr) > 0 else 0.0,
            profit_factor=(
                abs(wins.sum() / losses.sum())
                if len(losses) > 0 and losses.sum() != 0
                else float("inf")
            ),
        )
