"""
Konfiguracja kosztów transakcyjnych dla silnika backtesting.py.

Symuluje warunki Fusion Markets — Raw Spread Account:
- Prowizja stała per lot ($2.25/side)
- Spread (dodawany do ceny wejścia/wyjścia)
- Poślizg cenowy (slippage)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradingCosts:
    """
    Struktura kosztów transakcyjnych Fusion Markets.

    Atrybuty:
        commission_per_side: Prowizja per side per lot (USD).
        spread_points: Symulowany spread w punktach cenowych.
        slippage_points: Poślizg w punktach cenowych.
        lot_size: Rozmiar pozycji w lotach.
    """

    commission_per_side: float = 2.25
    spread_points: float = 0.05
    slippage_points: float = 0.01
    lot_size: float = 0.01

    @property
    def total_cost_per_trade(self) -> float:
        """
        Łączny koszt round-turn jednej transakcji w punktach cenowych.

        Składa się z:
        - Prowizja: 2x commission_per_side (buy + sell)
        - Spread: spread_points (płacony przy wejściu)
        - Slippage: 2x slippage_points (wejście + wyjście)
        """
        return self.spread_points + (2 * self.slippage_points)

    @property
    def commission_round_turn(self) -> float:
        """Prowizja round-turn w USD (buy + sell)."""
        return 2 * self.commission_per_side * self.lot_size

    def log_costs(self) -> None:
        """Loguje podsumowanie kosztów transakcyjnych."""
        logger.info("=" * 50)
        logger.info("KOSZTY TRANSAKCYJNE (Fusion Markets Raw Spread)")
        logger.info("=" * 50)
        logger.info("  Prowizja per side:    $%.2f / lot", self.commission_per_side)
        logger.info("  Prowizja round-turn:  $%.4f (%.2f lot)", self.commission_round_turn, self.lot_size)
        logger.info("  Spread:               $%.2f (punkty cenowe)", self.spread_points)
        logger.info("  Slippage:             $%.2f (punkty cenowe)", self.slippage_points)
        logger.info(
            "  Koszt per trade (bez prowizji): $%.2f pkt",
            self.total_cost_per_trade,
        )


def calculate_commission_pct(
    commission_per_side: float,
    lot_size: float,
    typical_price: float,
    contract_size: float = 100.0,
) -> float:
    """
    Przelicza stałą prowizję ($2.25/side/lot) na procent wartości transakcji.

    Backtesting.py wymaga prowizji procentowej (commission=0.001 = 0.1%).
    Przeliczamy stałą kwotę na procent od wartości pozycji.

    Args:
        commission_per_side: Prowizja per side w USD.
        lot_size: Rozmiar pozycji w lotach.
        typical_price: Typowa cena instrumentu (np. 2600 dla złota).
        contract_size: Rozmiar kontraktu (100 oz dla XAUUSD).

    Returns:
        Prowizja jako ułamek dziesiętny (np. 0.0001 = 0.01%).
    """
    # Wartość nominalna pozycji = cena * lot * contract_size
    # Dla 0.01 lota XAUUSD przy cenie 2600: 2600 * 0.01 * 100 = $2,600
    position_value = typical_price * lot_size * contract_size

    if position_value == 0:
        return 0.0

    # Prowizja jako % wartości pozycji (per side)
    commission_pct = commission_per_side * lot_size / position_value

    logger.info(
        "Prowizja przeliczona: $%.2f/side/lot -> %.4f%% "
        "(przy cenie $%.0f, lot=%.2f, contract=%d oz)",
        commission_per_side,
        commission_pct * 100,
        typical_price,
        lot_size,
        contract_size,
    )

    return commission_pct
