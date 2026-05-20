"""
Wskaźniki techniczne — momentum i trend.

Generuje RSI, MACD (linia, sygnał, histogram) i ADX
z wykorzystaniem biblioteki `ta` (pure Python, brak zależności od numba).
"""

from __future__ import annotations

import logging

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

from config.settings import TechnicalConfig

logger = logging.getLogger(__name__)


def add_technical_indicators(
    df: pd.DataFrame,
    config: TechnicalConfig | None = None,
) -> pd.DataFrame:
    """
    Dodaje wskaźniki techniczne (momentum/trend) do DataFrame'a.

    Generowane cechy:
        - rsi_{period}           — Relative Strength Index
        - macd                   — linia MACD
        - macd_signal            — linia sygnału MACD
        - macd_histogram         — histogram MACD (macd - signal)
        - adx_{period}           — Average Directional Index
        - dmp_{period}           — Directional Movement Plus (+DI)
        - dmn_{period}           — Directional Movement Minus (-DI)

    Args:
        df: DataFrame z kolumnami OHLCV.
        config: Parametry wskaźników (domyślnie TechnicalConfig).

    Returns:
        DataFrame z dodanymi kolumnami wskaźników technicznych.
    """
    if config is None:
        config = TechnicalConfig()

    # --- RSI ---
    rsi_indicator = RSIIndicator(close=df["close"], window=config.rsi_period)
    df[f"rsi_{config.rsi_period}"] = rsi_indicator.rsi()

    # --- MACD ---
    macd_indicator = MACD(
        close=df["close"],
        window_fast=config.macd_fast,
        window_slow=config.macd_slow,
        window_sign=config.macd_signal,
    )
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_histogram"] = macd_indicator.macd_diff()

    # --- ADX ---
    adx_indicator = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=config.adx_period,
    )
    df[f"adx_{config.adx_period}"] = adx_indicator.adx()
    df[f"dmp_{config.adx_period}"] = adx_indicator.adx_pos()
    df[f"dmn_{config.adx_period}"] = adx_indicator.adx_neg()

    logger.info(
        "Dodano wskaźniki techniczne: RSI(%d), MACD(%d/%d/%d), ADX(%d).",
        config.rsi_period,
        config.macd_fast,
        config.macd_slow,
        config.macd_signal,
        config.adx_period,
    )

    return df
