"""
Cechy mikro-struktury cenowej (Price Action Features).

Analizuje anatomię świecy: rozmiar korpusu, długość knotów,
kierunek i znormalizowany zakres — kluczowe sygnały price action.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Epsilon zapobiegający dzieleniu przez zero przy świecach o zerowym zakresie (doji)
_EPSILON = 1e-10


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje cechy mikro-struktury cenowej do DataFrame'a.

    Generowane cechy:
        - body_ratio        — |close - open| / (high - low) — znormalizowany rozmiar korpusu
        - upper_wick_ratio  — (high - max(open, close)) / (high - low) — górny knot
        - lower_wick_ratio  — (min(open, close) - low) / (high - low) — dolny knot
        - candle_direction  — 1 (bullish), -1 (bearish), 0 (doji)
        - range_norm        — (high - low) / close — znormalizowany zakres świecy

    Args:
        df: DataFrame z kolumnami: open, high, low, close.

    Returns:
        DataFrame z dodanymi kolumnami price action.
    """
    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Zakres świecy (high - low) z zabezpieczeniem epsilon
    candle_range = high - low + _EPSILON

    # --- Znormalizowany rozmiar korpusu ---
    body_size = np.abs(close - open_price)
    df["body_ratio"] = body_size / candle_range

    # --- Górny knot (upper wick) ---
    upper_boundary = np.maximum(open_price, close)
    df["upper_wick_ratio"] = (high - upper_boundary) / candle_range

    # --- Dolny knot (lower wick) ---
    lower_boundary = np.minimum(open_price, close)
    df["lower_wick_ratio"] = (lower_boundary - low) / candle_range

    # --- Kierunek świecy ---
    # 1 = bullish (close > open), -1 = bearish, 0 = doji (body < epsilon)
    df["candle_direction"] = np.sign(close - open_price)

    # --- Znormalizowany zakres świecy ---
    df["range_norm"] = (high - low) / close

    logger.info("Dodano 5 cech price action (body, wicks, direction, range).")

    return df
