"""
Cechy zmienności (Volatility Features).

Generuje ATR, znormalizowaną szerokość wstęg Bollingera,
rolling StdDev logarytmicznych stóp zwrotu oraz estymator Garman-Klass.
Wykorzystuje bibliotekę `ta` (pure Python, brak zależności od numba).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange, BollingerBands

from config.settings import VolatilityConfig

logger = logging.getLogger(__name__)


def add_volatility_features(
    df: pd.DataFrame,
    config: VolatilityConfig | None = None,
) -> pd.DataFrame:
    """
    Dodaje miary zmienności do DataFrame'a.

    Generowane cechy:
        - atr_{period}           — Average True Range
        - bb_width_norm          — szerokość Bollingera znormalizowana do ceny
        - bb_pct_b               — %B (pozycja ceny w wstęgach)
        - log_return_std_{win}   — rolling StdDev log returns
        - garman_klass_{win}     — estymator zmienności Garman-Klass

    Args:
        df: DataFrame z kolumnami OHLCV.
        config: Parametry zmienności (domyślnie VolatilityConfig).

    Returns:
        DataFrame z dodanymi kolumnami zmienności.
    """
    if config is None:
        config = VolatilityConfig()

    _add_atr(df, config.atr_period)
    _add_bollinger(df, config.bb_period, config.bb_std)
    _add_log_return_std(df, config.volatility_window)
    _add_garman_klass(df, config.volatility_window)

    logger.info(
        "Dodano cechy zmienności: ATR(%d), BB(%d, %.1fσ), StdDev(%d), Garman-Klass(%d).",
        config.atr_period,
        config.bb_period,
        config.bb_std,
        config.volatility_window,
        config.volatility_window,
    )

    return df


def _add_atr(df: pd.DataFrame, period: int) -> None:
    """Oblicza Average True Range."""
    atr_indicator = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=period,
    )
    df[f"atr_{period}"] = atr_indicator.average_true_range()


def _add_bollinger(df: pd.DataFrame, period: int, std: float) -> None:
    """
    Oblicza znormalizowaną szerokość wstęg Bollingera i wskaźnik %B.

    - bb_width_norm = (upper - lower) / close — zakres wstęg względem ceny
    - bb_pct_b = (close - lower) / (upper - lower) — pozycja ceny w wstęgach
    """
    bb_indicator = BollingerBands(
        close=df["close"],
        window=period,
        window_dev=std,
    )

    bb_upper = bb_indicator.bollinger_hband()
    bb_lower = bb_indicator.bollinger_lband()
    band_width = bb_upper - bb_lower

    # Normalizacja szerokości do ceny zamknięcia
    df["bb_width_norm"] = band_width / df["close"]

    # Pozycja ceny w wstęgach (%B) — 0 = dolna, 1 = górna
    df["bb_pct_b"] = bb_indicator.bollinger_pband()


def _add_log_return_std(df: pd.DataFrame, window: int) -> None:
    """
    Oblicza rolling StdDev logarytmicznych stóp zwrotu.

    Miara historycznej zmienności — standardowa w modelach kwantowych.
    """
    log_returns = np.log(df["close"] / df["close"].shift(1))
    df[f"log_return_std_{window}"] = log_returns.rolling(window=window).std()


def _add_garman_klass(df: pd.DataFrame, window: int) -> None:
    """
    Oblicza estymator zmienności Garman-Klass (1980).

    Wykorzystuje pełne dane OHLC — efektywniejszy niż close-to-close.
    Formuła: GK = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    gk_single = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    # Rolling mean dla wygładzenia
    df[f"garman_klass_{window}"] = gk_single.rolling(window=window).mean()
