"""
Zmienne opóźnione (Lagged Features).

Generuje logarytmiczne i procentowe stopy zwrotu z N poprzednich okresów
oraz opóźnione wartości ceny zamknięcia i wolumenu.
Wszystkie operacje w pełni wektoryzowane (numpy/pandas).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import LaggedConfig

logger = logging.getLogger(__name__)


def add_lagged_features(
    df: pd.DataFrame,
    config: LaggedConfig | None = None,
) -> pd.DataFrame:
    """
    Dodaje zmienne opóźnione do DataFrame'a.

    Generowane cechy (dla każdego okresu p w `config.periods`):
        - log_return_{p}   — logarytmiczna stopa zwrotu z p okresów
        - pct_return_{p}   — procentowa stopa zwrotu z p okresów
        - close_lag_{p}    — opóźniona cena zamknięcia o p okresów
        - volume_lag_{p}   — opóźniony wolumen o p okresów

    Args:
        df: DataFrame z kolumnami 'close' i 'volume'.
        config: Parametry (domyślnie LaggedConfig z defaultami).

    Returns:
        DataFrame z dodanymi kolumnami lag features.
    """
    if config is None:
        config = LaggedConfig()

    close = df["close"]
    volume = df["volume"]

    for period in config.periods:
        # Logarytmiczna stopa zwrotu: ln(close_t / close_{t-p})
        df[f"log_return_{period}"] = np.log(close / close.shift(period))

        # Procentowa stopa zwrotu
        df[f"pct_return_{period}"] = close.pct_change(periods=period)

        # Opóźniona cena zamknięcia
        df[f"close_lag_{period}"] = close.shift(period)

        # Opóźniony wolumen
        df[f"volume_lag_{period}"] = volume.shift(period)

    features_count = len(config.periods) * 4
    logger.info(
        "Dodano %d cech opóźnionych (periody: %s).",
        features_count,
        list(config.periods),
    )

    return df
