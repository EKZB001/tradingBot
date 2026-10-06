"""
Definicja targetu dla modelu HFT (Triple-Barrier Method).

Dla kazdego tiku sprawdza w oknie max_holding_ticks:
- Czy cena mid_price wzrosnie o tp_pips -> target = 1 (sygnal BUY)
- Czy cena mid_price spadnie o sl_pips -> target = 0 (brak sygnalu)
- Przekroczono max_holding -> target = 0 (timeout)

Implementacja wektoryzowana z uzyciem numpy dla wydajnosci.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import HFTConfig

logger = logging.getLogger(__name__)


def compute_triple_barrier_target(
    df: pd.DataFrame,
    config: HFTConfig | None = None,
) -> pd.DataFrame:
    """
    Wylicza target binarny metoda Triple-Barrier.

    Args:
        df: DataFrame z kolumna 'mid_price'.
        config: Konfiguracja HFT (tp_pips, sl_pips, max_holding_ticks).

    Returns:
        DataFrame z dodana kolumna 'target' (0 lub 1).
    """
    config = config or HFTConfig()

    if "mid_price" not in df.columns:
        raise ValueError("Brak kolumny 'mid_price' — uruchom compute_mid_price() z data_loader.")

    mid = df["mid_price"].values.astype(np.float64)
    n = len(mid)

    tp_delta = config.tp_pips * config.pip_size
    sl_delta = config.sl_pips * config.pip_size
    max_hold = config.max_holding_ticks

    logger.info(
        "Obliczanie Triple-Barrier target: TP=%.2f pips (%.4f), "
        "SL=%.2f pips (%.4f), max_hold=%d tikow, n=%d ...",
        config.tp_pips, tp_delta, config.sl_pips, sl_delta, max_hold, n,
    )

    target = _compute_barriers_vectorized(mid, tp_delta, sl_delta, max_hold)

    df = df.copy()
    df["target"] = target

    # Statystyki rozkladu klas
    n_valid = np.sum(~np.isnan(target))
    n_positive = np.sum(target == 1)
    n_negative = np.sum(target == 0)
    ratio = n_positive / n_negative if n_negative > 0 else float("inf")

    logger.info(
        "Target obliczony: 1=%d (%.1f%%), 0=%d (%.1f%%), ratio=%.4f, NaN=%d",
        n_positive, n_positive / n_valid * 100 if n_valid > 0 else 0,
        n_negative, n_negative / n_valid * 100 if n_valid > 0 else 0,
        ratio,
        np.sum(np.isnan(target)),
    )

    return df


def _compute_barriers_vectorized(
    mid: np.ndarray,
    tp_delta: float,
    sl_delta: float,
    max_hold: int,
) -> np.ndarray:
    """
    Wektorowa implementacja Triple-Barrier.

    Dla kazdego punktu i, sprawdza w oknie [i+1, i+max_hold]:
    - max(mid[i+1:i+max_hold+1]) >= mid[i] + tp_delta -> TP hit?
    - min(mid[i+1:i+max_hold+1]) <= mid[i] - sl_delta -> SL hit?
    - Jesli TP trafiony PIERWSZY -> target=1
    - Jesli SL trafiony PIERWSZY lub timeout -> target=0

    Uzywamy petli numpy z pre-alokacja (nie ma wektorowej alternatywy
    dla first-hit barrier w czystym numpy, ale numba mogloby przyspieszyc).
    """
    n = len(mid)
    target = np.full(n, np.nan, dtype=np.float64)

    # Ostatnie max_hold tikow nie moga miec pelnego okna — zostawic NaN
    for i in range(n - max_hold):
        entry_price = mid[i]
        tp_level = entry_price + tp_delta
        sl_level = entry_price - sl_delta

        window = mid[i + 1: i + 1 + max_hold]

        # Indeksy pierwszego trafienia TP i SL
        tp_hits = np.where(window >= tp_level)[0]
        sl_hits = np.where(window <= sl_level)[0]

        tp_first = tp_hits[0] if len(tp_hits) > 0 else max_hold + 1
        sl_first = sl_hits[0] if len(sl_hits) > 0 else max_hold + 1

        if tp_first <= sl_first and tp_first < max_hold:
            target[i] = 1.0
        else:
            target[i] = 0.0

    return target
