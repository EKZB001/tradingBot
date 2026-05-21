"""
Feature Engineering specyficzny dla HFT / danych tikowych.

Dobudowuje cechy HFT do istniejacych cech z datasetu HuggingFace:
- Tick Price Velocity (predkosc zmiany ceny w oknach N tikow)
- Micro-Volatility (odchylenie standardowe zwrotow w oknach krotkich)
- Spread Z-Score (znormalizowany spread — anomalia)
- Volume Surge (anomalia wolumenu)
- Cechy rezimu rynkowego: cykliczne kodowanie godziny/minuty, makro-zmiennosc

UWAGA: Cechy z datasetu (SMA, EMA, RSI, MACD, BB etc.) pozostaja bez zmian.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import HFTConfig

logger = logging.getLogger(__name__)


def add_hft_features(
    df: pd.DataFrame,
    config: HFTConfig | None = None,
) -> pd.DataFrame:
    """
    Dobudowuje cechy HFT do DataFrame z danymi tikowymi.

    Zbiera wszystkie nowe kolumny w slowniku i laczy je
    jednym pd.concat(axis=1) — unika fragmentacji DataFrame.

    Args:
        df: DataFrame z danymi (musi zawierac 'mid_price').
        config: Konfiguracja HFT (okna, parametry).

    Returns:
        DataFrame z dodanymi kolumnami cech HFT.
    """
    config = config or HFTConfig()

    if "mid_price" not in df.columns:
        raise ValueError(
            "Brak kolumny 'mid_price'. Uruchom _compute_mid_price() z data_loader."
        )

    df = df.copy()
    new_cols: dict[str, np.ndarray] = {}

    _collect_velocity_features(df, config.velocity_windows, new_cols)
    _collect_micro_volatility(df, config.micro_vol_windows, new_cols)
    _collect_spread_zscore(df, new_cols)
    _collect_volume_surge(df, new_cols)
    _collect_time_regime_features(df, new_cols)
    _collect_macro_volatility(df, config.macro_vol_window, new_cols)

    # Jedno przypisanie — brak fragmentacji DataFrame
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    logger.info("Dodano %d cech HFT do DataFrame.", len(new_cols))
    return df


# ======================================================================
# Tick Price Velocity
# ======================================================================

def _collect_velocity_features(
    df: pd.DataFrame,
    windows: tuple[int, ...],
    out: dict[str, np.ndarray],
) -> None:
    """
    Predkosc zmiany ceny w oknach N tikow.

    velocity_N = (mid_price - mid_price_lagged_N) / N
    Mierzy tempo ruchu cenowego — kluczowa cecha scalpingowa.
    """
    mid = df["mid_price"].values

    for w in windows:
        velocity = np.empty_like(mid, dtype=np.float32)
        velocity[:w] = np.nan
        velocity[w:] = ((mid[w:] - mid[:-w]) / w).astype(np.float32)
        out[f"tick_velocity_{w}"] = velocity


# ======================================================================
# Micro-Volatility
# ======================================================================

def _collect_micro_volatility(
    df: pd.DataFrame,
    windows: tuple[int, ...],
    out: dict[str, np.ndarray],
) -> None:
    """
    Mikro-zmiennosc — odchylenie standardowe zwrotow procentowych
    w krotkich oknach tikowych.

    Wysoka wartosc = rynek niestabilny, niski sygnal/szum.
    """
    mid = df["mid_price"]
    pct_change = mid.pct_change()

    for w in windows:
        out[f"micro_vol_{w}"] = (
            pct_change.rolling(w, min_periods=w).std().values.astype(np.float32)
        )


# ======================================================================
# Spread Z-Score
# ======================================================================

def _collect_spread_zscore(
    df: pd.DataFrame,
    out: dict[str, np.ndarray],
    window: int = 100,
) -> None:
    """
    Znormalizowany spread — ile odchylen standardowych od sredniej.

    Wysoki Z-Score = anomalnie szeroki spread (ryzyko).
    """
    if "spread" not in df.columns:
        return

    spread = df["spread"]
    rolling_mean = spread.rolling(window, min_periods=window).mean()
    rolling_std = spread.rolling(window, min_periods=window).std()

    # Zabezpieczenie przed dzieleniem przez zero
    safe_std = rolling_std.replace(0, np.nan)
    out["spread_zscore"] = ((spread - rolling_mean) / safe_std).values.astype(np.float32)


# ======================================================================
# Volume Surge
# ======================================================================

def _collect_volume_surge(
    df: pd.DataFrame,
    out: dict[str, np.ndarray],
    window: int = 100,
) -> None:
    """
    Anomalia wolumenu — stosunek biezacego wolumenu do sredniej krocacej.

    > 2.0 = nadzwyczajny wolumen (potencjalny ruch cenowy).
    """
    if "volume" not in df.columns:
        return

    vol = df["volume"].astype(np.float32)
    rolling_mean = vol.rolling(window, min_periods=window).mean()

    # Zabezpieczenie przed dzieleniem przez zero
    safe_mean = rolling_mean.replace(0, np.nan)
    out["vol_surge"] = (vol / safe_mean).values.astype(np.float32)


# ======================================================================
# Time Regime Features (kodowanie cykliczne)
# ======================================================================

def _collect_time_regime_features(
    df: pd.DataFrame,
    out: dict[str, np.ndarray],
) -> None:
    """
    Cykliczne kodowanie czasu — godzina i minuta zakodowane sin/cos.

    Pozwala modelowi rozrozniac sesje (Azja, Europa, USA)
    bez tworzenia skokow na granicach (23:59 -> 00:00).
    """
    if "time" not in df.columns:
        return

    # Parsuj czas jesli nie jest datetime
    time_col = pd.to_datetime(df["time"], errors="coerce")

    hour = time_col.dt.hour + time_col.dt.minute / 60.0
    minute = time_col.dt.minute + time_col.dt.second / 60.0

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).values.astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).values.astype(np.float32)
    out["minute_sin"] = np.sin(2 * np.pi * minute / 60.0).values.astype(np.float32)
    out["minute_cos"] = np.cos(2 * np.pi * minute / 60.0).values.astype(np.float32)


# ======================================================================
# Macro-Volatility (rezimu rynkowy)
# ======================================================================

def _collect_macro_volatility(
    df: pd.DataFrame,
    window: int,
    out: dict[str, np.ndarray],
) -> None:
    """
    Makro-zmiennosc — odchylenie standardowe mid_price w duzym oknie.

    Informuje model czy rynek jest w fazie spokojnej (niski)
    czy eksplozywnej (wysoki) — kontekst dla decyzji scalpingowych.
    """
    mid = df["mid_price"]
    out["macro_vol"] = mid.rolling(window, min_periods=window).std().values.astype(np.float32)

