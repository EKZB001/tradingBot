"""
Loader danych z pliku Parquet do formatu wymaganego przez backtesting.py.

Wczytuje surowe dane OHLCV (nie przeskalowane features!),
mapuje kolumny i waliduje kompletność.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_backtest_data(path: Path) -> pd.DataFrame:
    """
    Wczytuje dane OHLCV z pliku Parquet i przygotowuje je
    do użycia przez silnik backtesting.py.

    Backtesting.py wymaga DataFrame z kolumnami:
    'Open', 'High', 'Low', 'Close', 'Volume' (wielkimi literami)
    i indeksem DatetimeIndex (timezone-naive).

    Args:
        path: Ścieżka do pliku .parquet z surowymi danymi OHLCV.

    Returns:
        DataFrame gotowy do podania do Backtest().

    Raises:
        FileNotFoundError: Gdy plik nie istnieje.
        ValueError: Gdy brakuje wymaganych kolumn lub dane zawierają NaN.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Plik danych nie istnieje: {path}. "
            f"Upewnij sie, ze Faza 1 (pipeline.py) zostala uruchomiona."
        )

    logger.info("Wczytywanie danych z: %s", path)
    df = pd.read_parquet(path, engine="pyarrow")

    # Walidacja wymaganych kolumn OHLCV
    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Brakujące kolumny w pliku: {missing}. "
            f"Dostępne: {list(df.columns)}"
        )

    # Mapowanie nazw kolumn na format backtesting.py (PascalCase)
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    # Wybranie tylko kolumn OHLCV (odrzucenie ewentualnych dodatkowych)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Konwersja indeksu timezone-aware -> naive (wymóg backtesting.py)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        logger.info("Usunięto informację o strefie czasowej z indeksu.")

    # Walidacja: brak NaN w danych OHLCV
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_per_col = df.isna().sum()
        logger.warning(
            "Znaleziono %d wartości NaN w danych OHLCV:\n%s",
            nan_count,
            nan_per_col[nan_per_col > 0].to_string(),
        )
        df = df.dropna()
        logger.info("Usunięto wiersze z NaN. Pozostało: %d wierszy.", len(df))

    logger.info(
        "Załadowano dane: %d barek, zakres: %s -> %s",
        len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
    )

    return df
