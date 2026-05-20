"""
Preprocessing danych pod Machine Learning.

Czyszczenie NaN, skalowanie cech (RobustScaler / StandardScaler)
i zapis/odczyt obiektów scalerów do joblib — dla spójności
między treningiem a inferencją na żywo (Faza 3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

# Typ scalera — albo RobustScaler (odporny na outliers), albo StandardScaler
ScalerType = Literal["robust", "standard"]

# Mapowanie nazw na klasy scalerów
_SCALER_MAP = {
    "robust": RobustScaler,
    "standard": StandardScaler,
}

# Kolumny wykluczone ze skalowania (targety, indeks, flagi)
_COLUMNS_EXCLUDED_FROM_SCALING = {
    "target_cls",
    "target_reg",
    "candle_direction",
    "session_asia",
    "session_london",
    "session_ny",
    "session_overlap",
}


class Preprocessor:
    """
    Preprocessor danych — czyszczenie NaN i skalowanie cech.

    Skaluje wyłącznie kolumny numeryczne (features), pomijając targety,
    flagi binarne i indeks czasowy.
    """

    def clean_nans(
        self,
        df: pd.DataFrame,
        strategy: Literal["drop", "ffill"] = "drop",
    ) -> pd.DataFrame:
        """
        Czyści wartości NaN z DataFrame'a.

        NaN powstają naturalnie z:
        - Rolling indicators (początek serii)
        - Forward-looking targets (koniec serii)
        - Lagged variables (początek serii)

        Args:
            df: DataFrame do oczyszczenia.
            strategy: 'drop' (usunięcie wierszy) lub 'ffill' (forward fill + backfill).

        Returns:
            Oczyszczony DataFrame.
        """
        rows_before = len(df)

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "ffill":
            df = df.ffill().bfill()
        else:
            raise ValueError(f"Nieznana strategia: '{strategy}'. Użyj 'drop' lub 'ffill'.")

        rows_after = len(df)
        rows_removed = rows_before - rows_after
        pct_removed = (rows_removed / rows_before * 100) if rows_before > 0 else 0.0

        logger.info(
            "Czyszczenie NaN (strategia='%s'): usunięto %d / %d wierszy (%.1f%%).",
            strategy,
            rows_removed,
            rows_before,
            pct_removed,
        )

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        scaler_type: ScalerType = "robust",
        exclude_columns: set[str] | None = None,
    ) -> tuple[pd.DataFrame, RobustScaler | StandardScaler, list[str]]:
        """
        Dopasowuje scaler i transformuje cechy numeryczne.

        Args:
            df: DataFrame z cechami.
            scaler_type: Typ scalera ('robust' lub 'standard').
            exclude_columns: Kolumny do pominięcia przy skalowaniu.

        Returns:
            Tuple: (przeskalowany DataFrame, obiekt scalera, lista skalowanych kolumn).
        """
        excluded = _COLUMNS_EXCLUDED_FROM_SCALING.copy()
        if exclude_columns:
            excluded.update(exclude_columns)

        # Skaluj tylko kolumny numeryczne (float/int), pomijając excluded
        scale_columns = [
            col
            for col in df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
            if col not in excluded
        ]

        scaler_class = _SCALER_MAP.get(scaler_type)
        if scaler_class is None:
            raise ValueError(
                f"Nieznany typ scalera: '{scaler_type}'. Dostępne: {list(_SCALER_MAP.keys())}"
            )

        scaler = scaler_class()
        df[scale_columns] = scaler.fit_transform(df[scale_columns])

        logger.info(
            "Skalowanie cech (%s): %d kolumn przeskalowanych.",
            scaler_type.upper(),
            len(scale_columns),
        )

        return df, scaler, scale_columns

    def transform(
        self,
        df: pd.DataFrame,
        scaler: RobustScaler | StandardScaler,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Transformuje cechy przy użyciu wcześniej dopasowanego scalera.

        Używane w Fazie 3 (inference) — ładujemy scaler z joblib
        i aplikujemy na nowych danych.

        Args:
            df: Nowe dane do przeskalowania.
            scaler: Wcześniej dopasowany scaler.
            columns: Lista kolumn do przeskalowania.

        Returns:
            DataFrame z przeskalowanymi kolumnami.
        """
        available_columns = [col for col in columns if col in df.columns]

        if len(available_columns) < len(columns):
            missing = set(columns) - set(available_columns)
            logger.warning(
                "Brak %d kolumn w nowych danych (pominięte): %s",
                len(missing),
                missing,
            )

        df[available_columns] = scaler.transform(df[available_columns])
        return df

    @staticmethod
    def save_scaler(
        scaler: RobustScaler | StandardScaler,
        path: Path,
        columns: list[str] | None = None,
    ) -> None:
        """
        Zapisuje obiekt scalera (i opcjonalnie listę kolumn) do pliku joblib.

        Args:
            scaler: Dopasowany scaler do zapisu.
            path: Ścieżka docelowa pliku .joblib.
            columns: Lista skalowanych kolumn (zapisywana razem ze scalerem).
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {"scaler": scaler, "columns": columns or []}
        joblib.dump(artifact, path)

        logger.info("Zapisano scaler -> %s", path)

    @staticmethod
    def load_scaler(path: Path) -> tuple[RobustScaler | StandardScaler, list[str]]:
        """
        Wczytuje scaler i listę kolumn z pliku joblib.

        Args:
            path: Ścieżka do pliku .joblib.

        Returns:
            Tuple: (obiekt scalera, lista skalowanych kolumn).

        Raises:
            FileNotFoundError: Gdy plik nie istnieje.
        """
        if not path.exists():
            raise FileNotFoundError(f"Plik scalera nie istnieje: {path}")

        artifact = joblib.load(path)
        logger.info("Wczytano scaler <- %s", path)

        return artifact["scaler"], artifact["columns"]
