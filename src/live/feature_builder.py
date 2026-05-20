"""
Replikacja pipeline'u Feature Engineering z Fazy 1 dla trybu live.

Pobiera bufor N swiec z MT5, generuje identyczny zestaw cech
jak pipeline.py (lagged, technical, volatility, calendar, price action),
skaluje je zapisanym RobustScalerem i zwraca wektor cech
dla najnowszej zamknietej swiece.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    LaggedConfig,
    TechnicalConfig,
    VolatilityConfig,
)
from src.features.calendar import add_calendar_features
from src.features.lagged import add_lagged_features
from src.features.price_action import add_price_action_features
from src.features.technical import add_technical_indicators
from src.features.volatility import add_volatility_features

logger = logging.getLogger(__name__)


class LiveFeatureBuilder:
    """
    Buduje wektor cech na zywo -- replika pipeline'u Fazy 1.

    Uzywa tych samych funkcji FE co pipeline.py, ale operuje
    na malym buforze swiec (100-200) zamiast pelnego zbioru historycznego.
    """

    def __init__(
        self,
        scaler_path: Path,
        feature_columns: list[str],
    ) -> None:
        """
        Inicjalizuje builder z zapisanym scalerem i lista kolumn.

        Args:
            scaler_path: Sciezka do artefaktu scalera (.joblib).
            feature_columns: Lista kolumn wymaganych przez model ML.
        """
        artifact = joblib.load(scaler_path)
        self._scaler = artifact["scaler"]
        self._scaler_columns = artifact["columns"]
        self._feature_columns = feature_columns
        logger.info(
            "Zaladowano scaler (%d kolumn) i konfiguracje features (%d kolumn).",
            len(self._scaler_columns),
            len(self._feature_columns),
        )

    def build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame | None:
        """
        Generuje pelny zestaw cech z surowych danych OHLCV.

        Wykonuje dokladnie te same kroki co pipeline.py:
        1. Lagged features (zwroty, opóźnione ceny)
        2. Technical indicators (RSI, MACD, ADX)
        3. Volatility features (ATR, Bollinger, Garman-Klass)
        4. Calendar features (sin/cos godzina/dzien, flagi sesji)
        5. Price action features (body ratio, wicks, direction)
        6. Skalowanie RobustScaler (identyczny obiekt z Fazy 1)

        Args:
            df_raw: DataFrame z kolumnami OHLCV i indeksem datetime.

        Returns:
            Jednowierszowy DataFrame z cechami dla ostatniej swiece,
            lub None jesli brakuje danych.
        """
        df = df_raw.copy()

        # 1. Feature Engineering -- ta sama kolejnosc co pipeline.py
        df = add_lagged_features(df, LaggedConfig())
        df = add_technical_indicators(df, TechnicalConfig())
        df = add_volatility_features(df, VolatilityConfig())
        df = add_calendar_features(df)
        df = add_price_action_features(df)

        # 2. Wyciagnij ostatni kompletny wiersz (najnowsza zamknieta swieca)
        last_row = df.iloc[[-1]].copy()

        # 3. Skalowanie -- uzyj zapisanego scalera z Fazy 1
        # Scaler byl fitowany na WSZYSTKICH kolumnach numerycznych (lacznie z OHLCV),
        # wiec musimy przekazac mu pelny zestaw kolumn, a POTEM wybrac kolumny modelu.
        scalable_cols = [c for c in self._scaler_columns if c in last_row.columns]
        if scalable_cols:
            last_row[scalable_cols] = self._scaler.transform(
                last_row[scalable_cols]
            )

        # 4. Sprawdz czy mamy wszystkie wymagane kolumny modelu
        missing = set(self._feature_columns) - set(last_row.columns)
        if missing:
            logger.error("Brakujace kolumny w features: %s", missing)
            return None

        # 5. Wyodrebnij tylko kolumny wymagane przez model
        feature_row = last_row[self._feature_columns]

        # 6. Sprawdz NaN w wymaganych kolumnach
        if feature_row.isna().any(axis=1).iloc[0]:
            nan_cols = feature_row.columns[feature_row.isna().iloc[0]].tolist()
            logger.warning(
                "NaN w cechach ostatniej swiece (brak wystarczajacego bufora): %s",
                nan_cols[:5],
            )
            return None

        return feature_row
