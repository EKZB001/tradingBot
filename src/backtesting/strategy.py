"""
Strategie backtestingowe -- SMA placeholder i ML prediction.

Zawiera dwie klasy:
- SMAStrategy: prosty SMA crossover (benchmark/placeholder z Fazy 2)
- MLPredictionStrategy: predykcje z wytrenowanego modelu XGBoost
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

logger = logging.getLogger(__name__)


# ============================================
# Strategia placeholder: SMA Crossover
# ============================================

class SMAStrategy(Strategy):
    """
    Prosty SMA crossover jako benchmark.

    Parametry:
        sma_fast: Okres krotkiej sredniej.
        sma_slow: Okres dlugiej sredniej.
        position_size_pct: Procent kapitalu na pozycje.
    """

    sma_fast = 10
    sma_slow = 50
    spread_points = 0.05
    slippage_points = 0.01
    position_size_pct = 0.99

    def init(self):
        """Inicjalizacja wskaznikow SMA."""
        self.sma_fast_line = self.I(
            _sma, self.data.Close, self.sma_fast, name=f"SMA({self.sma_fast})"
        )
        self.sma_slow_line = self.I(
            _sma, self.data.Close, self.sma_slow, name=f"SMA({self.sma_slow})"
        )

    def next(self):
        """Logika decyzyjna SMA crossover."""
        if crossover(self.sma_fast_line, self.sma_slow_line):
            if not self.position:
                self.buy(size=self.position_size_pct)
        elif crossover(self.sma_slow_line, self.sma_fast_line):
            if self.position:
                self.position.close()


# ============================================
# Strategia ML: Predykcje XGBoost
# ============================================

class MLPredictionStrategy(Strategy):
    """
    Strategia oparta na predykcjach modelu ML (XGBoost).

    Prekomputacja: W init() obliczamy predykcje dla WSZYSTKICH barow
    naraz (wektoryzowane). Wynik jest rejestrowany jako indicator
    via self.I(), co jest standardowym i bezpiecznym podejsciem
    w backtesting.py (brak lookahead bias).

    Parametry (ustawiane z poziomu Backtest):
        model_path: Sciezka do artefaktu modelu (.joblib).
        scaler_path: Sciezka do artefaktu scalera (.joblib).
        processed_data_path: Sciezka do przetworzonych danych (.parquet).
        position_size_pct: Procent kapitalu na pozycje.
    """

    # Sciezki do artefaktow (ustawiane z backtest.py)
    model_path = ""
    scaler_path = ""
    processed_data_path = ""
    position_size_pct = 0.99

    def init(self):
        """
        Laduje model i scaler, prekomputuje predykcje ML.

        Proces:
        1. Wczytaj artefakt modelu (model + feature_columns)
        2. Wczytaj artefakt scalera (scaler + kolumny)
        3. Wczytaj przetworzone dane (features juz przeskalowane)
        4. Dopasuj indeksy do danych backtestu
        5. Generuj predykcje dla kazdego bara
        6. Zarejestruj jako indicator via self.I()
        """
        import joblib

        # Zaladuj model
        model_artifact = joblib.load(self.model_path)
        self._model = model_artifact["model"]
        self._feature_columns = model_artifact["feature_columns"]

        logger.info(
            "Zaladowano model ML: %d features, best_iter=%s",
            len(self._feature_columns),
            model_artifact.get("metadata", {}).get("best_iteration", "?"),
        )

        # Wczytaj przetworzone dane z features (juz przeskalowane przez RobustScaler)
        processed_df = pd.read_parquet(self.processed_data_path, engine="pyarrow")

        # Usun timezone jesli obecny (aby dopasowac do danych backtestu)
        if processed_df.index.tz is not None:
            processed_df.index = processed_df.index.tz_localize(None)

        # Dopasuj indeksy: przetworzone dane <-> dane backtestu (OHLCV)
        backtest_index = self.data.df.index
        common_index = processed_df.index.intersection(backtest_index)

        if len(common_index) == 0:
            raise ValueError(
                "Brak wspolnych indeksow miedzy danymi backtestu a przetworzonymi features. "
                "Upewnij sie, ze oba zbiory pochodza z tego samego zakresu czasowego."
            )

        logger.info(
            "Dopasowanie indeksow: %d/%d barow backtestu ma features",
            len(common_index), len(backtest_index),
        )

        # Przygotuj feature matrix dla wspolnych indeksow
        features_aligned = processed_df.loc[common_index, self._feature_columns]

        # Generuj predykcje (wektoryzowane - szybkie!)
        predictions = self._model.predict(features_aligned)

        # Stworz pelna serie predykcji dopasowana do backtest_index
        # Bary bez features dostana sygnal 0 (brak pozycji)
        full_predictions = pd.Series(0, index=backtest_index, dtype=int)
        full_predictions.loc[common_index] = predictions

        # Zarejestruj jako indicator backtesting.py
        self.ml_signal = self.I(
            lambda: full_predictions.values,
            name="ML Signal",
        )

        signal_dist = pd.Series(predictions).value_counts()
        logger.info(
            "Predykcje ML: 0 (brak)=%d, 1 (buy)=%d, ratio=%.1f%%",
            signal_dist.get(0, 0),
            signal_dist.get(1, 0),
            signal_dist.get(1, 0) / len(predictions) * 100,
        )

    def next(self):
        """
        Logika decyzyjna oparta na predykcjach ML.

        Odczytuje prekomputowany sygnal z self.ml_signal.
        Sygnal 1 -> buy/utrzymaj pozycje dlugo.
        Sygnal 0 -> zamknij pozycje.

        Ochrona przed Lookahead Bias:
        - Predykcje oparte wylacznie na danych historycznych (features
          obliczone z przeszlych barow w Fazie 1).
        - self.ml_signal[-1] odpowiada sygnalowi dla biezacego bara.
        """
        signal = self.ml_signal[-1]

        if signal == 1:
            # Sygnal wzrostowy -> otworz/utrzymaj long
            if not self.position:
                self.buy(size=self.position_size_pct)
        else:
            # Sygnal 0 -> zamknij long jesli istnieje
            if self.position:
                self.position.close()


# ============================================
# Utilities
# ============================================

# Alias dla kompatybilnosci wstecznej z Faza 2
MLStrategy = SMAStrategy


def _sma(data, period):
    """
    Oblicza prosta srednia kroczaca (Simple Moving Average).

    Args:
        data: Seria cenowa (np. Close).
        period: Okres usredniania.

    Returns:
        Seria z wartosciami SMA.
    """
    return pd.Series(data).rolling(window=period).mean()
