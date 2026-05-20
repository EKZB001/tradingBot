"""
Budowa zmiennych objaśnianych (Targetów) — klasyfikacja i regresja.

Targety są forward-looking (patrzą w przyszłość o N świec),
co oznacza NaN na końcu serii — obsługiwane przez Preprocessor.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import PIP_SIZE_MAP, TargetConfig

logger = logging.getLogger(__name__)


class TargetBuilder:
    """
    Generuje kolumny targetów dla modeli ML.

    Obsługuje dwa typy:
        1. Klasyfikacja — binarna etykieta (1/0) na podstawie progu pipsowego
        2. Regresja — dokładna wartość przyszłego zwrotu procentowego
    """

    def __init__(
        self,
        config: TargetConfig | None = None,
        pip_size: float | None = None,
        symbol: str = "EURUSD",
    ) -> None:
        """
        Inicjalizuje TargetBuilder.

        Args:
            config: Parametry targetu (horizon, threshold_pips).
            pip_size: Rozmiar pipsa (nadpisuje auto-detekcję z symbolu).
            symbol: Symbol instrumentu — używany do auto-detekcji pip_size.
        """
        self.config = config or TargetConfig()
        self.pip_size = pip_size or PIP_SIZE_MAP.get(symbol, 0.0001)

    def add_classification_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje binarny target klasyfikacyjny.

        target_cls = 1 jeśli cena za `horizon` świec wzrośnie o ≥ `threshold_pips` pipsów.
        target_cls = 0 w przeciwnym razie.

        Args:
            df: DataFrame z kolumną 'close'.

        Returns:
            DataFrame z kolumną 'target_cls'.
        """
        horizon = self.config.horizon
        threshold = self.config.threshold_pips * self.pip_size

        # Przyszła zmiana ceny (forward-looking)
        future_change = df["close"].shift(-horizon) - df["close"]

        # Binaryzacja: 1 jeśli wzrost ≥ próg, 0 w.p.p.
        df["target_cls"] = (future_change >= threshold).astype("Int8")

        # NaN na końcu serii (brak danych przyszłych) — zachowujemy jako NaN
        df.loc[df["close"].shift(-horizon).isna(), "target_cls"] = pd.NA

        logger.info(
            "Dodano target klasyfikacyjny: horizon=%d, próg=%d pips (%.5f).",
            horizon,
            self.config.threshold_pips,
            threshold,
        )

        return df

    def add_regression_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje ciągły target regresyjny (procentowy zwrot).

        target_reg = (close[t+horizon] - close[t]) / close[t] * 100 [%]

        Args:
            df: DataFrame z kolumną 'close'.

        Returns:
            DataFrame z kolumną 'target_reg'.
        """
        horizon = self.config.horizon

        future_close = df["close"].shift(-horizon)
        df["target_reg"] = (future_close - df["close"]) / df["close"] * 100

        logger.info(
            "Dodano target regresyjny: horizon=%d (przyszły zwrot %%).",
            horizon,
        )

        return df

    def add_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje oba typy targetów jednocześnie.

        Args:
            df: DataFrame z kolumną 'close'.

        Returns:
            DataFrame z kolumnami 'target_cls' i 'target_reg'.
        """
        df = self.add_classification_target(df)
        df = self.add_regression_target(df)
        return df
