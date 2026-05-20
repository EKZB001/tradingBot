"""
Trener modelu XGBoost z akceleracja GPU (CUDA).

Obsluguje:
- Chronologiczny podzial train/test (bez shuffle - zapobieganie Lookahead Bias)
- Trening XGBoost na GPU (RTX 4070 Super)
- Early stopping na zbiorze walidacyjnym
- Zapis/odczyt artefaktow modelu (model + feature_columns + metadata)
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trener modelu klasyfikacyjnego XGBoost z obsluga GPU.

    Uzycie:
        trainer = ModelTrainer(config)
        X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(df)
        model = trainer.train(X_train, y_train, X_test, y_test)
        trainer.save_model(model, feature_cols, path)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
        """
        Przygotowuje dane do treningu z chronologicznym podzialem.

        WAZNE: Uzywa prostego podzialu chronologicznego (80/20),
        NIE losowego train_test_split -- zapobiega Lookahead Bias.

        Args:
            df: DataFrame z features i targetami (z Fazy 1).

        Returns:
            Tuple: (X_train, X_test, y_train, y_test, feature_columns)
        """
        # Wyodrebnienie feature columns (wykluczenie OHLCV i targetow)
        feature_columns = [
            col for col in df.columns
            if col not in self.config.exclude_columns
        ]

        logger.info("Feature columns: %d (z %d total)", len(feature_columns), len(df.columns))
        logger.info("Wykluczone kolumny: %s", list(self.config.exclude_columns))

        # Przygotowanie X i y
        X = df[feature_columns]
        y = df["target_cls"].astype(int)

        # Chronologiczny split (bez shuffle!)
        split_idx = int(len(df) * self.config.train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info("=" * 50)
        logger.info("PODZIAL DANYCH (chronologiczny)")
        logger.info("=" * 50)
        logger.info("  Train: %d wierszy (%.0f%%)", len(X_train), self.config.train_ratio * 100)
        logger.info("  Test:  %d wierszy (%.0f%%)", len(X_test), (1 - self.config.train_ratio) * 100)
        logger.info(
            "  Train range: %s -> %s",
            X_train.index[0].strftime("%Y-%m-%d %H:%M"),
            X_train.index[-1].strftime("%Y-%m-%d %H:%M"),
        )
        logger.info(
            "  Test range:  %s -> %s",
            X_test.index[0].strftime("%Y-%m-%d %H:%M"),
            X_test.index[-1].strftime("%Y-%m-%d %H:%M"),
        )

        # Rozklad klas
        train_dist = y_train.value_counts(normalize=True)
        logger.info("  Rozklad klas (train): 0=%.1f%%, 1=%.1f%%",
                     train_dist.get(0, 0) * 100, train_dist.get(1, 0) * 100)

        return X_train, X_test, y_train, y_test, feature_columns

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> XGBClassifier:
        """
        Trenuje model XGBoost na GPU z early stopping.

        Args:
            X_train: Cechy treningowe.
            y_train: Target treningowy.
            X_test: Cechy testowe (do early stopping).
            y_test: Target testowy.

        Returns:
            Wytrenowany XGBClassifier.
        """
        # Oblicz scale_pos_weight (balansowanie klas)
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        logger.info("=" * 50)
        logger.info("TRENING MODELU XGBoost")
        logger.info("=" * 50)
        logger.info("  Device: %s", self.config.device)
        logger.info("  Tree method: %s", self.config.tree_method)
        logger.info("  n_estimators: %d", self.config.n_estimators)
        logger.info("  max_depth: %d", self.config.max_depth)
        logger.info("  learning_rate: %.3f", self.config.learning_rate)
        logger.info("  early_stopping_rounds: %d", self.config.early_stopping_rounds)
        logger.info("  scale_pos_weight: %.4f", scale_pos_weight)

        model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            device=self.config.device,
            tree_method=self.config.tree_method,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_state=42,
            verbosity=1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        best_iteration = model.best_iteration
        best_score = model.best_score
        logger.info("Trening zakonczony: best_iteration=%d, best_logloss=%.6f",
                     best_iteration, best_score)

        return model

    @staticmethod
    def save_model(
        model: XGBClassifier,
        feature_columns: list[str],
        path: Path,
    ) -> None:
        """
        Zapisuje artefakt modelu (model + feature_columns + metadata).

        Args:
            model: Wytrenowany model XGBoost.
            feature_columns: Lista nazw kolumn uzywaych jako features.
            path: Sciezka do pliku .joblib.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": model,
            "feature_columns": feature_columns,
            "metadata": {
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "n_features": len(feature_columns),
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
            },
        }

        joblib.dump(artifact, path)
        logger.info("Zapisano model -> %s", path)

    @staticmethod
    def load_model(path: Path) -> tuple[XGBClassifier, list[str]]:
        """
        Wczytuje artefakt modelu z pliku joblib.

        Args:
            path: Sciezka do pliku .joblib.

        Returns:
            Tuple: (model XGBClassifier, lista feature_columns)

        Raises:
            FileNotFoundError: Gdy plik nie istnieje.
        """
        if not path.exists():
            raise FileNotFoundError(f"Plik modelu nie istnieje: {path}")

        artifact = joblib.load(path)
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        metadata = artifact.get("metadata", {})

        logger.info("Wczytano model <- %s", path)
        logger.info(
            "  Metadata: features=%d, best_iter=%s, trained=%s",
            metadata.get("n_features", "?"),
            metadata.get("best_iteration", "?"),
            metadata.get("trained_at", "?"),
        )

        return model, feature_columns
