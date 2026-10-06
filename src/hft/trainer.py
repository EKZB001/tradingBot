"""
Trener modelu XGBoost HFT z treningiem przyrostowym (out-of-core) na GPU.

Architektura out-of-core:
1. Laduj dane batchami (2M wierszy) z data_loader
2. Wylicz cechy HFT + target na batchu
3. Trenuj XGBoost przyrostowo (xgb_model=previous_model)
4. Walidacja na gold_val.csv po kazdym batchu
5. Early stopping globalne

Ograniczenie VRAM:
RTX 4070 Super = 12 GB. Batch 2M wierszy * ~140 cech * 4 bajty ~= 1.1 GB.
XGBoost tree buffers dodaja ~2-4 GB. Razem ~5-6 GB — miesci sie z zapasem.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from config.settings import HFTConfig
from src.hft.data_loader import HFTDataLoader
from src.hft.features import add_hft_features
from src.hft.target import compute_triple_barrier_target

logger = logging.getLogger(__name__)


class HFTTrainer:
    """
    Trener modelu HFT z treningiem przyrostowym na GPU.

    Uzycie:
        trainer = HFTTrainer()
        trainer.train_incremental()  # pelny cykl out-of-core
    """

    def __init__(self, config: HFTConfig | None = None) -> None:
        self.config = config or HFTConfig()
        self._loader = HFTDataLoader(self.config)
        self._model: XGBClassifier | None = None
        self._scaler: RobustScaler | None = None
        self._feature_columns: list[str] | None = None

    # ------------------------------------------------------------------
    # Trening przyrostowy (out-of-core)
    # ------------------------------------------------------------------

    def train_incremental(self) -> XGBClassifier:
        """
        Pelny cykl treningu przyrostowego:
        1. Iteruj po batchach train
        2. Dla kazdego batcha: FE + target + scaler + fit
        3. Walidacja na val po kazdym batchu

        Returns:
            Wytrenowany XGBClassifier.
        """
        logger.info("=" * 60)
        logger.info("  TRENING HFT XGBoost (out-of-core, GPU)")
        logger.info("=" * 60)

        # Przygotuj dane walidacyjne (wczytaj calosc — val jest maly)
        val_df = self._prepare_split("val")
        if val_df is None:
            raise RuntimeError("Nie udalo sie przygotowac zbioru walidacyjnego.")

        X_val, y_val = self._extract_xy(val_df)
        logger.info("Zbior walidacyjny: %d wierszy, %d cech.", len(X_val), X_val.shape[1])

        total_rounds = 0
        batch_idx = 0
        best_val_score = float("inf")
        no_improve_batches = 0

        for batch_df in self._loader.iter_batches("train"):
            batch_idx += 1
            logger.info("-" * 40)
            logger.info("BATCH #%d: %d wierszy surowych", batch_idx, len(batch_df))

            # Feature Engineering + Target
            batch_df = add_hft_features(batch_df, self.config)
            batch_df = compute_triple_barrier_target(batch_df, self.config)

            # Usun NaN (poczatek i koniec — brak pelnego okna)
            batch_df = batch_df.dropna(subset=["target"])

            if len(batch_df) == 0:
                logger.warning("Batch #%d pusty po dropna — pomijam.", batch_idx)
                continue

            # Ustal kolumny features (pierwszy batch)
            if self._feature_columns is None:
                self._feature_columns = self._detect_feature_columns(batch_df)
                logger.info("Feature columns: %d kolumn.", len(self._feature_columns))

            # Fit scalera (pierwszy batch) lub transform
            X_batch, y_batch = self._extract_xy(batch_df, fit_scaler=(batch_idx == 1))

            # Oblicz scale_pos_weight
            neg = (y_batch == 0).sum()
            pos = (y_batch == 1).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            logger.info(
                "Batch #%d: X=%s, y: 0=%d, 1=%d, scale_pos_weight=%.4f",
                batch_idx, X_batch.shape, neg, pos, scale_pos_weight,
            )

            # Trenuj (przyrostowo jesli model juz istnieje)
            self._model = self._fit_batch(
                X_batch, y_batch,
                X_val, y_val,
                scale_pos_weight=scale_pos_weight,
            )

            total_rounds += self._model.best_iteration + 1

            # Walidacja
            val_score = self._model.best_score
            logger.info(
                "Batch #%d zakonczony: best_iter=%d, val_logloss=%.6f, total_rounds=%d",
                batch_idx, self._model.best_iteration, val_score, total_rounds,
            )

            # Early stopping globalne
            if val_score < best_val_score:
                best_val_score = val_score
                no_improve_batches = 0
            else:
                no_improve_batches += 1

            if no_improve_batches >= 3:
                logger.info(
                    "Global early stopping: brak poprawy od %d batchow. Koniec treningu.",
                    no_improve_batches,
                )
                break

            if total_rounds >= self.config.max_total_rounds:
                logger.info(
                    "Osiagnieto max_total_rounds=%d. Koniec treningu.",
                    self.config.max_total_rounds,
                )
                break

        logger.info("=" * 60)
        logger.info("  TRENING ZAKONCZONY: %d batchow, %d rund, best_logloss=%.6f",
                     batch_idx, total_rounds, best_val_score)
        logger.info("=" * 60)

        return self._model

    # ------------------------------------------------------------------
    # Zapis / odczyt artefaktow
    # ------------------------------------------------------------------

    def save_artifacts(self) -> None:
        """Zapisuje model, scaler i feature_columns do plikow .joblib."""
        if self._model is None:
            raise RuntimeError("Model nie zostal wytrenowany.")

        # Zapis modelu
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_artifact = {
            "model": self._model,
            "feature_columns": self._feature_columns,
            "metadata": {
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "n_features": len(self._feature_columns),
                "best_iteration": self._model.best_iteration,
                "best_score": self._model.best_score,
                "architecture": "HFT_tick_scalper",
            },
        }
        joblib.dump(model_artifact, self.config.model_path)
        logger.info("Zapisano model -> %s", self.config.model_path)

        # Zapis scalera
        self.config.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_artifact = {
            "scaler": self._scaler,
            "columns": self._feature_columns,
        }
        joblib.dump(scaler_artifact, self.config.scaler_path)
        logger.info("Zapisano scaler -> %s", self.config.scaler_path)

    @staticmethod
    def load_model(model_path: Path) -> tuple[XGBClassifier, list[str]]:
        """
        Wczytuje wytrenowany model i feature_columns z pliku .joblib.

        Returns:
            Tuple (model, feature_columns).
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Plik modelu nie istnieje: {model_path}")

        artifact = joblib.load(model_path)
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        metadata = artifact.get("metadata", {})

        logger.info("Wczytano model HFT <- %s", model_path)
        logger.info(
            "  features=%d, best_iter=%s, trained=%s",
            metadata.get("n_features", "?"),
            metadata.get("best_iteration", "?"),
            metadata.get("trained_at", "?"),
        )

        return model, feature_columns

    @staticmethod
    def load_scaler(scaler_path: Path) -> tuple[RobustScaler, list[str]]:
        """Wczytuje scaler i liste kolumn."""
        if not scaler_path.exists():
            raise FileNotFoundError(f"Plik scalera nie istnieje: {scaler_path}")

        artifact = joblib.load(scaler_path)
        return artifact["scaler"], artifact["columns"]

    # ------------------------------------------------------------------
    # Metody prywatne
    # ------------------------------------------------------------------

    def _prepare_split(self, split: str) -> pd.DataFrame | None:
        """Wczytuje caly split, dodaje cechy HFT i target."""
        try:
            df = self._loader.load_full(split)
            df = add_hft_features(df, self.config)
            df = compute_triple_barrier_target(df, self.config)
            df = df.dropna(subset=["target"])
            return df
        except Exception as e:
            logger.error("Blad przygotowania splitu '%s': %s", split, e)
            return None

    def _detect_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Zwraca scisle zdefiniowana liste cech HFT, ktore sa uzywane w live tradingu.
        Gwarantuje zgodnosc wymiarow miedzy treningiem a live_scalper.py.
        """
        from src.hft.live_buffer import TickBuffer
        names = TickBuffer(self.config).get_feature_names()
        
        missing = [c for c in names if c not in df.columns]
        if missing:
            logger.warning("Brakuje kolumn w DataFrame: %s", missing)
            
        return [c for c in names if c in df.columns]

    def _extract_xy(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Wyodrebnia macierz X (features) i wektor y (target) z DataFrame.

        Args:
            df: DataFrame z features i target.
            fit_scaler: Czy fitowac nowy scaler (pierwszy batch).

        Returns:
            Tuple (X jako numpy float32, y jako numpy int).
        """
        if self._feature_columns is None:
            self._feature_columns = self._detect_feature_columns(df)

        # Filtruj kolumny ktore istnieja w DataFrame
        available_cols = [c for c in self._feature_columns if c in df.columns]

        X = df[available_cols].values.astype(np.float32)
        y = df["target"].values.astype(int)

        # Skalowanie
        if fit_scaler:
            self._scaler = RobustScaler()
            X = self._scaler.fit_transform(X).astype(np.float32)
            logger.info("Scaler fitowany na batchu (%d kolumn).", X.shape[1])
        elif self._scaler is not None:
            X = self._scaler.transform(X).astype(np.float32)

        # Zastap NaN i inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def _fit_batch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_pos_weight: float = 1.0,
    ) -> XGBClassifier:
        """
        Trenuje jeden batch XGBoost (przyrostowo jesli model istnieje).

        Uzywa parametru xgb_model do kontynuacji treningu
        z poprzedniego stanu modelu (incremental learning).
        """
        model = XGBClassifier(
            n_estimators=self.config.n_estimators_per_batch,
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

        # Trening przyrostowy: xgb_model = poprzedni model
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": 50,
        }

        if self._model is not None:
            fit_kwargs["xgb_model"] = self._model

        model.fit(X_train, y_train, **fit_kwargs)

        return model
