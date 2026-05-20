"""
Ewaluator modelu ML -- metryki klasyfikacyjne i wizualizacje.

Generuje:
- Classification report (Accuracy, Precision, Recall, F1)
- Confusion Matrix (wykres + log)
- Feature Importance (top N cech jako wykres slupkowy)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend bez GUI (zapis do pliku)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Ewaluator modelu klasyfikacyjnego.

    Uzycie:
        evaluator = ModelEvaluator()
        evaluator.print_classification_report(y_true, y_pred)
        evaluator.plot_feature_importance(model, features, path)
        evaluator.plot_confusion_matrix(y_true, y_pred, path)
    """

    @staticmethod
    def print_classification_report(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Wyswietla pelny raport klasyfikacyjny.

        Args:
            y_true: Prawdziwe etykiety.
            y_pred: Predykcje modelu.

        Returns:
            Slownik z glownymi metrykami.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        logger.info("=" * 50)
        logger.info("METRYKI KLASYFIKACYJNE (zbiór testowy)")
        logger.info("=" * 50)
        logger.info("  Accuracy:   %.4f (%.1f%%)", accuracy, accuracy * 100)
        logger.info("  Precision:  %.4f (%.1f%%)", precision, precision * 100)
        logger.info("  Recall:     %.4f (%.1f%%)", recall, recall * 100)
        logger.info("  F1 Score:   %.4f (%.1f%%)", f1, f1 * 100)

        # Confusion matrix jako tekst
        cm = confusion_matrix(y_true, y_pred)
        logger.info("  Confusion Matrix:")
        logger.info("    TN=%d  FP=%d", cm[0, 0], cm[0, 1])
        logger.info("    FN=%d  TP=%d", cm[1, 0], cm[1, 1])

        # Pelny raport sklearn
        report_str = classification_report(y_true, y_pred, target_names=["0 (brak ruchu)", "1 (wzrost)"])
        logger.info("Pelny raport:\n%s", report_str)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def plot_feature_importance(
        model: XGBClassifier,
        feature_names: list[str],
        save_path: Path,
        top_n: int = 20,
    ) -> None:
        """
        Generuje wykres top N najwazniejszych cech modelu.

        Args:
            model: Wytrenowany XGBClassifier.
            feature_names: Lista nazw cech.
            save_path: Sciezka do zapisu wykresu PNG.
            top_n: Ile cech wyswietlic (domyslnie 20).
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        # Top N
        top_features = feature_importance_df.head(top_n)

        # Log
        logger.info("=" * 50)
        logger.info("TOP %d FEATURE IMPORTANCE", top_n)
        logger.info("=" * 50)
        for _, row in top_features.iterrows():
            logger.info("  %-25s  %.4f", row["feature"], row["importance"])

        # Wykres
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            top_features["feature"].values[::-1],
            top_features["importance"].values[::-1],
            color="#2196F3",
            edgecolor="#1565C0",
        )
        ax.set_xlabel("Importance (Gain)", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importance -- XGBoost XAUUSD M5", fontsize=14)
        ax.tick_params(axis="y", labelsize=10)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Wykres feature importance zapisany -> %s", save_path)

    @staticmethod
    def plot_confusion_matrix(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        save_path: Path,
    ) -> None:
        """
        Generuje wykres confusion matrix.

        Args:
            y_true: Prawdziwe etykiety.
            y_pred: Predykcje modelu.
            save_path: Sciezka do zapisu wykresu PNG.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        labels = ["0 (brak ruchu)", "1 (wzrost)"]

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predykcja",
            ylabel="Prawda",
            title="Confusion Matrix -- XGBoost XAUUSD M5",
        )

        # Dodaj wartosci liczbowe do komorek
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold",
                )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Wykres confusion matrix zapisany -> %s", save_path)
