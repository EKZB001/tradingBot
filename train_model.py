"""
Skrypt treningowy modelu ML -- Faza 3.

Uruchamia sekwencje:
    1. Wczytanie przetworzonych danych z Fazy 1 (.parquet)
    2. Przygotowanie danych (chronologiczny split 80/20)
    3. Trening XGBoost na GPU (CUDA)
    4. Ewaluacja na zbiorze testowym
    5. Zapis modelu do models/
    6. Generacja wykresow (feature importance, confusion matrix)

Uzycie:
    python train_model.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Dodaj root projektu do sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ModelConfig, RESULTS_DIR
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator

# ============================================
# Konfiguracja logowania
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "train_model.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_model")


def main() -> None:
    """Punkt wejscia -- trenuje model XGBoost na GPU."""
    start_time = time.perf_counter()

    logger.info("[START] Trening modelu -- Faza 3: Model Training & Inference")

    # 1. Konfiguracja
    config = ModelConfig()
    logger.info(
        "Konfiguracja: device=%s, n_estimators=%d, max_depth=%d, lr=%.3f",
        config.device,
        config.n_estimators,
        config.max_depth,
        config.learning_rate,
    )

    # 2. Wczytanie danych
    logger.info("Wczytywanie danych z: %s", config.data_path)
    df = pd.read_parquet(config.data_path, engine="pyarrow")
    logger.info("Zaladowano dane: %d wierszy, %d kolumn", len(df), len(df.columns))

    # 3. Przygotowanie danych (chronologiczny split)
    trainer = ModelTrainer(config)
    X_train, X_test, y_train, y_test, feature_columns = trainer.prepare_data(df)

    # 4. Trening modelu na GPU
    model = trainer.train(X_train, y_train, X_test, y_test)

    # 5. Ewaluacja na zbiorze testowym
    y_pred = model.predict(X_test)
    evaluator = ModelEvaluator()
    metrics = evaluator.print_classification_report(y_test, y_pred)

    # 6. Zapis modelu
    trainer.save_model(model, feature_columns, config.model_path)

    # 7. Generacja wykresow
    evaluator.plot_feature_importance(
        model, feature_columns,
        save_path=RESULTS_DIR / "feature_importance.png",
    )
    evaluator.plot_confusion_matrix(
        y_test, y_pred,
        save_path=RESULTS_DIR / "confusion_matrix.png",
    )

    elapsed = time.perf_counter() - start_time
    logger.info("[OK] Trening zakonczony pomyslnie w %.2f sekund.", elapsed)
    logger.info(
        "Podsumowanie: Accuracy=%.1f%%, F1=%.1f%%, model -> %s",
        metrics["accuracy"] * 100,
        metrics["f1"] * 100,
        config.model_path,
    )


if __name__ == "__main__":
    main()
