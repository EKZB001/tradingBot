"""
Orkiestrator treningu modelu HFT Tick Scalper.

Pelny pipeline:
1. Pobranie danych z HuggingFace (jesli brak lokalnie)
2. Trening out-of-core XGBoost na GPU (gold_train.csv)
3. Walidacja na gold_val.csv
4. Backtest na gold_test.csv
5. Zapis modelu + scalera + raportu

Uzycie:
    python train_hft.py                 # pelny pipeline
    python train_hft.py --download-only # tylko pobranie danych
    python train_hft.py --backtest-only # tylko backtest (wymaga modelu)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Dodaj root projektu do sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import HFTConfig, RESULTS_DIR
from src.hft.data_loader import HFTDataLoader
from src.hft.features import add_hft_features
from src.hft.target import compute_triple_barrier_target
from src.hft.trainer import HFTTrainer
from src.hft.backtester import TickBacktester

# ============================================
# Konfiguracja logowania
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "train_hft.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_hft")


def download_data(config: HFTConfig) -> None:
    """Pobiera dane z HuggingFace Hub."""
    loader = HFTDataLoader(config)
    paths = loader.download()
    for split, path in paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("[%s] %s (%.1f MB)", split.upper(), path, size_mb)


def train_model(config: HFTConfig) -> None:
    """Uruchamia pelny cykl treningu out-of-core."""
    trainer = HFTTrainer(config)

    start_time = time.time()
    trainer.train_incremental()
    elapsed = time.time() - start_time

    logger.info("Czas treningu: %.1f sekund (%.1f minut).", elapsed, elapsed / 60)

    # Zapis artefaktow
    trainer.save_artifacts()
    logger.info("Artefakty zapisane pomyslnie.")


def run_backtest(config: HFTConfig) -> None:
    """Uruchamia backtest na zbiorze testowym."""
    logger.info("=" * 60)
    logger.info("  BACKTEST NA ZBIORZE TESTOWYM")
    logger.info("=" * 60)

    # Wczytaj model
    model, feature_columns = HFTTrainer.load_model(config.model_path)
    scaler, scaler_columns = HFTTrainer.load_scaler(config.scaler_path)

    # Wczytaj dane testowe
    loader = HFTDataLoader(config)
    test_df = loader.load_full("test")

    # Feature Engineering + Target
    test_df = add_hft_features(test_df, config)
    test_df = compute_triple_barrier_target(test_df, config)
    test_df = test_df.dropna(subset=["target"])

    logger.info("Dane testowe: %d wierszy po dropna.", len(test_df))

    # Przygotuj features
    available_cols = [c for c in feature_columns if c in test_df.columns]
    X_test = test_df[available_cols].values.astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Predykcje
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    n_signals = (predictions == 1).sum()
    logger.info(
        "Predykcje: %d sygnalow BUY z %d tikow (%.1f%%).",
        n_signals, len(predictions), n_signals / len(predictions) * 100,
    )

    # Backtest
    mid_prices = test_df["mid_price"].values.astype(np.float64)
    backtester = TickBacktester(config)
    result = backtester.run(mid_prices, predictions, volume=config.live_volume)

    # Zapisz raport
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "backtest_hft_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(str(result))
        f.write(f"\n\nModel: {config.model_path.name}")
        f.write(f"\nFeatures: {len(feature_columns)}")
        f.write(f"\nTest samples: {len(test_df)}")
        f.write(f"\nBUY signals: {n_signals}")

    logger.info("Raport zapisany -> %s", report_path)


def main() -> None:
    """Punkt wejscia — parsuje argumenty i uruchamia pipeline."""
    parser = argparse.ArgumentParser(
        description="HFT Tick Scalper — trening modelu XGBoost na GPU",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Tylko pobierz dane z HuggingFace (bez treningu).",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Tylko backtest na zbiorze testowym (wymaga wytrenowanego modelu).",
    )
    args = parser.parse_args()

    config = HFTConfig()

    logger.info("=" * 60)
    logger.info("  HFT TICK SCALPER — TRAINING PIPELINE")
    logger.info("  Symbol: %s | GPU: %s | Batch: %d wierszy",
                config.symbol, config.device, config.batch_rows)
    logger.info("=" * 60)

    if args.download_only:
        download_data(config)
        return

    if args.backtest_only:
        run_backtest(config)
        return

    # Pelny pipeline
    # 1. Pobranie danych
    logger.info("\n--- KROK 1: Pobieranie danych ---")
    download_data(config)

    # 2. Trening
    logger.info("\n--- KROK 2: Trening modelu ---")
    train_model(config)

    # 3. Backtest
    logger.info("\n--- KROK 3: Backtest ---")
    run_backtest(config)

    logger.info("\n[OK] Pipeline HFT zakonczony pomyslnie.")


if __name__ == "__main__":
    main()
