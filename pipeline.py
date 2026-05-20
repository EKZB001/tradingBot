"""
Główny orkiestrator pipeline'u danych — Faza 1.

Uruchamia sekwencję:
    1. Połączenie z MT5 → pobranie surowych OHLCV → zapis raw .parquet
    2. Feature Engineering (lagged, technical, volatility, calendar, price action)
    3. Definicja targetów (klasyfikacja + regresja)
    4. Preprocessing (czyszczenie NaN, skalowanie, zapis scalera)
    5. Zapis przetworzonego zbioru → processed .parquet

Użycie:
    python pipeline.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Dodaj root projektu do sys.path (uruchomienie z dowolnego katalogu)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SCALERS_DIR,
    PipelineConfig,
)
from src.features.calendar import add_calendar_features
from src.features.lagged import add_lagged_features
from src.features.price_action import add_price_action_features
from src.features.technical import add_technical_indicators
from src.features.volatility import add_volatility_features
from src.ingestion.mt5_client import MT5Client
from src.preprocessing.preprocessor import Preprocessor
from src.targets.target_builder import TargetBuilder

# ============================================
# Konfiguracja logowania
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


def run_ingestion(config: PipelineConfig) -> pd.DataFrame:
    """
    Krok 1: Pobiera surowe dane OHLCV z MetaTrader 5.

    Returns:
        DataFrame z danymi OHLCV (indeks: datetime UTC).
    """
    logger.info("=" * 60)
    logger.info("KROK 1: Ekstrakcja danych z MetaTrader 5")
    logger.info("=" * 60)

    ingestion = config.ingestion
    raw_path = RAW_DATA_DIR / f"{ingestion.symbol}_{ingestion.timeframe_name}.parquet"

    with MT5Client() as client:
        df = client.fetch_ohlcv(
            symbol=ingestion.symbol,
            timeframe_name=ingestion.timeframe_name,
            date_from=ingestion.date_from,
            date_to=ingestion.date_to,
            chunk_days=ingestion.chunk_days,
            chunk_sleep_seconds=ingestion.chunk_sleep_seconds,
        )
        client.save_raw(df, raw_path)

    return df


def run_feature_engineering(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Krok 2: Generuje pełen zestaw cech (features) z surowych OHLCV.

    Returns:
        DataFrame wzbogacony o cechy lagged, technical, volatility,
        calendar i price action.
    """
    logger.info("=" * 60)
    logger.info("KROK 2: Feature Engineering")
    logger.info("=" * 60)

    cols_before = len(df.columns)

    df = add_lagged_features(df, config.lagged)
    df = add_technical_indicators(df, config.technical)
    df = add_volatility_features(df, config.volatility)
    df = add_calendar_features(df)
    df = add_price_action_features(df)

    cols_after = len(df.columns)
    logger.info(
        "Feature Engineering zakończony: %d → %d kolumn (+%d cech).",
        cols_before,
        cols_after,
        cols_after - cols_before,
    )

    return df


def run_target_definition(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Krok 3: Definiuje zmienne objaśniane (targety).

    Returns:
        DataFrame z kolumnami 'target_cls' i 'target_reg'.
    """
    logger.info("=" * 60)
    logger.info("KROK 3: Definicja targetów")
    logger.info("=" * 60)

    builder = TargetBuilder(
        config=config.target,
        symbol=config.ingestion.symbol,
    )
    df = builder.add_all_targets(df)

    return df


def run_preprocessing(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Krok 4: Czyści NaN, skaluje cechy i zapisuje scaler.

    Returns:
        Oczyszczony i przeskalowany DataFrame gotowy do treningu ML.
    """
    logger.info("=" * 60)
    logger.info("KROK 4: Preprocessing")
    logger.info("=" * 60)

    preprocessor = Preprocessor()

    # Czyszczenie NaN (rolling indicators + forward-looking targets)
    df = preprocessor.clean_nans(df, strategy="drop")

    # Skalowanie cech (RobustScaler — odporny na outliers finansowe)
    df, scaler, scaled_columns = preprocessor.fit_transform(
        df, scaler_type="robust"
    )

    # Zapis scalera do joblib (potrzebny w Fazie 3 — inference na żywo)
    scaler_path = SCALERS_DIR / f"{config.ingestion.symbol}_scaler.joblib"
    preprocessor.save_scaler(scaler, scaler_path, columns=scaled_columns)

    return df


def save_processed(df: pd.DataFrame, config: PipelineConfig) -> Path:
    """
    Krok 5: Zapisuje przetworzone dane do Parquet.

    Returns:
        Ścieżka do zapisanego pliku.
    """
    logger.info("=" * 60)
    logger.info("KROK 5: Zapis przetworzonego zbioru")
    logger.info("=" * 60)

    ingestion = config.ingestion
    output_path = (
        PROCESSED_DATA_DIR
        / f"{ingestion.symbol}_{ingestion.timeframe_name}_features.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, engine="pyarrow", compression="snappy")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Zapisano przetworzone dane -> %s (%.2f MB)", output_path, size_mb)

    return output_path


def print_summary(df: pd.DataFrame) -> None:
    """Wypisuje podsumowanie końcowego zbioru danych."""
    logger.info("=" * 60)
    logger.info("PODSUMOWANIE")
    logger.info("=" * 60)

    logger.info("Kształt danych: %s", df.shape)
    logger.info("Zakres czasowy: %s -> %s", df.index[0], df.index[-1])
    logger.info("Liczba cech (bez targetów): %d", len(df.columns) - 2)
    logger.info("Typy danych:\n%s", df.dtypes.value_counts().to_string())
    logger.info("Brak NaN: %s", df.isna().sum().sum() == 0)

    # Rozkład targetu klasyfikacyjnego
    if "target_cls" in df.columns:
        distribution = df["target_cls"].value_counts(normalize=True)
        logger.info(
            "Rozkład target_cls:\n%s",
            distribution.to_string(),
        )

    # Statystyki targetu regresyjnego
    if "target_reg" in df.columns:
        logger.info(
            "Statystyki target_reg:\n%s",
            df["target_reg"].describe().to_string(),
        )

    logger.info("Pierwsze 5 wierszy:\n%s", df.head().to_string())


def main() -> None:
    """Punkt wejścia — uruchamia pełen pipeline Fazy 1."""
    start_time = time.perf_counter()

    logger.info("[START] Start pipeline'u — Faza 1: Data Pipeline & Feature Engineering")

    config = PipelineConfig()

    logger.info(
        "Konfiguracja: symbol=%s, timeframe=%s, lata=%.1f",
        config.ingestion.symbol,
        config.ingestion.timeframe_name,
        config.ingestion.years_back,
    )

    # Sekwencja pipeline'u
    df = run_ingestion(config)
    df = run_feature_engineering(df, config)
    df = run_target_definition(df, config)
    df = run_preprocessing(df, config)
    save_processed(df, config)
    print_summary(df)

    elapsed = time.perf_counter() - start_time
    logger.info("[OK] Pipeline zakończony pomyślnie w %.2f sekund.", elapsed)


if __name__ == "__main__":
    main()
