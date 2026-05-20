"""
Centralna konfiguracja pipeline'u danych.

Wszystkie parametry Feature Engineering, ekstrakcji i targetów
zdefiniowane w jednym miejscu — łatwe do modyfikacji bez grzebania w kodzie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ============================================
# Ścieżki katalogów danych
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCALERS_DIR = DATA_DIR / "scalers"


# ============================================
# Mapa rozmiaru pipsa dla popularnych instrumentów
# ============================================

PIP_SIZE_MAP: dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.01,
    "XAGUSD": 0.001,
}


@dataclass(frozen=True)
class IngestionConfig:
    """Parametry pobierania danych z MT5."""

    symbol: str = "XAUUSD"
    timeframe_name: str = "TIMEFRAME_M5"
    years_back: float = 5.0
    chunk_days: int = 180
    chunk_sleep_seconds: float = 2.0

    @property
    def date_from(self) -> datetime:
        """Oblicz datę początkową na podstawie `years_back`."""
        return datetime.now(tz=timezone.utc) - timedelta(days=self.years_back * 365)

    @property
    def date_to(self) -> datetime:
        """Data końcowa — aktualny moment UTC."""
        return datetime.now(tz=timezone.utc)

    @property
    def pip_size(self) -> float:
        """Rozmiar pipsa dla wybranego symbolu."""
        if self.symbol not in PIP_SIZE_MAP:
            raise ValueError(
                f"Nieznany symbol '{self.symbol}'. "
                f"Dostępne: {list(PIP_SIZE_MAP.keys())}. "
                f"Dodaj symbol do PIP_SIZE_MAP w config/settings.py."
            )
        return PIP_SIZE_MAP[self.symbol]


@dataclass(frozen=True)
class LaggedConfig:
    """Parametry zmiennych opóźnionych."""

    periods: tuple[int, ...] = (1, 3, 5, 10, 21)


@dataclass(frozen=True)
class TechnicalConfig:
    """Parametry wskaźników technicznych."""

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14


@dataclass(frozen=True)
class VolatilityConfig:
    """Parametry miar zmienności."""

    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    volatility_window: int = 21


@dataclass(frozen=True)
class TargetConfig:
    """Parametry zmiennej objaśnianej (target)."""

    horizon: int = 10
    threshold_pips: int = 20


# ============================================
# Faza 2: Konfiguracja backtestingu
# ============================================

RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================
# Faza 3: Konfiguracja modelu ML
# ============================================

MODELS_DIR = PROJECT_ROOT / "models"

# Kolumny wykluczone z feature matrix (surowe ceny + targety)
_MODEL_EXCLUDE_COLUMNS = (
    "open", "high", "low", "close", "volume",
    "target_cls", "target_reg",
)


@dataclass(frozen=True)
class ModelConfig:
    """
    Parametry treningu modelu XGBoost.

    GPU: RTX 4070 Super (CUDA).
    """

    # Podział chronologiczny train/test
    train_ratio: float = 0.80

    # Hyperparametry XGBoost
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.01
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Early stopping
    early_stopping_rounds: int = 50

    # Akceleracja GPU
    device: str = "cuda"
    tree_method: str = "hist"

    # Kolumny wykluczone z features
    exclude_columns: tuple[str, ...] = _MODEL_EXCLUDE_COLUMNS

    @property
    def data_path(self) -> Path:
        """Sciezka do przetworzonych danych z Fazy 1."""
        return PROCESSED_DATA_DIR / "XAUUSD_TIMEFRAME_M5_features.parquet"

    @property
    def scaler_path(self) -> Path:
        """Sciezka do zapisanego scalera z Fazy 1."""
        return SCALERS_DIR / "XAUUSD_scaler.joblib"

    @property
    def model_path(self) -> Path:
        """Sciezka do zapisu wytrenowanego modelu."""
        return MODELS_DIR / "xgboost_XAUUSD_M5.joblib"


@dataclass(frozen=True)
class BacktestConfig:
    """
    Parametry środowiska backtestingowego.

    Koszty skalibrowane pod Fusion Markets — Raw Spread Account (XAUUSD).
    """

    # Kapitał początkowy (USD)
    initial_cash: float = 10_000.0

    # Koszty transakcyjne — Fusion Markets Raw Spread
    # Prowizja: $2.25 per side per lot ($4.50 round-turn)
    commission_per_side: float = 2.25

    # Symulowany spread w punktach cenowych ($0.05 = 5 punktów po 0.01)
    spread_points: float = 0.05

    # Poślizg cenowy w punktach ($0.01 = 1 punkt)
    slippage_points: float = 0.01

    # Rozmiar pozycji (lot) — micro lot na $10k konta
    position_size: float = 0.01

    # Parametry strategii placeholder (SMA crossover)
    sma_fast: int = 10
    sma_slow: int = 50

    # Ścieżka do danych (surowe OHLCV z Fazy 1)
    @property
    def data_path(self) -> Path:
        """Ścieżka do surowych danych OHLCV dla aktualnego symbolu."""
        return RAW_DATA_DIR / "XAUUSD_TIMEFRAME_M5.parquet"


@dataclass(frozen=True)
class PipelineConfig:
    """
    Główna konfiguracja pipeline'u — agreguje wszystkie sub-konfiguracje.

    Użycie:
        config = PipelineConfig()
        config.ingestion.symbol  # 'EURUSD'
        config.technical.rsi_period  # 14
    """

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    lagged: LaggedConfig = field(default_factory=LaggedConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
