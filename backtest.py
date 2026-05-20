"""
Glowny orkiestrator backtestu -- Faza 2 & 3.

Uruchamia backtest w jednym z dwoch trybow:
    - SMA: prosty SMA crossover (benchmark z Fazy 2)
    - ML:  predykcje z wytrenowanego modelu XGBoost (Faza 3)

Uzycie:
    python backtest.py          # Domyslnie: tryb ML
    python backtest.py --sma    # Benchmark SMA crossover
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Dodaj root projektu do sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting import Backtest

from config.settings import BacktestConfig, ModelConfig, RESULTS_DIR
from src.backtesting.costs import TradingCosts, calculate_commission_pct
from src.backtesting.data_feed import load_backtest_data
from src.backtesting.report import print_backtest_report
from src.backtesting.strategy import MLPredictionStrategy, SMAStrategy

# ============================================
# Konfiguracja logowania
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("backtest")


def run_sma_backtest(bt_config: BacktestConfig) -> None:
    """Uruchamia backtest z benchmark SMA crossover."""
    logger.info("[MODE] Benchmark SMA Crossover")

    df = load_backtest_data(bt_config.data_path)

    costs = TradingCosts(
        commission_per_side=bt_config.commission_per_side,
        spread_points=bt_config.spread_points,
        slippage_points=bt_config.slippage_points,
        lot_size=bt_config.position_size,
    )
    costs.log_costs()

    typical_price = df["Close"].median()
    commission_pct = calculate_commission_pct(
        commission_per_side=bt_config.commission_per_side,
        lot_size=bt_config.position_size,
        typical_price=typical_price,
    )

    bt = Backtest(
        data=df,
        strategy=SMAStrategy,
        cash=bt_config.initial_cash,
        commission=commission_pct,
        exclusive_orders=True,
        trade_on_close=False,
    )

    stats = bt.run(
        sma_fast=bt_config.sma_fast,
        sma_slow=bt_config.sma_slow,
        spread_points=bt_config.spread_points,
        slippage_points=bt_config.slippage_points,
    )

    report_path = RESULTS_DIR / "backtest_sma_report.txt"
    print_backtest_report(stats, initial_cash=bt_config.initial_cash, output_path=report_path)


def run_ml_backtest(bt_config: BacktestConfig, ml_config: ModelConfig) -> None:
    """Uruchamia backtest z predykcjami modelu ML."""
    logger.info("[MODE] ML Prediction (XGBoost)")

    # Sprawdz czy model istnieje
    if not ml_config.model_path.exists():
        logger.error(
            "Model ML nie znaleziony: %s. Uruchom najpierw 'python train_model.py'.",
            ml_config.model_path,
        )
        sys.exit(1)

    # Wczytaj surowe dane OHLCV (do backtestu — ceny realne)
    df = load_backtest_data(bt_config.data_path)

    costs = TradingCosts(
        commission_per_side=bt_config.commission_per_side,
        spread_points=bt_config.spread_points,
        slippage_points=bt_config.slippage_points,
        lot_size=bt_config.position_size,
    )
    costs.log_costs()

    typical_price = df["Close"].median()
    commission_pct = calculate_commission_pct(
        commission_per_side=bt_config.commission_per_side,
        lot_size=bt_config.position_size,
        typical_price=typical_price,
    )

    bt = Backtest(
        data=df,
        strategy=MLPredictionStrategy,
        cash=bt_config.initial_cash,
        commission=commission_pct,
        exclusive_orders=True,
        trade_on_close=False,
    )

    stats = bt.run(
        model_path=str(ml_config.model_path),
        scaler_path=str(ml_config.scaler_path),
        processed_data_path=str(ml_config.data_path),
    )

    report_path = RESULTS_DIR / "backtest_ml_report.txt"
    print_backtest_report(stats, initial_cash=bt_config.initial_cash, output_path=report_path)


def main() -> None:
    """Punkt wejscia -- uruchamia backtest w wybranym trybie."""
    parser = argparse.ArgumentParser(description="Backtest tradingBot")
    parser.add_argument(
        "--sma", action="store_true",
        help="Uzyj benchmark SMA crossover zamiast modelu ML",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    bt_config = BacktestConfig()
    ml_config = ModelConfig()

    if args.sma:
        logger.info("[START] Backtest -- tryb SMA (benchmark)")
        run_sma_backtest(bt_config)
    else:
        logger.info("[START] Backtest -- tryb ML (XGBoost)")
        run_ml_backtest(bt_config, ml_config)

    elapsed = time.perf_counter() - start_time
    logger.info("[OK] Backtest zakonczony pomyslnie w %.2f sekund.", elapsed)


if __name__ == "__main__":
    main()
