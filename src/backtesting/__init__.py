"""
Modul backtestingowy -- Faza 2 & 3.

Zawiera komponenty do symulacji strategii na danych historycznych:
- data_feed: wczytywanie danych z Parquet
- strategy: SMAStrategy (benchmark) + MLPredictionStrategy (XGBoost)
- costs: konfiguracja kosztow transakcyjnych Fusion Markets
- report: generowanie raportu wynikowego
"""

from src.backtesting.data_feed import load_backtest_data
from src.backtesting.report import print_backtest_report
from src.backtesting.strategy import MLPredictionStrategy, SMAStrategy

__all__ = [
    "load_backtest_data",
    "MLPredictionStrategy",
    "SMAStrategy",
    "print_backtest_report",
]
