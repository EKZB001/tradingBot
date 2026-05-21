"""
Modul HFT Tick Scalper — handel na danych tikowych XAUUSD.

Podmoduly:
    data_loader  — pobieranie i ladowanie danych z HuggingFace
    features     — cechy HFT (velocity, micro-vol, regime)
    target       — definicja targetu (Triple-Barrier)
    trainer      — trening przyrostowy XGBoost na GPU
    backtester   — wektorowy tick backtester
    live_buffer  — bufor deque dla live tick streaming
"""

from src.hft.data_loader import HFTDataLoader
from src.hft.features import add_hft_features
from src.hft.target import compute_triple_barrier_target
from src.hft.trainer import HFTTrainer
from src.hft.backtester import TickBacktester
from src.hft.live_buffer import TickBuffer

__all__ = [
    "HFTDataLoader",
    "add_hft_features",
    "compute_triple_barrier_target",
    "HFTTrainer",
    "TickBacktester",
    "TickBuffer",
]
