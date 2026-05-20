"""
Modul ML -- trening i ewaluacja modeli.

Zawiera:
- trainer: trening XGBoost z akceleracja GPU (CUDA)
- evaluator: metryki klasyfikacyjne i wykresy feature importance
"""

from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer

__all__ = [
    "ModelTrainer",
    "ModelEvaluator",
]
