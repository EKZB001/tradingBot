"""
Cechy kalendarzowe (Calendar / Time-based Features).

Kodowanie cykliczne (sin/cos) godzin i dni tygodnia
oraz flagi sesji giełdowych (Azja, Londyn, Nowy Jork, Overlap).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Granice sesji giełdowych w godzinach UTC
_SESSIONS: dict[str, tuple[int, int]] = {
    "session_asia": (0, 8),       # 00:00 – 08:00 UTC
    "session_london": (7, 16),    # 07:00 – 16:00 UTC
    "session_ny": (12, 21),       # 12:00 – 21:00 UTC
    "session_overlap": (12, 16),  # 12:00 – 16:00 UTC (Londyn ∩ NY)
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje cechy kalendarzowe z kodowaniem cyklicznym do DataFrame'a.

    Generowane cechy:
        - hour_sin, hour_cos     — cyklicznie zakodowana godzina (0–23)
        - dow_sin, dow_cos       — cyklicznie zakodowany dzień tygodnia (0–4, pon–pt)
        - session_asia           — flaga sesji azjatyckiej (0/1)
        - session_london         — flaga sesji londyńskiej (0/1)
        - session_ny             — flaga sesji nowojorskiej (0/1)
        - session_overlap        — flaga overlap'u Londyn/NY (0/1)

    Args:
        df: DataFrame z DatetimeIndex (UTC).

    Returns:
        DataFrame z dodanymi kolumnami kalendarzowymi.

    Note:
        Kodowanie sin/cos zapobiega "skokowi" na granicach cyklu —
        np. godziny 23:00 i 00:00 traktowane są jako bliskie, nie odległe.
    """
    index = df.index
    hours = index.hour
    day_of_week = index.dayofweek  # 0 = poniedziałek, 4 = piątek

    # --- Kodowanie cykliczne godziny (cykl 24h) ---
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # --- Kodowanie cykliczne dnia tygodnia (cykl 5 dni — forex pon–pt) ---
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 5)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 5)

    # --- Flagi sesji giełdowych ---
    for session_name, (start_hour, end_hour) in _SESSIONS.items():
        df[session_name] = ((hours >= start_hour) & (hours < end_hour)).astype(
            np.int8
        )

    logger.info(
        "Dodano cechy kalendarzowe: sin/cos (godzina, dzień), 4 flagi sesji."
    )

    return df
