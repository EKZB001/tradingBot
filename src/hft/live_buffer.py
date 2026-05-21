"""
Bufor tikowy dla live scalpera — przechowuje N ostatnich tikow z MT5.

Zoptymalizowany pod minimalne opoznienia:
- collections.deque z maxlen (O(1) push/pop)
- Obliczenia na czystych tablicach NumPy (bez DataFrame w petli)
- Pre-alokacja tablic dla cech

Kluczowa zasada: ZERO tworzenia DataFrame w petli live.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from config.settings import HFTConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TickData:
    """Pojedynczy tik z MT5 (minimalna struktura)."""

    bid: float
    ask: float
    time_seconds: float  # Unix timestamp (time.time())
    volume: int = 0


class TickBuffer:
    """
    Bufor collections.deque przechowujacy ostatnie N tikow z MT5.

    Oblicza cechy HFT na biezaco z czystych tablic NumPy,
    bez tworzenia obiektow Pandas DataFrame w petli live.

    Uzycie:
        buffer = TickBuffer(maxlen=500)
        while True:
            tick = mt5.symbol_info_tick("XAUUSD")
            buffer.push(TickData(bid=tick.bid, ask=tick.ask, ...))
            if buffer.is_ready():
                features = buffer.compute_features()
                prediction = model.predict(features.reshape(1, -1))
    """

    def __init__(self, config: HFTConfig | None = None) -> None:
        self.config = config or HFTConfig()
        self._buffer: deque[TickData] = deque(maxlen=self.config.tick_buffer_size)
        self._min_ticks = max(self.config.velocity_windows) + 1

        # Pre-alokacja tablic numpy (recykl buforow)
        maxlen = self.config.tick_buffer_size
        self._mid_prices = np.zeros(maxlen, dtype=np.float64)
        self._spreads = np.zeros(maxlen, dtype=np.float64)
        self._volumes = np.zeros(maxlen, dtype=np.float64)
        self._times = np.zeros(maxlen, dtype=np.float64)

    @property
    def size(self) -> int:
        """Aktualna liczba tikow w buforze."""
        return len(self._buffer)

    def is_ready(self) -> bool:
        """Czy bufor ma wystarczajaco tikow do obliczenia cech."""
        return len(self._buffer) >= self._min_ticks

    def push(self, tick: TickData) -> None:
        """
        Dodaje tik do bufora (O(1) — deque z maxlen).

        Automatycznie odrzuca najstarsze tiki po osiagnieciu maxlen.
        """
        self._buffer.append(tick)

    def compute_features(self) -> np.ndarray:
        """
        Oblicza wektor cech HFT z biezacego stanu bufora.

        Returns:
            1D numpy array (float32) z cechami gotowymi do model.predict().

        Raises:
            RuntimeError: Gdy bufor nie jest gotowy (za malo tikow).
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Bufor niegotowy: {len(self._buffer)}/{self._min_ticks} tikow."
            )

        n = len(self._buffer)

        # Kopiuj dane z deque do tablic numpy (O(n))
        for i, tick in enumerate(self._buffer):
            self._mid_prices[i] = (tick.bid + tick.ask) / 2.0
            self._spreads[i] = tick.ask - tick.bid
            self._volumes[i] = tick.volume
            self._times[i] = tick.time_seconds

        mid = self._mid_prices[:n]
        spreads = self._spreads[:n]
        volumes = self._volumes[:n]
        times = self._times[:n]

        features: list[float] = []

        # 1. Tick Velocity (predkosc zmiany ceny)
        for w in self.config.velocity_windows:
            if n >= w + 1:
                velocity = (mid[-1] - mid[-1 - w]) / w
            else:
                velocity = 0.0
            features.append(velocity)

        # 2. Micro-Volatility (std zwrotow)
        pct_changes = np.diff(mid) / mid[:-1] if n > 1 else np.array([0.0])
        for w in self.config.micro_vol_windows:
            if len(pct_changes) >= w:
                micro_vol = float(np.std(pct_changes[-w:]))
            else:
                micro_vol = 0.0
            features.append(micro_vol)

        # 3. Spread Z-Score
        if n >= 100:
            spread_mean = np.mean(spreads[-100:])
            spread_std = np.std(spreads[-100:])
            spread_zscore = (spreads[-1] - spread_mean) / spread_std if spread_std > 0 else 0.0
        else:
            spread_zscore = 0.0
        features.append(spread_zscore)

        # 4. Volume Surge
        if n >= 100:
            vol_mean = np.mean(volumes[-100:])
            vol_surge = volumes[-1] / vol_mean if vol_mean > 0 else 1.0
        else:
            vol_surge = 1.0
        features.append(vol_surge)

        # 5. Time regime features (sin/cos godziny i minuty)
        last_time = times[-1]
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(last_time, tz=timezone.utc)
        hour_frac = dt.hour + dt.minute / 60.0
        minute_frac = dt.minute + dt.second / 60.0

        features.append(np.sin(2 * np.pi * hour_frac / 24.0))
        features.append(np.cos(2 * np.pi * hour_frac / 24.0))
        features.append(np.sin(2 * np.pi * minute_frac / 60.0))
        features.append(np.cos(2 * np.pi * minute_frac / 60.0))

        # 6. Macro-Volatility
        macro_window = min(self.config.macro_vol_window, n)
        if macro_window >= 2:
            macro_vol = float(np.std(mid[-macro_window:]))
        else:
            macro_vol = 0.0
        features.append(macro_vol)

        # 7. Surowe dane biezacego tiku
        features.append(mid[-1])      # ostatni mid_price
        features.append(spreads[-1])  # ostatni spread
        features.append(volumes[-1])  # ostatni volume

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> list[str]:
        """Zwraca nazwy cech w kolejnosci identycznej jak compute_features()."""
        names: list[str] = []

        # Velocity
        for w in self.config.velocity_windows:
            names.append(f"tick_velocity_{w}")

        # Micro-vol
        for w in self.config.micro_vol_windows:
            names.append(f"micro_vol_{w}")

        # Spread, volume
        names.append("spread_zscore")
        names.append("vol_surge")

        # Time regime
        names.extend(["hour_sin", "hour_cos", "minute_sin", "minute_cos"])

        # Macro-vol
        names.append("macro_vol")

        # Raw tick data
        names.extend(["mid_price", "spread", "volume"])

        return names
