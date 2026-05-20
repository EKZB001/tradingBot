"""
Klient MetaTrader 5 — bezpieczna ekstrakcja danych OHLCV.

Połączenie z MT5 oparte wyłącznie na zmiennych środowiskowych (.env).
Obsługuje context manager (with) dla automatycznego connect/disconnect.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Self

import MetaTrader5 as mt5
import pandas as pd
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# Wymagane zmienne środowiskowe do połączenia z MT5
_REQUIRED_ENV_VARS = ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH")

# Mapowanie nazw timeframe'ów na stałe MT5
TIMEFRAME_MAP: dict[str, int] = {
    "TIMEFRAME_M1": mt5.TIMEFRAME_M1,
    "TIMEFRAME_M5": mt5.TIMEFRAME_M5,
    "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
    "TIMEFRAME_M30": mt5.TIMEFRAME_M30,
    "TIMEFRAME_H1": mt5.TIMEFRAME_H1,
    "TIMEFRAME_H4": mt5.TIMEFRAME_H4,
    "TIMEFRAME_D1": mt5.TIMEFRAME_D1,
}


class MT5ConnectionError(Exception):
    """Błąd połączenia z terminalem MetaTrader 5."""


class MT5Client:
    """
    Klient do bezpiecznego pobierania danych historycznych z MetaTrader 5.

    Przykład użycia:
        with MT5Client() as client:
            df = client.fetch_ohlcv("EURUSD", "TIMEFRAME_H1", date_from, date_to)
            client.save_raw(df, Path("data/raw/eurusd_h1.parquet"))
    """

    def __init__(self, env_path: Path | str = ".env") -> None:
        """
        Inicjalizuje klienta — ładuje credentials z pliku .env.

        Args:
            env_path: Ścieżka do pliku .env (domyślnie: katalog roboczy).

        Raises:
            ValueError: Gdy brakuje wymaganych zmiennych środowiskowych.
        """
        self._credentials = self._load_credentials(Path(env_path))
        self._connected = False

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Automatyczne połączenie przy wejściu do bloku `with`."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Automatyczne rozłączenie przy wyjściu z bloku `with`."""
        self.disconnect()

    # ------------------------------------------------------------------
    # Połączenie
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Inicjalizuje terminal MT5 i loguje się na konto.

        Raises:
            MT5ConnectionError: Gdy inicjalizacja lub logowanie nie powiedzie się.
        """
        logger.info("Inicjalizacja terminala MetaTrader 5...")

        if not mt5.initialize(self._credentials["MT5_PATH"]):
            error = mt5.last_error()
            raise MT5ConnectionError(
                f"Nie udało się zainicjalizować MT5: {error}"
            )

        authorized = mt5.login(
            login=int(self._credentials["MT5_LOGIN"]),
            password=self._credentials["MT5_PASSWORD"],
            server=self._credentials["MT5_SERVER"],
        )

        if not authorized:
            error = mt5.last_error()
            mt5.shutdown()
            raise MT5ConnectionError(
                f"Nie udało się zalogować do MT5: {error}"
            )

        self._connected = True
        account_info = mt5.account_info()
        logger.info(
            "Połączono z MT5 — konto: %s, serwer: %s, balans: %.2f %s",
            account_info.login,
            account_info.server,
            account_info.balance,
            account_info.currency,
        )

    def disconnect(self) -> None:
        """Bezpiecznie zamyka sesję MT5."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Rozłączono z MetaTrader 5.")

    # ------------------------------------------------------------------
    # Pobieranie danych
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe_name: str,
        date_from: datetime,
        date_to: datetime,
        chunk_days: int = 180,
        chunk_sleep_seconds: float = 2.0,
    ) -> pd.DataFrame:
        """
        Pobiera historyczne dane OHLCV z MT5 w zadanym zakresie dat.
        Wykorzystuje batching (dzielenie na kawałki) by ominąć limity MT5.

        Args:
            symbol: Instrument finansowy (np. 'EURUSD').
            timeframe_name: Nazwa timeframe'u (np. 'TIMEFRAME_H1').
            date_from: Data początkowa (UTC).
            date_to: Data końcowa (UTC).
            chunk_days: Rozmiar okna (batcha) w dniach.
            chunk_sleep_seconds: Czas oczekiwania (sen) między requestami w sekundach.

        Returns:
            DataFrame z kolumnami: time, open, high, low, close, volume.

        Raises:
            MT5ConnectionError: Gdy klient nie jest połączony.
            ValueError: Gdy timeframe jest nieznany lub brak danych.
        """
        import time
        from datetime import timedelta

        self._ensure_connected()

        timeframe = self._resolve_timeframe(timeframe_name)

        logger.info(
            "Pobieranie danych %s [%s] od %s do %s w batchach co %d dni...",
            symbol,
            timeframe_name,
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
            chunk_days,
        )

        all_dfs = []
        current_start = date_from

        while current_start < date_to:
            current_end = min(current_start + timedelta(days=chunk_days), date_to)
            logger.info("Pobieranie batcha: %s -> %s", current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"))
            
            rates = mt5.copy_rates_range(symbol, timeframe, current_start, current_end)
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.warning(
                    "Brak danych lub błąd dla batcha %s -> %s. Błąd: %s. "
                    "Prawdopodobnie brak historii u brokera tak daleko w przeszłość.",
                    current_start.strftime("%Y-%m-%d"),
                    current_end.strftime("%Y-%m-%d"),
                    error
                )
            else:
                all_dfs.append(pd.DataFrame(rates))
                
            current_start = current_end
            
            if current_start < date_to:
                time.sleep(chunk_sleep_seconds)

        if not all_dfs:
            raise ValueError(
                f"Brak danych dla {symbol} [{timeframe_name}]. "
                f"Sprawdź czy symbol jest dostępny u brokera."
            )

        df = pd.concat(all_dfs, ignore_index=True)

        # Konwersja UNIX timestamp -> datetime UTC
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        
        # Usunięcie ewentualnych duplikatów na stykach batchy (poza usunięciem wierszy)
        df = df[~df.index.duplicated(keep="first")]

        # Ustandaryzowanie nazw kolumn
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Usunięcie kolumn niepotrzebnych (spread, real_volume jeśli istnieją)
        columns_to_keep = ["open", "high", "low", "close", "volume"]
        df = df[[col for col in columns_to_keep if col in df.columns]]

        logger.info(
            "Pobrano łącznie %d barek -> zakres: %s -> %s",
            len(df),
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
        )

        return df

    # ------------------------------------------------------------------
    # Zapis danych
    # ------------------------------------------------------------------

    @staticmethod
    def save_raw(df: pd.DataFrame, path: Path) -> None:
        """
        Zapisuje DataFrame do skompresowanego pliku Parquet.

        Args:
            df: Dane do zapisu.
            path: Ścieżka docelowa pliku .parquet.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Zapisano surowe dane → %s (%.2f MB)", path, size_mb)

    # ------------------------------------------------------------------
    # Metody prywatne
    # ------------------------------------------------------------------

    @staticmethod
    def _load_credentials(env_path: Path) -> dict[str, str]:
        """
        Ładuje i waliduje zmienne środowiskowe z pliku .env.

        Raises:
            ValueError: Gdy brakuje wymaganych zmiennych.
        """
        values = dotenv_values(env_path)

        missing = [var for var in _REQUIRED_ENV_VARS if not values.get(var)]
        if missing:
            raise ValueError(
                f"Brakujące zmienne środowiskowe w {env_path}: {missing}. "
                f"Skopiuj .env.example → .env i uzupełnij danymi."
            )

        return {var: values[var] for var in _REQUIRED_ENV_VARS}

    def _ensure_connected(self) -> None:
        """Sprawdza czy klient jest połączony z MT5."""
        if not self._connected:
            raise MT5ConnectionError(
                "Klient nie jest połączony z MT5. "
                "Użyj connect() lub context managera (with MT5Client() as client)."
            )

    @staticmethod
    def _resolve_timeframe(name: str) -> int:
        """
        Mapuje nazwę timeframe'u na stałą MT5.

        Raises:
            ValueError: Gdy nazwa jest nieznana.
        """
        if name not in TIMEFRAME_MAP:
            raise ValueError(
                f"Nieznany timeframe: '{name}'. "
                f"Dostępne: {list(TIMEFRAME_MAP.keys())}"
            )
        return TIMEFRAME_MAP[name]
