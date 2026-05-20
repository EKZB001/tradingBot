"""
Autonomiczny demon handlowy -- Live Trading Bot.

Handluje na koncie demo Fusion Markets w czasie rzeczywistym,
wykorzystujac wytrenowany model XGBoost z Fazy 3.

Petla decyzyjna:
    1. Oczekiwanie na zamkniecie swieci M5 (wyrownanie do zegara)
    2. Pobranie bufora 200 swiec z MT5
    3. Replikacja pipeline'u FE z Fazy 1
    4. Predykcja modelu ML
    5. Egzekucja zlecenia (BUY / CLOSE / HOLD)

Bezpieczenstwo:
    - Zadne klucze/hasla nie sa w kodzie -- tylko z .env
    - Walidacja zmiennych srodowiskowych przy starcie
    - Graceful shutdown (Ctrl+C)

Uzycie:
    python live_bot.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

# Dodaj root projektu do sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ModelConfig, MODELS_DIR, SCALERS_DIR
from src.execution import close_position, get_open_position, open_buy
from src.ingestion.mt5_client import MT5Client, TIMEFRAME_MAP
from src.live.feature_builder import LiveFeatureBuilder
from src.models.trainer import ModelTrainer

# ============================================
# Konfiguracja
# ============================================

SYMBOL = "XAUUSD"
TIMEFRAME_NAME = "TIMEFRAME_M5"
TIMEFRAME_MINUTES = 5
VOLUME = 0.01  # Micro lot
BUFFER_BARS = 200  # Bufor swiec do obliczenia features (min ~50 dla rolling(21))
MAGIC_NUMBER = 20250520

# ============================================
# Konfiguracja logowania
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "live_bot.log", encoding="utf-8"),
    ],
)
# Wycisz debug logi z bibliotek FE (generuja duzo logow przy kazdym barze)
logging.getLogger("src.features").setLevel(logging.WARNING)
logger = logging.getLogger("live_bot")

# Flaga graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Obsluga sygnalu przerwania (Ctrl+C)."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("[STOP] Otrzymano sygnal zakonczenia. Zamykanie po biezacym cyklu...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================
# Pobieranie danych z MT5
# ============================================

def fetch_latest_bars(symbol: str, timeframe_name: str, count: int) -> pd.DataFrame:
    """
    Pobiera ostatnie N swiec z MT5.

    Args:
        symbol: Instrument finansowy.
        timeframe_name: Nazwa timeframe'u.
        count: Liczba swiec do pobrania.

    Returns:
        DataFrame z kolumnami OHLCV i indeksem datetime UTC.

    Raises:
        RuntimeError: Gdy MT5 nie zwroci danych.
    """
    timeframe = TIMEFRAME_MAP[timeframe_name]
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 nie zwrocil danych dla {symbol} [{timeframe_name}]. "
            f"Blad: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]

    return df


# ============================================
# Wyrownanie do zegara M5
# ============================================

def wait_for_next_bar(interval_minutes: int) -> None:
    """
    Czeka na zamkniecie biezacej swieci M5.

    Wyrownuje czas do pelnego interwalu (np. 14:05:00, 14:10:00)
    plus 5 sekund marginesu (aby MT5 zdazyl zamknac swiece).

    Args:
        interval_minutes: Interwał w minutach (5 dla M5).
    """
    now = datetime.now(tz=timezone.utc)
    current_minute = now.minute
    seconds_into_bar = (current_minute % interval_minutes) * 60 + now.second

    # Czas do nastepnego zamkniecia baru
    seconds_remaining = (interval_minutes * 60) - seconds_into_bar

    # Dodaj 5 sekund marginesu
    wait_seconds = seconds_remaining + 5

    if wait_seconds < 0:
        wait_seconds = 0

    next_bar = now.timestamp() + wait_seconds
    next_bar_dt = datetime.fromtimestamp(next_bar, tz=timezone.utc)

    logger.info(
        "Oczekiwanie %.0f sek. na zamkniecie baru (nastepny: %s UTC)...",
        wait_seconds,
        next_bar_dt.strftime("%H:%M:%S"),
    )

    # Czekaj w krotkich intervalach (sprawdzamy shutdown co sekunde)
    end_time = time.time() + wait_seconds
    while time.time() < end_time:
        if _shutdown_requested:
            return
        time.sleep(1)


# ============================================
# Petla decyzyjna
# ============================================

def decision_cycle(
    feature_builder: LiveFeatureBuilder,
    model,
    feature_columns: list[str],
) -> None:
    """
    Jeden cykl decyzyjny bota.

    1. Pobierz bufor swiec z MT5
    2. Zbuduj cechy (replikacja pipeline'u Fazy 1)
    3. Predykcja modelu ML
    4. Egzekucja zlecenia

    Args:
        feature_builder: Builder cech z zaladowanym scalerem.
        model: Wytrenowany model XGBoost.
        feature_columns: Lista kolumn wymaganych przez model.
    """
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info("[CYKL] Rozpoczecie cyklu decyzyjnego -- %s UTC", now_utc)

    # 1. Pobranie bufora swiec
    try:
        df_raw = fetch_latest_bars(SYMBOL, TIMEFRAME_NAME, BUFFER_BARS)
    except RuntimeError as e:
        logger.error("Nie udalo sie pobrac danych: %s", e)
        return

    last_bar_time = df_raw.index[-1].strftime("%Y-%m-%d %H:%M")
    last_close = df_raw["close"].iloc[-1]
    logger.info(
        "Pobrano %d swiec. Ostatnia: %s, close=%.2f",
        len(df_raw), last_bar_time, last_close,
    )

    # 2. Feature Engineering (replikacja Fazy 1)
    features = feature_builder.build_features(df_raw)
    if features is None:
        logger.warning("Nie udalo sie zbudowac cech -- pomijam cykl.")
        return

    # 3. Predykcja ML
    prediction = model.predict(features[feature_columns])[0]
    probability = model.predict_proba(features[feature_columns])[0]
    logger.info(
        "Predykcja ML: %d (prawdopodobienstwo: 0=%.3f, 1=%.3f)",
        prediction, probability[0], probability[1],
    )

    # 4. Sprawdz otwarte pozycje
    position = get_open_position(SYMBOL)
    if position:
        pos_type = "LONG" if position["type"] == 0 else "SHORT"
        logger.info(
            "Otwarta pozycja: %s, ticket=#%d, vol=%.2f, profit=%.2f",
            pos_type, position["ticket"], position["volume"], position["profit"],
        )
    else:
        logger.info("Brak otwartych pozycji na %s.", SYMBOL)

    # 5. Logika decyzyjna
    if prediction == 1:
        # Sygnal BUY
        if position is None:
            # Brak pozycji -- otworz LONG
            logger.info("[DECYZJA] Sygnal BUY -> otwieram pozycje dluga.")
            open_buy(SYMBOL, VOLUME, MAGIC_NUMBER)

        elif position["type"] == mt5.ORDER_TYPE_SELL:
            # Otwarta krotka -- zamknij i otworz dluga
            logger.info("[DECYZJA] Sygnal BUY -> zamykam SHORT i otwieram LONG.")
            close_position(position, MAGIC_NUMBER)
            open_buy(SYMBOL, VOLUME, MAGIC_NUMBER)

        else:
            # Juz mamy LONG -- utrzymaj
            logger.info("[DECYZJA] Sygnal BUY -> utrzymuje istniejaca pozycje dluga.")

    else:
        # Sygnal 0 (FLAT)
        if position is not None:
            # Mamy otwarta pozycje -- zamknij
            logger.info("[DECYZJA] Sygnal FLAT -> zamykam pozycje.")
            close_position(position, MAGIC_NUMBER)
        else:
            logger.info("[DECYZJA] Sygnal FLAT -> brak pozycji, czekam.")

    logger.info("[CYKL] Zakonczony.")


# ============================================
# Main
# ============================================

def main() -> None:
    """Punkt wejscia -- uruchamia demona handlowego."""
    logger.info("=" * 60)
    logger.info("  LIVE TRADING BOT -- XAUUSD M5")
    logger.info("  Model: XGBoost | Broker: Fusion Markets (Demo)")
    logger.info("=" * 60)

    # 1. Walidacja i ladowanie artefaktow
    config = ModelConfig()

    if not config.model_path.exists():
        logger.error("Model ML nie znaleziony: %s", config.model_path)
        logger.error("Uruchom najpierw: python train_model.py")
        sys.exit(1)

    if not config.scaler_path.exists():
        logger.error("Scaler nie znaleziony: %s", config.scaler_path)
        logger.error("Uruchom najpierw: python pipeline.py")
        sys.exit(1)

    # Zaladuj model
    model, feature_columns = ModelTrainer.load_model(config.model_path)
    logger.info("Model ML zaladowany: %d features.", len(feature_columns))

    # Zaladuj feature builder (ze scalerem)
    feature_builder = LiveFeatureBuilder(
        scaler_path=config.scaler_path,
        feature_columns=feature_columns,
    )

    # 2. Polaczenie z MT5
    logger.info("Laczenie z MetaTrader 5...")
    mt5_client = MT5Client()
    try:
        mt5_client.connect()
    except Exception as e:
        logger.error("Nie udalo sie polaczyc z MT5: %s", e)
        sys.exit(1)

    # Sprawdz dostepnosc symbolu
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        logger.error("Symbol %s nie jest dostepny w terminalu MT5.", SYMBOL)
        mt5_client.disconnect()
        sys.exit(1)

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)

    logger.info(
        "Symbol %s dostepny. Spread: %d pkt, Min lot: %.2f",
        SYMBOL, symbol_info.spread, symbol_info.volume_min,
    )

    # 3. Petla glowna
    logger.info("=" * 60)
    logger.info("  BOT URUCHOMIONY -- Ctrl+C aby zatrzymac")
    logger.info("=" * 60)

    try:
        # Pierwszy cykl natychmiast (aby nie czekac 5 min)
        decision_cycle(feature_builder, model, feature_columns)

        while not _shutdown_requested:
            wait_for_next_bar(TIMEFRAME_MINUTES)
            if _shutdown_requested:
                break
            decision_cycle(feature_builder, model, feature_columns)

    except KeyboardInterrupt:
        logger.info("[STOP] Przerwanie przez uzytkownika.")
    except Exception as e:
        logger.exception("[KRYTYCZNY BLAD] Nieoczekiwany wyjatek: %s", e)
    finally:
        # Graceful shutdown
        logger.info("Zamykanie polaczenia z MT5...")
        mt5_client.disconnect()
        logger.info("[OK] Bot zatrzymany prawidlowo.")


if __name__ == "__main__":
    main()
