"""
Autonomiczny demon HFT Tick Scalper — handel na zywo.

Nasłuchuje na tiki XAUUSD z MT5 w petli while True,
reagujac na kazda zmiane ceny (bez harmonogramu schedule).

Architektura:
    1. Bufor deque przechowuje N ostatnich tikow
    2. Na kazdym nowym tiku: oblicz cechy -> predykcja modelu
    3. Sygnal BUY -> otworz pozycje z SL/TP
    4. Zarzadzanie pozycjami: broker zamyka po SL/TP

Logowanie (pkt 6 z new_model.md):
    - Loguj TYLKO krytyczne zdarzenia (open/close/error)
    - Ignoruj puste przebiegi petli (brak sygnalu)

Bezpieczenstwo:
    - Credentials z .env (python-dotenv)
    - Graceful shutdown (Ctrl+C -> mt5.shutdown())

Uzycie:
    python live_scalper.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Dodaj root projektu do sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import HFTConfig
from src.execution import get_open_position, open_buy_with_sl_tp
from src.hft.live_buffer import TickBuffer, TickData
from src.hft.trainer import HFTTrainer
from src.ingestion.mt5_client import MT5Client

# ============================================
# Konfiguracja logowania
# ============================================

# Minimalne logowanie — tylko krytyczne zdarzenia
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "live_scalper.log", encoding="utf-8"),
    ],
)
# Wycisz debug logi z bibliotek wewnetrznych
logging.getLogger("src").setLevel(logging.WARNING)
logging.getLogger("config").setLevel(logging.WARNING)

logger = logging.getLogger("live_scalper")

# ============================================
# Graceful shutdown
# ============================================

_shutdown_requested = False


def _signal_handler(signum, frame):
    """Obsluga sygnalu przerwania (Ctrl+C)."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("[STOP] Otrzymano sygnal zakonczenia. Zamykanie...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================
# Glowna petla
# ============================================

def main() -> None:
    """Punkt wejscia — uruchamia demona HFT scalper."""
    import MetaTrader5 as mt5

    config = HFTConfig()

    logger.info("=" * 60)
    logger.info("  HFT TICK SCALPER — XAUUSD LIVE")
    logger.info("  SL=%.0f pips | TP=%.0f pips | Vol=%.2f lot",
                config.sl_pips, config.tp_pips, config.live_volume)
    logger.info("=" * 60)

    # 1. Zaladuj model i scaler
    if not config.model_path.exists():
        logger.error("Model HFT nie znaleziony: %s", config.model_path)
        logger.error("Uruchom najpierw: python train_hft.py")
        sys.exit(1)

    model, feature_columns = HFTTrainer.load_model(config.model_path)
    scaler, scaler_columns = HFTTrainer.load_scaler(config.scaler_path)
    logger.info("Model HFT zaladowany: %d features.", len(feature_columns))

    # 2. Polaczenie z MT5
    logger.info("Laczenie z MetaTrader 5...")
    mt5_client = MT5Client()
    try:
        mt5_client.connect()
    except Exception as e:
        logger.error("Nie udalo sie polaczyc z MT5: %s", e)
        sys.exit(1)

    # Sprawdz dostepnosc symbolu
    symbol_info = mt5.symbol_info(config.symbol)
    if symbol_info is None:
        logger.error("Symbol %s nie jest dostepny.", config.symbol)
        mt5_client.disconnect()
        sys.exit(1)

    if not symbol_info.visible:
        mt5.symbol_select(config.symbol, True)

    logger.info(
        "Symbol %s dostepny. Spread: %d pkt, Min lot: %.2f",
        config.symbol, symbol_info.spread, symbol_info.volume_min,
    )

    # 3. Inicjalizacja bufora tikowego
    tick_buffer = TickBuffer(config)
    last_tick_time = 0.0
    trades_count = 0

    logger.info("=" * 60)
    logger.info("  SCALPER URUCHOMIONY — Ctrl+C aby zatrzymac")
    logger.info("  Bufor: %d tikow, sleep: %.2fs",
                config.tick_buffer_size, config.tick_sleep_seconds)
    logger.info("=" * 60)

    # 4. Petla glowna — nasłuch na tiki
    try:
        while not _shutdown_requested:
            tick = mt5.symbol_info_tick(config.symbol)

            if tick is None:
                time.sleep(config.tick_sleep_seconds)
                continue

            # Pomin duplikaty (ten sam timestamp)
            if tick.time_msc == last_tick_time:
                time.sleep(config.tick_sleep_seconds)
                continue
            last_tick_time = tick.time_msc

            # Dodaj tik do bufora
            tick_data = TickData(
                bid=tick.bid,
                ask=tick.ask,
                time_seconds=tick.time,
                volume=int(getattr(tick, "volume", 0)),
            )
            tick_buffer.push(tick_data)

            # Czy bufor jest gotowy do predykcji?
            if not tick_buffer.is_ready():
                continue

            # Oblicz cechy (pure numpy — bez DataFrame)
            features = tick_buffer.compute_features()

            # Skalowanie (jesli scaler jest dostepny i rozmiar sie zgadza)
            if scaler is not None and features.shape[0] == len(scaler_columns):
                features = scaler.transform(
                    features.reshape(1, -1)
                ).astype(np.float32).flatten()
            else:
                features = features.reshape(1, -1)

            # Predykcja modelu
            features_2d = features.reshape(1, -1) if features.ndim == 1 else features
            prediction = model.predict(features_2d)[0]

            if prediction != 1:
                # Brak sygnalu — NIE logujemy (pkt 6: asynchroniczne logowanie)
                time.sleep(config.tick_sleep_seconds)
                continue

            # Sygnal BUY! Sprawdz czy nie mamy otwartej pozycji
            position = get_open_position(config.symbol)
            if position is not None:
                # Juz mamy pozycje — pomin (SL/TP zamknie automatycznie)
                time.sleep(config.tick_sleep_seconds)
                continue

            # Otworz BUY z SL/TP
            logger.info(
                "[SYGNAL] BUY @ bid=%.5f, ask=%.5f, spread=%.5f",
                tick.bid, tick.ask, tick.ask - tick.bid,
            )

            success = open_buy_with_sl_tp(
                symbol=config.symbol,
                volume=config.live_volume,
                sl_pips=config.sl_pips,
                tp_pips=config.tp_pips,
                pip_size=config.pip_size,
                magic=config.live_magic,
            )

            if success:
                trades_count += 1
                logger.info("[TRADE #%d] Pozycja otwarta pomyslnie.", trades_count)
            else:
                logger.error("[BLAD] Nie udalo sie otworzyc pozycji.")

            time.sleep(config.tick_sleep_seconds)

    except KeyboardInterrupt:
        logger.info("[STOP] Przerwanie przez uzytkownika.")
    except Exception as e:
        logger.exception("[KRYTYCZNY BLAD] Nieoczekiwany wyjatek: %s", e)
    finally:
        logger.info("Zamykanie polaczenia z MT5...")
        mt5_client.disconnect()
        logger.info(
            "[OK] Scalper zatrzymany. Laczna liczba transakcji: %d",
            trades_count,
        )


if __name__ == "__main__":
    main()
