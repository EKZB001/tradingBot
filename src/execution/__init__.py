"""
Modul egzekucji zlecen na MetaTrader 5.

Obsluguje:
- Sprawdzanie otwartych pozycji
- Otwieranie pozycji BUY/SELL
- Zamykanie pozycji
- Walidacja wynikow zlecen i obsluga bledow (requote, odrzucenia)
"""

from __future__ import annotations

import logging

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class OrderExecutionError(Exception):
    """Blad egzekucji zlecenia na MT5."""


def get_open_position(symbol: str) -> dict | None:
    """
    Sprawdza czy istnieje otwarta pozycja na danym symbolu.

    Args:
        symbol: Instrument finansowy (np. 'XAUUSD').

    Returns:
        Slownik z danymi pozycji lub None jesli brak.
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return None

    # Zwracamy pierwsza otwarta pozycje na tym symbolu
    pos = positions[0]
    return {
        "ticket": pos.ticket,
        "type": pos.type,  # 0 = BUY, 1 = SELL
        "volume": pos.volume,
        "price_open": pos.price_open,
        "profit": pos.profit,
        "symbol": pos.symbol,
    }


def open_buy(symbol: str, volume: float, magic: int = 20250520) -> bool:
    """
    Otwiera pozycje dluga (BUY) na danym symbolu.

    Args:
        symbol: Instrument finansowy.
        volume: Wolumen w lotach (np. 0.01).
        magic: Numer identyfikacyjny bota (magic number).

    Returns:
        True jesli zlecenie wykonane pomyslnie.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Symbol %s nie jest dostepny w terminalu MT5.", symbol)
        return False

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    price = mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 20,  # Max slippage w punktach
        "magic": magic,
        "comment": "LiveBot BUY",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return _validate_order_result(result, "BUY", symbol, volume, price)


def open_sell(symbol: str, volume: float, magic: int = 20250520) -> bool:
    """
    Otwiera pozycje krotka (SELL) na danym symbolu.

    Args:
        symbol: Instrument finansowy.
        volume: Wolumen w lotach.
        magic: Magic number bota.

    Returns:
        True jesli zlecenie wykonane pomyslnie.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Symbol %s nie jest dostepny w terminalu MT5.", symbol)
        return False

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    price = mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 20,
        "magic": magic,
        "comment": "LiveBot SELL",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return _validate_order_result(result, "SELL", symbol, volume, price)


def close_position(position: dict, magic: int = 20250520) -> bool:
    """
    Zamyka otwarta pozycje na MT5.

    Args:
        position: Slownik pozycji (z get_open_position).
        magic: Magic number bota.

    Returns:
        True jesli zamkniecie wykonane pomyslnie.
    """
    symbol = position["symbol"]
    ticket = position["ticket"]
    volume = position["volume"]

    # Zlecenie przeciwne do otwartego
    if position["type"] == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
        direction_name = "CLOSE LONG"
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
        direction_name = "CLOSE SHORT"

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": magic,
        "comment": f"LiveBot {direction_name}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return _validate_order_result(result, direction_name, symbol, volume, price)


def _validate_order_result(
    result, direction: str, symbol: str, volume: float, price: float
) -> bool:
    """
    Waliduje wynik zlecenia MT5 i loguje szczegoly.

    Returns:
        True jesli zlecenie wykonane pomyslnie (retcode == 10009).
    """
    if result is None:
        logger.error(
            "[BLAD] %s %s %.2f lot @ %.2f -- brak odpowiedzi z MT5. last_error=%s",
            direction, symbol, volume, price, mt5.last_error(),
        )
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            "[OK] %s %s %.2f lot @ %.5f -- ticket #%d",
            direction, symbol, volume, result.price, result.order,
        )
        return True

    # Obsluga bledow
    error_map = {
        mt5.TRADE_RETCODE_REQUOTE: "REQUOTE -- cena sie zmienila",
        mt5.TRADE_RETCODE_REJECT: "ODRZUCONE przez serwer",
        mt5.TRADE_RETCODE_CANCEL: "ANULOWANE",
        mt5.TRADE_RETCODE_INVALID: "NIEPRAWIDLOWE zlecenie",
        mt5.TRADE_RETCODE_INVALID_VOLUME: "NIEPRAWIDLOWY wolumen",
        mt5.TRADE_RETCODE_NO_MONEY: "BRAK SRODKOW na koncie",
        mt5.TRADE_RETCODE_MARKET_CLOSED: "RYNEK ZAMKNIETY",
        mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "ZBYT WIELE ZLECEN",
    }

    error_msg = error_map.get(result.retcode, f"Nieznany blad (retcode={result.retcode})")
    logger.error(
        "[BLAD] %s %s %.2f lot @ %.2f -- %s (comment: %s)",
        direction, symbol, volume, price, error_msg, result.comment,
    )

    return False


def open_buy_with_sl_tp(
    symbol: str,
    volume: float,
    sl_pips: float,
    tp_pips: float,
    pip_size: float,
    magic: int = 20250521,
) -> bool:
    """
    Otwiera pozycje BUY z predefiniowanym Stop Loss i Take Profit.

    Uzywane przez HFT scalper — kazda pozycja musi miec sztywne SL/TP.

    Args:
        symbol: Instrument finansowy (np. 'XAUUSD').
        volume: Wolumen w lotach.
        sl_pips: Stop Loss w pipsach (odleglosc od ceny wejscia).
        tp_pips: Take Profit w pipsach.
        pip_size: Rozmiar pipsa (0.01 dla XAUUSD).
        magic: Magic number bota.

    Returns:
        True jesli zlecenie wykonane pomyslnie.
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Symbol %s nie jest dostepny w terminalu MT5.", symbol)
        return False

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    price = mt5.symbol_info_tick(symbol).ask
    sl = round(price - sl_pips * pip_size, symbol_info.digits)
    tp = round(price + tp_pips * pip_size, symbol_info.digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": magic,
        "comment": "HFT Scalper BUY",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(
        "Wysylanie BUY+SL/TP: %s %.2f lot @ %.5f, SL=%.5f, TP=%.5f",
        symbol, volume, price, sl, tp,
    )

    result = mt5.order_send(request)
    return _validate_order_result(result, "BUY+SL/TP", symbol, volume, price)

