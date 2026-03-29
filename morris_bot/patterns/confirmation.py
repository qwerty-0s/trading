"""
Модуль подтверждения паттернов.

Детектор находит паттерн на свече [idx] — это "паттерн-свеча".
Подтверждение проверяется на свече [idx+1] — "подтверждающая свеча".

Использование:
    from morris_bot.patterns.confirmation import needs_confirmation, is_confirmed

    patterns = detector.get_pattern_at_index(df, idx)
    for pattern in patterns:
        if needs_confirmation(pattern):
            if idx + 1 < len(df):
                confirmed = is_confirmed(pattern, df.iloc[idx], df.iloc[idx + 1])
            else:
                confirmed = False  # Свеча ещё не закрылась
        else:
            confirmed = True  # Паттерн самодостаточен
"""

from typing import NamedTuple
import pandas as pd


# ---------------------------------------------------------------------------
# Классификация паттернов по необходимости подтверждения
# ---------------------------------------------------------------------------

# Паттерны, которые НЕ требуют подтверждения — сигнал самодостаточен.
# Three Soldiers/Crows и Engulfing сами по себе являются продолжением/разворотом.
SELF_CONFIRMING = {
    "Bullish Engulfing (Бычье поглощение)",
    "Bearish Engulfing (Медвежье поглощение)",
    "Three White Soldiers (Три белых солдата)",
    "Three Black Crows (Три черные вороны)",
    "Morning Star (Утренняя звезда)",
    "Evening Star (Вечерняя звезда)",
}

# Паттерны, требующие обязательного подтверждения следующей свечой.
NEEDS_CONFIRMATION = {
    # Одиночные тени — слабые без follow-through
    "Hammer (Молот)",
    "Inverted Hammer (Перевернутый молот)",
    "Hanging Man (Висельник)",
    "Shooting Star (Падающая звезда)",
    # Харами — сами по себе лишь пауза, не разворот
    "Bullish Harami (Бычье Харами)",
    "Bearish Harami (Медвежье Харами)",
    "Bullish Harami Cross (Бычий Крест Харами)",
    "Bearish Harami Cross (Медвежий Крест Харами)",
    # Двойные паттерны — нужно follow-through
    "Piercing Line (Просвет в облаках)",
    "Dark Cloud Cover (Темные облака)",
}


def needs_confirmation(pattern: str) -> bool:
    """
    Возвращает True, если паттерн требует подтверждения следующей свечой.
    Паттерны из SELF_CONFIRMING возвращают False.
    Неизвестные паттерны — True (осторожный дефолт).
    """
    if pattern in SELF_CONFIRMING:
        return False
    return True  # NEEDS_CONFIRMATION и любые неизвестные


# ---------------------------------------------------------------------------
# Правила подтверждения
# ---------------------------------------------------------------------------

def _is_bullish_candle(candle) -> bool:
    return candle.close > candle.open


def _is_bearish_candle(candle) -> bool:
    return candle.close < candle.open


def _closes_above(confirm, pattern_candle) -> bool:
    """Подтверждающая свеча закрывается выше максимума тела паттерн-свечи."""
    pattern_body_top = max(pattern_candle.open, pattern_candle.close)
    return confirm.close > pattern_body_top


def _closes_below(confirm, pattern_candle) -> bool:
    """Подтверждающая свеча закрывается ниже минимума тела паттерн-свечи."""
    pattern_body_bottom = min(pattern_candle.open, pattern_candle.close)
    return confirm.close < pattern_body_bottom


# Маппинг: имя паттерна → функция подтверждения (pattern_candle, confirm_candle) -> bool
_CONFIRMATION_RULES = {

    # --- Бычьи одиночные ---
    "Hammer (Молот)": lambda p, c: _is_bullish_candle(c) and c.close > p.close,

    "Inverted Hammer (Перевернутый молот)": lambda p, c: _is_bullish_candle(c) and c.close > p.high,

    # --- Медвежьи одиночные ---
    "Hanging Man (Висельник)": lambda p, c: _is_bearish_candle(c) and c.close < p.close,

    "Shooting Star (Падающая звезда)": lambda p, c: _is_bearish_candle(c) and c.close < p.low,

    # --- Харами бычьи ---
    # Достаточно бычьего закрытия выше тела харами-свечи
    "Bullish Harami (Бычье Харами)": lambda p, c: _is_bullish_candle(c) and _closes_above(c, p),

    "Bullish Harami Cross (Бычий Крест Харами)": lambda p, c: _is_bullish_candle(c) and _closes_above(c, p),

    # --- Харами медвежьи ---
    "Bearish Harami (Медвежье Харами)": lambda p, c: _is_bearish_candle(c) and _closes_below(c, p),

    "Bearish Harami Cross (Медвежий Крест Харами)": lambda p, c: _is_bearish_candle(c) and _closes_below(c, p),

    # --- Двойные паттерны ---
    # Piercing Line уже наполовину подтверждён структурой, но нужен продолжающий рост
    "Piercing Line (Просвет в облаках)": lambda p, c: _is_bullish_candle(c) and c.close > p.close,

    "Dark Cloud Cover (Темные облака)": lambda p, c: _is_bearish_candle(c) and c.close < p.close,
}


def is_confirmed(pattern: str, pattern_candle, confirm_candle) -> bool:
    """
    Проверяет, подтверждён ли паттерн следующей свечой.

    Args:
        pattern:        Название паттерна (строка из PatternDetector).
        pattern_candle: Строка df — свеча, на которой найден паттерн.
        confirm_candle: Строка df — следующая (подтверждающая) свеча.

    Returns:
        True  — паттерн подтверждён, сигнал валиден.
        False — подтверждения нет, сигнал игнорируем.
    """
    if not needs_confirmation(pattern):
        return True  # Самодостаточный паттерн всегда подтверждён

    rule = _CONFIRMATION_RULES.get(pattern)
    if rule is None:
        # Неизвестный паттерн — пропускаем без подтверждения (осторожный дефолт)
        return False

    try:
        return bool(rule(pattern_candle, confirm_candle))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Удобная функция для воркера / бэктеста
# ---------------------------------------------------------------------------

def filter_confirmed(
    patterns: list[str],
    df: pd.DataFrame,
    pattern_idx: int,
) -> list[str]:
    """
    Фильтрует список паттернов, оставляя только подтверждённые.

    Для самодостаточных паттернов подтверждение не нужно.
    Для остальных нужна закрытая свеча [pattern_idx + 1].
    Если подтверждающая свеча ещё недоступна — паттерн отбрасывается.

    Args:
        patterns:    Список паттернов от PatternDetector.get_pattern_at_index().
        df:          Датафрейм со свечами.
        pattern_idx: Индекс свечи, на которой найдены паттерны.

    Returns:
        Отфильтрованный список подтверждённых паттернов.
    """
    confirmed = []
    confirm_idx = pattern_idx + 1

    for pattern in patterns:
        if not needs_confirmation(pattern):
            confirmed.append(pattern)
            continue

        if confirm_idx >= len(df):
            # Подтверждающая свеча ещё не закрылась — ждём
            continue

        pattern_candle  = df.iloc[pattern_idx]
        confirm_candle  = df.iloc[confirm_idx]

        if is_confirmed(pattern, pattern_candle, confirm_candle):
            confirmed.append(pattern)

    return confirmed
