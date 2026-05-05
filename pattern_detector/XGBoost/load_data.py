"""
load_data.py
============
Загрузка, склейка Si-фьючерсов, feature engineering, разметка.

Workflow:
    python load_data.py              # собрать датасет за 90 дней
    python load_data.py --days 60    # короче

Выход: si_patterns_dataset.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from moexalgo import Ticker

# Импорт детектора из основного модуля (лежит рядом)
from patterns.detector import PatternDetector
from indicators.base import NoIndicator
from config import ScannerConfig


# ── Логирование ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════

# Контракты в хронологическом порядке экспирации.
# Переход на следующий контракт — за ROLLOVER_DAYS дней до экспирации текущего.
SI_CONTRACTS: List[Dict] = [
    {"ticker": "SiZ5", "expiry": datetime(2025, 12, 18)},
    {"ticker": "SiH6", "expiry": datetime(2026,  3, 19)},
    {"ticker": "SiM6", "expiry": datetime(2026,  6, 18)},
]
ROLLOVER_DAYS = 3       # дней до экспирации для переключения

TF_PRIMARY  = "15min"  # основной таймфрейм (строка для moexalgo)
TF_PRIMARY_MIN = 15    # в минутах (для агрегации)
TF_HIGHER_MIN  = 60    # старший ТФ в минутах

ATR_PERIOD  = 14
ATR_TARGET  = 1.5      # движение ≥ 1.5 ATR → label=1
ATR_STOP    = 1.0      # движение ≤ −1.0 ATR → label=0 (стоп)

FORWARD_BARS = 10      # свечей вперёд для разметки

BB_PERIOD = 20
BB_STD    = 2.0

DAYS_BACK_DEFAULT = 90

# Списки паттернов — используем как справочник для классификации и признаков
BULLISH_PATTERNS: set = {
    "Hammer (Молот)",
    "Inverted Hammer (Перевернутый молот)",
    "Bullish Engulfing (Бычье поглощение)",
    "Bullish Harami (Бычье Харами)",
    "Bullish Harami Cross (Бычий Крест Харами)",
    "Piercing Line (Просвет в облаках)",
    "Morning Star (Утренняя звезда)",
    "Three White Soldiers (Три белых солдата)",
}
BEARISH_PATTERNS: set = {
    "Hanging Man (Висельник)",
    "Shooting Star (Падающая звезда)",
    "Bearish Engulfing (Медвежье поглощение)",
    "Bearish Harami (Медвежье Харами)",
    "Bearish Harami Cross (Медвежий Крест Харами)",
    "Dark Cloud Cover (Темные облака)",
    "Evening Star (Вечерняя звезда)",
    "Three Black Crows (Три черные вороны)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА И СКЛЕЙКА КОНТРАКТОВ
# ══════════════════════════════════════════════════════════════════════════════

def _rollover_dt(contract: Dict) -> datetime:
    """Дата переключения НА следующий контракт (за ROLLOVER_DAYS до экспирации)."""
    return contract["expiry"] - timedelta(days=ROLLOVER_DAYS)


def load_stitched_si(days_back: int = DAYS_BACK_DEFAULT) -> pd.DataFrame:
    """
    Загружает 15min свечи Si, склеивая контракты по дате переключения.

    Пример при days_back=90 от 02.05.2026:
        SiH6 : ~22.01.2026 → 16.03.2026   (rollover SiH6 = 19.03 - 3д = 16.03)
        SiM6 : 16.03.2026  → 02.05.2026   (текущий активный)

    Возвращает DataFrame: datetime, open, high, low, close, volume
    """
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)

    log.info("Период загрузки: %s → %s",
             start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    segments: List[pd.DataFrame] = []

    for i, contract in enumerate(SI_CONTRACTS):
        # Сегмент начинается от rollover предыдущего контракта (или от start_dt)
        seg_start = (
            _rollover_dt(SI_CONTRACTS[i - 1]) if i > 0 else datetime(2000, 1, 1)
        )
        # Сегмент заканчивается на нашем rollover (или end_dt для последнего)
        seg_end = _rollover_dt(contract)

        # Обрезаем по запрошенному диапазону
        seg_start = max(seg_start, start_dt)
        seg_end   = min(seg_end, end_dt)

        if seg_start >= seg_end:
            log.debug("Пропускаю %s (вне диапазона)", contract["ticker"])
            continue

        log.info("Загрузка %s: %s → %s",
                 contract["ticker"],
                 seg_start.strftime("%Y-%m-%d"),
                 seg_end.strftime("%Y-%m-%d"))

        df = _fetch_candles(contract["ticker"], seg_start, seg_end)
        if df is not None and not df.empty:
            segments.append(df)
            log.info("  → %d свечей 15min", len(df))
        else:
            log.warning("  → нет данных для %s", contract["ticker"])

    if not segments:
        raise RuntimeError(
            "Не удалось загрузить данные ни по одному контракту Si. "
            "Проверь подключение и наличие тикеров в MOEX."
        )

    result = (
        pd.concat(segments, ignore_index=True)
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )
    log.info("Итого склеено: %d свечей 15min", len(result))
    return result


def _fetch_candles(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Загружает 15min свечи одного контракта через moexalgo."""
    try:
        t    = Ticker(ticker)
        data = t.candles(
            start  = start.strftime("%Y-%m-%d"),
            end    = end.strftime("%Y-%m-%d"),
            period = TF_PRIMARY,
        )
        df = pd.DataFrame(data)
        if df.empty:
            return None

        # moexalgo называет время открытия 'begin'
        df = df.rename(columns={"begin": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["datetime", "open", "high", "low", "close", "volume"]].copy()

    except Exception as exc:
        log.error("[%s] Ошибка загрузки: %s", ticker, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. АГРЕГАЦИЯ В СТАРШИЙ ТФ
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_to_higher_tf(df_15: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует 15min → 1h, группируя по выровненному часу.
    Включает только полные 1h-свечи (ровно 4 исходных свечи в группе).

    Возвращает DataFrame: datetime, open, high, low, close, volume
    datetime = время открытия 1h-свечи.
    """
    df = df_15.copy().sort_values("datetime").reset_index(drop=True)
    df["_hour"] = df["datetime"].dt.floor("1h")

    agg = (
        df.groupby("_hour")
        .agg(
            open   = ("open",   "first"),
            high   = ("high",   "max"),
            low    = ("low",    "min"),
            close  = ("close",  "last"),
            volume = ("volume", "sum"),
            _count = ("close",  "count"),
        )
        .reset_index()
        .rename(columns={"_hour": "datetime"})
    )

    # Только полные часы: ровно TF_HIGHER_MIN / TF_PRIMARY_MIN свечей
    bars_per_hour = TF_HIGHER_MIN // TF_PRIMARY_MIN   # = 4
    agg = agg[agg["_count"] == bars_per_hour].drop(columns=["_count"])
    agg = agg.reset_index(drop=True)

    log.info("Агрегировано: %d свечей 1h", len(agg))
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 3. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame колонки:
        ema10, atr, bb_mid, bb_upper, bb_lower, bb_width, bb_pct_b,
        vol_avg20, _no_indicator (нужна PatternDetector при NoIndicator)
    """
    df = df.copy()
    eps = 1e-9

    # EMA10 — нужна PatternDetector для определения тренда
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()

    # ATR (Wilder's smoothing = EWM с com = period-1)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_PERIOD - 1, min_periods=ATR_PERIOD).mean()

    # Bollinger Bands
    df["bb_mid"]   = df["close"].rolling(BB_PERIOD).mean()
    bb_std         = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + eps)
    df["bb_pct_b"] = (
        (df["close"] - df["bb_lower"]) /
        (df["bb_upper"] - df["bb_lower"] + eps)
    )

    # Средний объём за 20 свечей
    df["vol_avg20"] = df["volume"].rolling(20).mean()

    # Заглушка для NoIndicator — PatternDetector смотрит эту колонку
    df["_no_indicator"] = np.nan

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTF-СИНХРОНИЗАЦИЯ (без look-ahead bias)
# ══════════════════════════════════════════════════════════════════════════════

def build_htf_features(df_15: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой 15min-свечи добавляет признаки с закрытой 1h-свечи.

    Правило синхронизации:
        1h-свеча c open_time=T закрывается в T+60min.
        Значит, для 15min-свечи в момент t мы можем использовать
        1h-свечу только если её open_time + 60min <= t.

    Реализация: сдвигаем "время доступности" 1h-свечи на +60min,
    затем merge_asof(..., direction="backward") — берём последнюю
    доступную запись, не позже текущей 15min-свечи.
    """
    df_1h_ind = compute_indicators(df_1h).copy()

    # Признаки HTF, которые нас интересуют
    df_1h_ind["htf_trend"]     = np.where(
        df_1h_ind["close"] > df_1h_ind["ema10"], 1.0, -1.0
    )
    df_1h_ind["htf_ema_slope"] = (
        df_1h_ind["ema10"] - df_1h_ind["ema10"].shift(5)
    ) / (df_1h_ind["ema10"].shift(5) + 1e-9)
    df_1h_ind["htf_bb_pct_b"]  = df_1h_ind["bb_pct_b"]
    df_1h_ind["htf_bb_width"]  = df_1h_ind["bb_width"]
    df_1h_ind["htf_atr"]       = df_1h_ind["atr"]
    df_1h_ind["htf_vol_ratio"] = df_1h_ind["volume"] / (df_1h_ind["vol_avg20"] + 1e-9)

    htf = df_1h_ind[[
        "datetime",
        "htf_trend", "htf_ema_slope",
        "htf_bb_pct_b", "htf_bb_width",
        "htf_atr", "htf_vol_ratio",
    ]].copy()

    # Сдвигаем datetime → момент закрытия свечи (когда данные реально доступны)
    htf["datetime"] = htf["datetime"] + timedelta(hours=1)
    htf = htf.dropna(subset=["datetime"]).sort_values("datetime")

    df_15_sorted = df_15.sort_values("datetime")

    merged = pd.merge_asof(
        df_15_sorted,
        htf,
        on        = "datetime",
        direction = "backward",   # берём последнюю закрытую 1h-свечу
    ).reset_index(drop=True)

    htf_cols = ["htf_trend", "htf_ema_slope", "htf_bb_pct_b",
                "htf_bb_width", "htf_atr", "htf_vol_ratio"]
    n_missing = merged[htf_cols].isna().all(axis=1).sum()
    if n_missing > 0:
        log.warning("HTF: %d свечей без данных старшего ТФ (начало истории)", n_missing)

    log.info("HTF-признаки добавлены: %d строк", len(merged))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 5. ОБНАРУЖЕНИЕ ПАТТЕРНОВ
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Прогоняет PatternDetector по всему df с NoIndicator.
    Останавливается за FORWARD_BARS свечей до конца — чтобы хватило
    данных для разметки каждого паттерна.

    Возвращает DataFrame: bar_idx, datetime, pattern, is_bullish, close, atr
    """
    config   = ScannerConfig(indicator=NoIndicator())
    detector = PatternDetector(config)

    records = []
    # Верхняя граница: оставляем FORWARD_BARS свечей для разметки
    upper = len(df) - FORWARD_BARS - 1

    for idx in range(12, upper):
        patterns = detector.get_pattern_at_index(df, idx)
        for p in patterns:
            records.append({
                "bar_idx":    idx,
                "datetime":   df.loc[idx, "datetime"],
                "pattern":    p,
                "is_bullish": int(p in BULLISH_PATTERNS),
                "close":      df.loc[idx, "close"],
                "atr":        df.loc[idx, "atr"],
            })

    result = pd.DataFrame(records)

    if result.empty:
        log.warning("Паттерны не обнаружены!")
        return result

    log.info("Обнаружено паттернов: %d", len(result))

    counts = result.groupby("pattern").size().sort_values(ascending=False)
    log.info(
        "Распределение по типам:\n%s",
        counts.to_string()
    )
    log.info(
        "Бычьих: %d | Медвежьих: %d",
        result["is_bullish"].sum(),
        (result["is_bullish"] == 0).sum(),
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 6. РАЗМЕТКА (LABELING)
# ══════════════════════════════════════════════════════════════════════════════

def label_patterns(df: pd.DataFrame, patterns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Размечает каждый паттерн, смотря FORWARD_BARS свечей вперёд.

    Логика для БЫЧЬИХ паттернов:
        Смотрим свечи вперёд по очереди.
        Если high >= entry_close + 1.5*ATR → label=1 (разворот)
        Если low  <= entry_close − 1.0*ATR → label=0 (стоп, ложный сигнал)
        Если за FORWARD_BARS свечей ни одно не сработало → label=0

    Логика для МЕДВЕЖЬИХ паттернов — зеркально.

    NaN ставим только если данных для разметки недостаточно (край датасета).
    Такие строки потом отбрасываются.
    """
    labels = []

    for _, row in patterns_df.iterrows():
        idx        = int(row["bar_idx"])
        is_bullish = bool(row["is_bullish"])
        entry      = df.loc[idx, "close"]
        atr_val    = df.loc[idx, "atr"]

        # Нет ATR — отбрасываем
        if pd.isna(atr_val) or atr_val <= 0:
            labels.append(np.nan)
            continue

        future = df.iloc[idx + 1 : idx + 1 + FORWARD_BARS]

        # Недостаточно данных вперёд — отбрасываем
        if len(future) < FORWARD_BARS:
            labels.append(np.nan)
            continue

        target = ATR_TARGET * atr_val
        stop   = ATR_STOP   * atr_val

        label = 0  # по умолчанию — ложный сигнал

        if is_bullish:
            for _, fc in future.iterrows():
                if fc["high"] >= entry + target:
                    label = 1
                    break
                if fc["low"] <= entry - stop:
                    label = 0
                    break
        else:
            for _, fc in future.iterrows():
                if fc["low"] <= entry - target:
                    label = 1
                    break
                if fc["high"] >= entry + stop:
                    label = 0
                    break

        labels.append(label)

    result = patterns_df.copy()
    result["label"] = labels
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _bars_in_trend(df: pd.DataFrame, idx: int, max_look: int = 20) -> int:
    """
    Сколько свечей подряд цена была по одну сторону от EMA10.
    Показывает "усталость" тренда.
    """
    side  = df.loc[idx, "close"] > df.loc[idx, "ema10"]
    count = 0
    for i in range(idx - 1, max(0, idx - max_look - 1), -1):
        if (df.loc[i, "close"] > df.loc[i, "ema10"]) == side:
            count += 1
        else:
            break
    return count


def _pattern_cleanliness(
    body: float,
    rng: float,
    upper_sh: float,
    lower_sh: float,
    avg_body: float,
    is_bullish: bool,
) -> float:
    """
    Числовая оценка "чистоты" паттерна [0, 1].
    Отражает насколько конкретная свеча близка к идеальному паттерну.

    Для бычьих:  хорошо — большое тело или длинная нижняя тень.
    Для медвежьих: большое тело или длинная верхняя тень.
    Плюс бонус за размер тела относительно исторического среднего.
    """
    eps = 1e-9
    if rng < eps:
        return 0.0

    if is_bullish:
        shape_score = 0.5 * (body / (rng + eps)) + 0.5 * (lower_sh / (rng + eps))
    else:
        shape_score = 0.5 * (body / (rng + eps)) + 0.5 * (upper_sh / (rng + eps))

    # Насколько тело крупнее среднего (кэп на 3x)
    rel_body = min(body / (avg_body + eps), 3.0) / 3.0

    return float((shape_score + rel_body) / 2.0)


def extract_features(df: pd.DataFrame, idx: int, pattern: str) -> Optional[Dict]:
    """
    Извлекает вектор числовых признаков для свечи на позиции idx.
    Возвращает None если данных недостаточно.

    Блоки признаков:
        A — геометрия текущей свечи и "чистота" паттерна
        B — контекст (предыдущие 2 свечи)
        C — Bollinger Bands
        D — объём
        E — тренд и позиция цены
        F — волатильность
        G — старший ТФ
        H — время (внутридневная сезонность)
    """
    if idx < 22:   # нужно ≥22 свечей истории для стабильных индикаторов
        return None

    c  = df.iloc[idx]
    p  = df.iloc[idx - 1]
    pp = df.iloc[idx - 2]
    eps = 1e-9

    is_bullish = pattern in BULLISH_PATTERNS

    # Геометрия
    c_body     = abs(c.close - c.open)
    c_range    = c.high - c.low
    c_top      = max(c.open, c.close)
    c_bottom   = min(c.open, c.close)
    c_upper_sh = c.high - c_top
    c_lower_sh = c_bottom - c.low

    p_body  = abs(p.close - p.open)
    pp_body = abs(pp.close - pp.open)

    avg_body = (
        (df["close"].iloc[idx - 10 : idx] - df["open"].iloc[idx - 10 : idx])
        .abs().mean()
    )

    feats: Dict = {
        # ── A. Геометрия текущей свечи ───────────────────────────────────────
        "c_body_ratio":        c_body     / (c_range + eps),
        "c_upper_sh_ratio":    c_upper_sh / (c_range + eps),
        "c_lower_sh_ratio":    c_lower_sh / (c_range + eps),
        "c_rel_body":          c_body     / (avg_body + eps),
        "c_is_white":          int(c.close > c.open),
        "pattern_cleanliness": _pattern_cleanliness(
            c_body, c_range, c_upper_sh, c_lower_sh, avg_body, is_bullish
        ),

        # ── B. Контекст: предыдущие свечи ────────────────────────────────────
        "p_body_ratio":   p_body  / (c_range + eps),
        "pp_body_ratio":  pp_body / (c_range + eps),
        "p_is_white":     int(p.close > p.open),
        "pp_is_white":    int(pp.close > pp.open),
        # Совпадение направления: все три свечи одного цвета — сильный импульс
        "same_color_streak": int(
            (c.close > c.open) == (p.close > p.open) == (pp.close > pp.open)
        ),

        # ── C. Bollinger Bands ────────────────────────────────────────────────
        "bb_pct_b":      float(c.bb_pct_b),
        "bb_width":      float(c.bb_width),
        "dist_to_upper": (c.bb_upper - c.close) / (c.close + eps),
        "dist_to_lower": (c.close - c.bb_lower) / (c.close + eps),

        # ── D. Объём ──────────────────────────────────────────────────────────
        # vol_ratio > 1 = повышенный объём → паттерн надёжнее
        "vol_ratio": c.volume / (c.vol_avg20 + eps),
        # Динамика объёма: растёт или падает
        "vol_trend": (
            df["volume"].iloc[idx - 3 : idx].mean() /
            (df["volume"].iloc[idx - 10 : idx - 3].mean() + eps)
        ),

        # ── E. Тренд и позиция цены ───────────────────────────────────────────
        "price_vs_ema":  (c.close - c.ema10)        / (c.ema10 + eps),
        "ema_slope":     (c.ema10 - df.loc[idx - 5, "ema10"]) / (df.loc[idx - 5, "ema10"] + eps),
        "bars_in_trend": _bars_in_trend(df, idx),

        # ── F. Волатильность ──────────────────────────────────────────────────
        "atr_pct":      c.atr / (c.close + eps),        # нормированный ATR
        "atr_ratio_5_20": (                              # сужение/расширение
            df["atr"].iloc[idx - 5 : idx].mean() /
            (df["atr"].iloc[idx - 20 : idx].mean() + eps)
        ),

        # ── G. Старший ТФ (1h) ────────────────────────────────────────────────
        "htf_trend":     _safe(c, "htf_trend",     0.0),
        "htf_ema_slope": _safe(c, "htf_ema_slope", 0.0),
        "htf_bb_pct_b":  _safe(c, "htf_bb_pct_b",  0.5),
        "htf_bb_width":  _safe(c, "htf_bb_width",  0.0),
        "htf_vol_ratio": _safe(c, "htf_vol_ratio", 1.0),
        # 1 если HTF тренд согласован с паттерном, 0 если противоположен
        "htf_agreement": int(
            (is_bullish  and _safe(c, "htf_trend", 0.0) > 0) or
            (not is_bullish and _safe(c, "htf_trend", 0.0) < 0)
        ),

        # ── H. Время (внутридневная сезонность) ───────────────────────────────
        "hour":        c.datetime.hour,
        "day_of_week": c.datetime.dayofweek,

        # ── Метаданные (не фичи, используются для анализа) ───────────────────
        "pattern":    pattern,
        "is_bullish": int(is_bullish),
    }

    return feats


def _safe(row, col: str, default: float) -> float:
    """Безопасное чтение колонки из строки датафрейма."""
    val = getattr(row, col, None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


# ══════════════════════════════════════════════════════════════════════════════
# 8. СБОРКА ДАТАСЕТА
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(days_back: int = DAYS_BACK_DEFAULT) -> pd.DataFrame:
    """
    Главная функция. Оркестрирует весь pipeline:
        загрузка → агрегация → индикаторы → HTF-синхронизация
        → детекция паттернов → разметка → feature engineering

    Возвращает готовый датасет.
    """
    log.info("══ Старт сборки датасета: %d дней ══", days_back)

    # 1. Загрузка и склейка 15min
    df_15 = load_stitched_si(days_back)

    # 2. Агрегация в 1h
    df_1h = aggregate_to_higher_tf(df_15)

    # 3. Индикаторы на 15min
    df_15 = compute_indicators(df_15)

    # 4. HTF-признаки (no look-ahead)
    df_15 = build_htf_features(df_15, df_1h)
    df_15 = df_15.reset_index(drop=True)

    log.info("df_15 готов: %d строк, %d колонок", *df_15.shape)

    # 5. Детекция паттернов (NoIndicator)
    patterns_df = detect_all_patterns(df_15)
    if patterns_df.empty:
        return pd.DataFrame()

    # 6. Разметка
    patterns_df = label_patterns(df_15, patterns_df)
    before = len(patterns_df)
    patterns_df = patterns_df.dropna(subset=["label"]).reset_index(drop=True)
    log.info(
        "После разметки: %d / %d паттернов (отброшено %d с краёв)",
        len(patterns_df), before, before - len(patterns_df),
    )

    # 7. Извлечение признаков
    feature_rows = []
    skipped = 0
    for _, row in patterns_df.iterrows():
        idx   = int(row["bar_idx"])
        feats = extract_features(df_15, idx, row["pattern"])
        if feats is None:
            skipped += 1
            continue
        feats["label"]    = int(row["label"])
        feats["datetime"] = row["datetime"]
        feats["bar_idx"]  = idx
        feature_rows.append(feats)

    if skipped:
        log.warning("Пропущено при извлечении признаков: %d", skipped)

    dataset = pd.DataFrame(feature_rows)

    # ── Итоговая статистика ───────────────────────────────────────────────────
    n_features = dataset.shape[1] - 4  # -label, -datetime, -bar_idx, -pattern
    log.info("══ Датасет готов: %d строк, %d признаков ══", len(dataset), n_features)
    log.info(
        "Метки: разворот (1) = %d (%.1f%%) | ложный (0) = %d (%.1f%%)",
        dataset["label"].sum(),       100 * dataset["label"].mean(),
        (dataset["label"] == 0).sum(), 100 * (1 - dataset["label"].mean()),
    )

    return dataset


def print_dataset_summary(dataset: pd.DataFrame) -> None:
    """Печатает подробную статистику по датасету."""
    if dataset.empty:
        print("Датасет пуст.")
        return

    print("\n" + "═" * 60)
    print("  СТАТИСТИКА ДАТАСЕТА")
    print("═" * 60)
    print(f"  Всего паттернов : {len(dataset)}")
    print(f"  Разворотов  (1) : {int(dataset['label'].sum())}"
          f"  ({100 * dataset['label'].mean():.1f}%)")
    print(f"  Ложных      (0) : {int((dataset['label'] == 0).sum())}")

    print("\n  По типам паттернов:")
    summary = (
        dataset.groupby("pattern")["label"]
        .agg(count="count", reversal_rate="mean")
        .sort_values("count", ascending=False)
    )
    summary["reversal_rate"] = (summary["reversal_rate"] * 100).round(1).astype(str) + "%"
    print(summary.to_string())

    print("\n  HTF-согласованность:")
    print(dataset.groupby("htf_agreement")["label"].agg(["count", "mean"]).to_string())

    print("\n  По часам торговли:")
    hourly = dataset.groupby("hour")["label"].agg(count="count", reversal_rate="mean")
    hourly["reversal_rate"] = (hourly["reversal_rate"] * 100).round(1)
    print(hourly.to_string())

    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка датасета паттернов Si")
    parser.add_argument("--days",   type=int, default=DAYS_BACK_DEFAULT,
                        help="Дней истории для загрузки (default: 90)")
    parser.add_argument("--output", type=str, default="si_patterns_dataset.parquet",
                        help="Путь для сохранения датасета")
    args = parser.parse_args()

    dataset = build_dataset(days_back=args.days)

    if dataset.empty:
        log.error("Датасет пуст — проверь подключение и наличие данных.")
        sys.exit(1)

    print_dataset_summary(dataset)

    dataset.to_parquet(args.output, index=False)
    log.info("Датасет сохранён: %s", args.output)
    log.info("Загрузка: pd.read_parquet('%s')", args.output)