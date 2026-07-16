"""
load_data.py
============
Загрузка, склейка Si-фьючерсов, feature engineering, разметка.

Workflow:
    python load_data.py              # собрать датасет за 3500 дней
    python load_data.py --days 180   # за полгода

Выход: si_patterns_dataset.parquet — готовый датасет для обучения XGBoost.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from moexalgo import Ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт детектора из основного модуля
from config import ScannerConfig
from indicators.base import NoIndicator
from patterns.detector import PatternDetector

# ── Логирование ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ И ВЫЧИСЛЕНИЯ КРАТНОСТИ КОНТРАКТОВ
# ══════════════════════════════════════════════════════════════════════════════

def _third_thursday(year: int, month: int) -> datetime:
    """Третий четверг месяца — правило экспирации квартальных контрактов Si."""
    d = datetime(year, month, 1)
    offset = (3 - d.weekday()) % 7  # weekday(): Mon=0 ... Thu=3
    first_thursday = d + timedelta(days=offset)
    return first_thursday + timedelta(weeks=2)


def _generate_si_contracts(start_year: int, forward_buffer_days: int = 100) -> List[Dict]:
    """
    Генерирует квартальные контракты Si (H=март, M=июнь, U=сентябрь, Z=декабрь)
    от start_year до "сейчас + forward_buffer_days".
    """
    MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
    horizon = datetime.now() + timedelta(days=forward_buffer_days)

    contracts: List[Dict] = []
    year = start_year
    while True:
        stop = False
        for month in (3, 6, 9, 12):
            expiry = _third_thursday(year, month)
            if expiry > horizon:
                stop = True
                break
            contracts.append({
                "ticker": f"Si{MONTH_CODES[month]}{year % 10}",
                "expiry": expiry,
            })
        if stop:
            break
        year += 1
    return contracts


SI_CONTRACTS_START_YEAR = datetime.now().year - 9
SI_CONTRACTS: List[Dict] = _generate_si_contracts(SI_CONTRACTS_START_YEAR)

ROLLOVER_DAYS = 3       # дней до экспирации для переключения

# ── Устойчивость сетевых запросов к MOEX ISS ────────────────────────────────────
FETCH_TIMEOUT_SEC       = 30    # макс. время ожидания одного запроса t.candles()
FETCH_MAX_RETRIES       = 3     # повторных попыток при таймауте/ошибке на контракт
FETCH_RETRY_BACKOFF_SEC = 5     # база экспоненциального backoff
RATE_LIMIT_DELAY_SEC    = 1.0   # пауза между запросами к разным контрактам

TF_PRIMARY     = "15min"
TF_PRIMARY_MIN = 15
TF_HIGHER_MIN  = 60

ATR_PERIOD = 14
ATR_TARGET = 1.5      # движение ≥ 1.5 ATR → label=1
ATR_STOP   = 1.0      # движение ≤ −1.0 ATR → label=0 (стоп)

FORWARD_BARS = 10     # свечей вперёд для разметки

BB_PERIOD = 20
BB_STD    = 2.0

DAYS_BACK_DEFAULT = 3500

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
    """Дата переключения НА следующий контракт."""
    return contract["expiry"] - timedelta(days=ROLLOVER_DAYS)


def load_stitched_si(days_back: int = DAYS_BACK_DEFAULT) -> pd.DataFrame:
    """Загружает 15min свечи Si, склеивая контракты по дате переключения."""
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)

    log.info("Период загрузки: %s → %s",
             start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    segments: List[pd.DataFrame] = []

    for i, contract in enumerate(SI_CONTRACTS):
        seg_start = (
            _rollover_dt(SI_CONTRACTS[i - 1]) if i > 0 else datetime(2000, 1, 1)
        )
        seg_end = _rollover_dt(contract)

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

        time.sleep(RATE_LIMIT_DELAY_SEC)

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


def _call_with_timeout(fn, timeout_sec: float):
    """Запускает fn() в daemon-потоке с таймаутом."""
    box: Dict = {}

    def _runner():
        try:
            box["result"] = fn()
        except Exception as exc:
            box["error"] = exc

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)

    if th.is_alive():
        return None, None, True
    if "error" in box:
        return None, box["error"], False
    return box.get("result"), None, False


def _is_moexalgo_empty_data_bug(error: Exception) -> bool:
    """Распознаёт детерминированный баг moexalgo на пустом ответе."""
    return isinstance(error, AttributeError) and "isoformat" in str(error)


def _fetch_candles(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Загружает 15min свечи одного контракта через moexalgo."""
    def _do_request():
        t = Ticker(ticker)
        return t.candles(
            start  = start.strftime("%Y-%m-%d"),
            end    = end.strftime("%Y-%m-%d"),
            period = TF_PRIMARY,
        )

    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        data, error, timed_out = _call_with_timeout(_do_request, FETCH_TIMEOUT_SEC)

        if timed_out:
            log.warning(
                "[%s] Таймаут запроса (>%ds), попытка %d/%d",
                ticker, FETCH_TIMEOUT_SEC, attempt, FETCH_MAX_RETRIES,
            )
        elif error is not None:
            if _is_moexalgo_empty_data_bug(error):
                log.info(
                    "[%s] Нет данных на MOEX (баг moexalgo на пустом ответе) — пропускаю",
                    ticker,
                )
                return None
            log.warning("[%s] Ошибка запроса: %s — попытка %d/%d", ticker, error, attempt, FETCH_MAX_RETRIES)
        else:
            df = pd.DataFrame(data)
            if df.empty:
                return None

            df = df.rename(columns={"begin": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df[["datetime", "open", "high", "low", "close", "volume"]].copy()

        if attempt < FETCH_MAX_RETRIES:
            backoff = FETCH_RETRY_BACKOFF_SEC * attempt
            log.info("[%s] Жду %.0fс перед повтором...", ticker, backoff)
            time.sleep(backoff)

    log.error("[%s] Не удалось загрузить данные за %d попыток", ticker, FETCH_MAX_RETRIES)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. АГРЕГАЦИЯ В СТАРШИЙ ТФ
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_to_higher_tf(df_15: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует 15min → 1h, оставляя только полные часы."""
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

    bars_per_hour = TF_HIGHER_MIN // TF_PRIMARY_MIN  # = 4
    agg = agg[agg["_count"] == bars_per_hour].drop(columns=["_count"]).reset_index(drop=True)

    log.info("Агрегировано: %d свечей 1h", len(agg))
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 3. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (С УЧЕТОМ ЭКСТРЕМУМОВ)
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет индикаторы: EMA10, ATR14, Bollinger Bands, rolling volume,
    а также 20- и 50-баровые экстремумы (High/Low).
    """
    df = df.copy()
    eps = 1e-9

    # EMA10
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()

    # ATR
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

    # Средний объем
    df["vol_avg20"] = df["volume"].rolling(20).mean()

    # Локальные экстремумы (для расчета расстояний до High/Low)
    df["low_20"]  = df["low"].rolling(20).min()
    df["high_20"] = df["high"].rolling(20).max()
    df["low_50"]  = df["low"].rolling(50).min()
    df["high_50"] = df["high"].rolling(50).max()

    # Заглушка для PatternDetector
    df["_no_indicator"] = np.nan

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTF-СИНХРОНИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def build_htf_features(df_15: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Для каждой 15min-свечи добавляет признаки с закрытой 1h-свечи (без look-ahead)."""
    df_1h_ind = compute_indicators(df_1h).copy()

    df_1h_ind["htf_trend"]     = np.where(df_1h_ind["close"] > df_1h_ind["ema10"], 1.0, -1.0)
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

    # Сдвиг на 1 час вперед (момент закрытия свечи)
    htf["datetime"] = htf["datetime"] + timedelta(hours=1)
    htf = htf.dropna(subset=["datetime"]).sort_values("datetime")

    merged = pd.merge_asof(
        df_15.sort_values("datetime"),
        htf,
        on        = "datetime",
        direction = "backward",
    ).reset_index(drop=True)

    log.info("HTF-признаки добавлены: %d строк", len(merged))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 5. ОБНАРУЖЕНИЕ И РАЗМЕТКА ПАТТЕРНОВ
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Детектирует свечные паттерны по всему датасету."""
    config   = ScannerConfig(indicator=NoIndicator())
    detector = PatternDetector(config)

    records = []
    upper = len(df) - FORWARD_BARS - 1

    # Начинаем с 52-го бара, чтобы хватило истории для rolling(50)
    for idx in range(52, upper):
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
    return result


def label_patterns(df: pd.DataFrame, patterns_df: pd.DataFrame) -> pd.DataFrame:
    """Размечает целевую переменную: 1 (успешный разворот) или 0 (ложный)."""
    labels = []

    for _, row in patterns_df.iterrows():
        idx        = int(row["bar_idx"])
        is_bullish = bool(row["is_bullish"])
        entry      = df.loc[idx, "close"]
        atr_val    = df.loc[idx, "atr"]

        if pd.isna(atr_val) or atr_val <= 0:
            labels.append(np.nan)
            continue

        future = df.iloc[idx + 1 : idx + 1 + FORWARD_BARS]
        if len(future) < FORWARD_BARS:
            labels.append(np.nan)
            continue

        target = ATR_TARGET * atr_val
        stop   = ATR_STOP   * atr_val

        label = 0
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
# 6. FEATURE ENGINEERING (ОПТИМИЗИРОВАННЫЙ НАБОР)
# ══════════════════════════════════════════════════════════════════════════════

def _bars_in_trend(df: pd.DataFrame, idx: int, max_look: int = 20) -> int:
    """Считает количество свечей подряд по одну сторону от EMA10."""
    side  = df.loc[idx, "close"] > df.loc[idx, "ema10"]
    count = 0
    for i in range(idx - 1, max(0, idx - max_look - 1), -1):
        if (df.loc[i, "close"] > df.loc[i, "ema10"]) == side:
            count += 1
        else:
            break
    return count


def extract_features(df: pd.DataFrame, idx: int, pattern: str) -> Optional[Dict]:
    """
    Извлекает оптимизированный вектор признаков.
    Исключены шумные геометрии и дубликаты, добавлены контекстные экстремумы и синергии.
    """
    if idx < 52:  # нужно ≥50 свечей истории для стабильных экстремумов
        return None

    c  = df.iloc[idx]
    eps = 1e-9

    is_bullish = pattern in BULLISH_PATTERNS

    c_body   = abs(c.close - c.open)
    c_range  = c.high - c.low
    avg_body = (
        (df["close"].iloc[idx - 10 : idx] - df["open"].iloc[idx - 10 : idx])
        .abs().mean()
    )

    bars_trend = _bars_in_trend(df, idx)

    # HTF-согласованность
    htf_tr = _safe(c, "htf_trend", 0.0)
    htf_agreement = int(
        (is_bullish and htf_tr > 0) or (not is_bullish and htf_tr < 0)
    )

    vol_ratio = c.volume / (c.vol_avg20 + eps)

    feats: Dict = {
        # ── A. Базовая геометрия текущей свечи ──────────────────────────────────
        "c_body_ratio":  c_body / (c_range + eps),
        "c_rel_body":    c_body / (avg_body + eps),
        "c_is_white":    int(c.close > c.open),

        # ── B. Bollinger Bands (без дубликатов dist_to_upper/lower) ───────────────
        "bb_pct_b": float(c.bb_pct_b),
        "bb_width": float(c.bb_width),

        # ── C. Дистанция до локальных экстремумов (в ATR) ───────────────────────
        "dist_to_low_20":  (c.close - c.low_20)  / (c.atr + eps),
        "dist_to_high_20": (c.high_20 - c.close) / (c.atr + eps),
        "dist_to_low_50":  (c.close - c.low_50)  / (c.atr + eps),
        "dist_to_high_50": (c.high_50 - c.close) / (c.atr + eps),

        # ── D. Объём и Комбо-синергия ─────────────────────────────────────────
        "vol_ratio":     vol_ratio,
        "vol_trend":     (
            df["volume"].iloc[idx - 3 : idx].mean() /
            (df["volume"].iloc[idx - 10 : idx - 3].mean() + eps)
        ),
        "vol_htf_combo": vol_ratio * htf_agreement,

        # ── E. Тренд и динамика ──────────────────────────────────────────────
        "price_vs_ema":      (c.close - c.ema10) / (c.ema10 + eps),
        "ema_slope":         (c.ema10 - df.loc[idx - 5, "ema10"]) / (df.loc[idx - 5, "ema10"] + eps),
        "bars_in_trend":     bars_trend,
        "is_extended_trend": int(bars_trend >= 5),

        # ── F. Волатильность ──────────────────────────────────────────────────
        "atr_pct": c.atr / (c.close + eps),
        "atr_ratio_5_20": (
            df["atr"].iloc[idx - 5 : idx].mean() /
            (df["atr"].iloc[idx - 20 : idx].mean() + eps)
        ),

        # ── G. Старший ТФ (1h) ────────────────────────────────────────────────
        "htf_trend":     htf_tr,
        "htf_ema_slope": _safe(c, "htf_ema_slope", 0.0),
        "htf_bb_pct_b":  _safe(c, "htf_bb_pct_b",  0.5),
        "htf_bb_width":  _safe(c, "htf_bb_width",  0.0),
        "htf_vol_ratio": _safe(c, "htf_vol_ratio", 1.0),
        "htf_agreement": htf_agreement,

        # ── Метаданные ────────────────────────────────────────────────────────
        "pattern":    pattern,
        "is_bullish": int(is_bullish),
    }

    return feats


def _safe(row, col: str, default: float) -> float:
    """Безопасное чтение атрибута строки."""
    val = getattr(row, col, None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


# ══════════════════════════════════════════════════════════════════════════════
# 7. СБОРКА И СТАТИСТИКА
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(days_back: int = DAYS_BACK_DEFAULT) -> pd.DataFrame:
    """Оркестратор сборки датасета."""
    log.info("══ Старт сборки датасета: %d дней ══", days_back)

    df_15 = load_stitched_si(days_back)
    df_1h = aggregate_to_higher_tf(df_15)

    df_15 = compute_indicators(df_15)
    df_15 = build_htf_features(df_15, df_1h)

    patterns_df = detect_all_patterns(df_15)
    if patterns_df.empty:
        return pd.DataFrame()

    patterns_df = label_patterns(df_15, patterns_df)
    before = len(patterns_df)
    patterns_df = patterns_df.dropna(subset=["label"]).reset_index(drop=True)
    log.info("После разметки: %d / %d паттернов (отброшено %d с краёв)", len(patterns_df), before, before - len(patterns_df))

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

    n_features = dataset.shape[1] - 4  # -label, -datetime, -bar_idx, -pattern
    log.info("══ Датасет готов: %d строк, %d признаков ══", len(dataset), n_features)
    log.info(
        "Метки: разворот (1) = %d (%.1f%%) | ложный (0) = %d (%.1f%%)",
        dataset["label"].sum(),       100 * dataset["label"].mean(),
        (dataset["label"] == 0).sum(), 100 * (1 - dataset["label"].mean()),
    )

    return dataset


def print_dataset_summary(dataset: pd.DataFrame) -> None:
    """Печать сводной статистики."""
    if dataset.empty:
        print("Датасет пуст.")
        return

    print("\n" + "═" * 60)
    print("  СТАТИСТИКА ДАТАСЕТА")
    print("═" * 60)
    print(f"  Всего паттернов : {len(dataset)}")
    print(f"  Разворотов  (1) : {int(dataset['label'].sum())} ({100 * dataset['label'].mean():.1f}%)")
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
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка датасета паттернов Si")
    parser.add_argument("--days",   type=int, default=DAYS_BACK_DEFAULT, help="Дней истории для загрузки")
    parser.add_argument("--output", type=str, default="si_patterns_dataset.parquet", help="Путь к результату")
    args = parser.parse_args()

    dataset = build_dataset(days_back=args.days)

    if dataset.empty:
        log.error("Датасет пуст — проверь подключение и наличие данных.")
        sys.exit(1)

    print_dataset_summary(dataset)
    dataset.to_parquet(args.output, index=False)
    log.info("Датасет сохранён: %s", args.output)