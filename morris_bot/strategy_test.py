"""
strategy_test.py — полный стратегический бэктест для MorrisBot.

Логика сделки
─────────────
  Вход    : открытие свечи [pattern_idx + offset]
              offset=2 при use_pattern_confirmation=True, иначе 1
  Стоп    : зависит от sl_mode:
              'lookback' — min(low) / max(high) последних sl_lookback свечей
              'atr'      — entry ± ATR(atr_period) × atr_sl_mult
  Тейк    : зависит от tp_mode:
              'fixed_rr' — entry ± sl_dist × rr_multiplier   ← ОСНОВНОЙ РЕЖИМ
              'ema10'    — EMA10 в момент входа (классический режим)
              'atr'      — entry ± ATR × atr_tp_mult
  Выход   : первое касание TP или SL по high/low свечи
  Таймаут : close последней свечи окна
  Трейлинг: если trailing_stop=True, SL подтягивается после каждого нового
              максимума/минимума (шаг = trailing_step_r × исходный sl_dist)
  Частичный тейк: если partial_take=True, при достижении 1R закрывается 50%
              позиции, SL переносится в безубыток

Параметры
─────────
  tp_mode          : 'fixed_rr' | 'ema10' | 'atr'
  sl_mode          : 'lookback' | 'atr'
  rr_multiplier    : цель в R при tp_mode='fixed_rr'  (default 2.0)
  atr_period       : период ATR                       (default 14)
  atr_sl_mult      : ATR × коэфф. для SL при sl_mode='atr'
  atr_tp_mult      : ATR × коэфф. для TP при tp_mode='atr'
  min_rr           : минимальное теоретическое R:R; сделки ниже пропускаются
  trailing_stop    : True — динамический трейлинг-стоп
  trailing_step_r  : шаг трейлинга в долях от исходного sl_dist
  partial_take     : True — 50% на 1R, остаток до полного TP
"""

import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from morris_bot.config import ScannerConfig
from morris_bot.indicators.base import BaseIndicator, NoIndicator
from morris_bot.indicators.dual import DualConfirmIndicator
from morris_bot.indicators.bollinger import BollingerPercentBIndicator
from morris_bot.indicators.mfi import AdaptiveMFIIndicator
from morris_bot.indicators.rsi import RSIIndicator
from morris_bot.patterns.detector import PatternDetector
from morris_bot.patterns.confirmation import filter_confirmed, needs_confirmation
from morris_bot.backtest import _fetch_data, _prepare_df, _is_bullish

warnings.filterwarnings("ignore")


# ==============================================================================
# ПАРАМЕТРЫ СТРАТЕГИИ
# ==============================================================================

@dataclass
class StrategyParams:
    """
    Параметры стратегического бэктеста.

    ── Стоп-лосс ────────────────────────────────────────────────────────────
    sl_mode         : 'lookback' — локальный экстремум (классика)
                      'atr'      — entry ± ATR × atr_sl_mult (адаптивный)
    sl_lookback     : кол-во свечей для lookback-стопа
    atr_period      : период ATR (используется в обоих sl_mode как контекст)
    atr_sl_mult     : множитель ATR для SL при sl_mode='atr'

    ── Тейк-профит ──────────────────────────────────────────────────────────
    tp_mode         : 'fixed_rr' — entry ± sl_dist × rr_multiplier  ← ОСНОВНОЙ
                      'ema10'    — EMA10 (старый режим, нестабильный R:R)
                      'atr'      — entry ± ATR × atr_tp_mult
    rr_multiplier   : целевой R:R при tp_mode='fixed_rr'
    atr_tp_mult     : множитель ATR для TP при tp_mode='atr'

    ── Управление позицией ───────────────────────────────────────────────────
    max_candles         : таймаут в свечах
    simultaneous_hit    : 'sl_first' | 'tp_first' при одновременном касании
    trailing_stop       : True — трейлинг-стоп
    trailing_step_r     : шаг трейлинга = trailing_step_r × sl_dist (0.5 = каждые 0.5R)
    partial_take        : True — 50% на 1R, SL → безубыток, остаток до TP
    min_rr              : минимально допустимое теоретическое R:R (иначе пропуск)

    ── Фильтры сигналов ─────────────────────────────────────────────────────
    use_indicator           : применять фильтр индикатора
    use_pattern_confirmation: подтверждение следующей свечой
    one_per_candle          : не более одной позиции каждого направления на свечу
    min_risk_pct            : минимальный риск в % от входа
    max_risk_pct            : максимальный риск в % от входа
    """
    # Стоп-лосс
    sl_mode         : str   = 'lookback'
    sl_lookback     : int   = 5
    atr_period      : int   = 14
    atr_sl_mult     : float = 1.5

    # Тейк-профит
    tp_mode         : str   = 'fixed_rr'
    rr_multiplier   : float = 1.5
    atr_tp_mult     : float = 3.0

    # Управление позицией
    max_candles         : int   = 20
    simultaneous_hit    : str   = 'sl_first'
    trailing_stop       : bool  = False
    trailing_step_r     : float = 0.5
    partial_take        : bool  = False

    # Фильтры качества сделки
    min_rr              : float = 0.8   # пропускаем сделки с теор. R:R < порога
    min_risk_pct        : float = 0.05
    max_risk_pct        : float = 5.0

    # Фильтры сигналов
    use_indicator           : bool = True
    use_pattern_confirmation: bool = True
    one_per_candle          : bool = True


# ==============================================================================
# РЕЗУЛЬТАТ СДЕЛКИ
# ==============================================================================

@dataclass
class Trade:
    """Результат одной сделки."""
    pattern      : str
    direction    : str           # 'bullish' | 'bearish'
    pattern_idx  : int
    entry_idx    : int
    exit_idx     : int
    entry_time   : object
    exit_time    : object
    entry_price  : float
    sl_price     : float
    tp_price     : float
    sl_risk_pct  : float         # abs(entry - sl) / entry × 100
    tp_dist_pct  : float         # abs(tp - entry) / entry × 100
    rr_ratio     : float         # tp_dist_pct / sl_risk_pct
    exit_price   : float
    exit_reason  : str           # 'tp' | 'tp_partial' | 'sl' | 'timeout'
    pnl_pct      : float         # прибыль/убыток в %
    r_multiple   : float         # pnl_pct / sl_risk_pct
    atr_at_entry : float         # ATR на момент входа (для справки)


# ==============================================================================
# ATR
# ==============================================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Средний истинный диапазон (Average True Range).
    Вычисляется «с нуля», не требуя внешней библиотеки.
    """
    high  = df['high'].astype(float)
    low   = df['low'].astype(float)
    close = df['close'].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def _ensure_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Добавляет колонку atr_{period} в DataFrame, если ещё нет."""
    col = f'atr_{period}'
    if col not in df.columns:
        df = df.copy()
        df[col] = _compute_atr(df, period)
    return df


# ==============================================================================
# SL / TP
# ==============================================================================

def _compute_sl(df: pd.DataFrame,
                pattern_idx: int,
                params: StrategyParams,
                direction: str,
                entry_price: float) -> float:
    """
    Вычисляет цену стоп-лосса согласно params.sl_mode.

    lookback : min(low) за sl_lookback свечей (бычий) / max(high) (медвежий)
    atr      : entry ± ATR(atr_period) × atr_sl_mult
    """
    if params.sl_mode == 'atr':
        col = f'atr_{params.atr_period}'
        atr_val = float(df.iloc[pattern_idx][col])
        if direction == 'bullish':
            return entry_price - atr_val * params.atr_sl_mult
        else:
            return entry_price + atr_val * params.atr_sl_mult
    else:  # 'lookback'
        start  = max(0, pattern_idx - params.sl_lookback + 1)
        window = df.iloc[start: pattern_idx + 1]
        if direction == 'bullish':
            return float(window['low'].min())
        else:
            return float(window['high'].max())


def _compute_tp(entry_price: float,
                sl_price: float,
                direction: str,
                params: StrategyParams,
                atr_val: float) -> float:
    """
    Вычисляет цену тейк-профита согласно params.tp_mode.

    fixed_rr : entry ± sl_dist × rr_multiplier
    ema10    : используется EMA10 — передаётся через atr_val=ema10 (backward compat)
    atr      : entry ± ATR × atr_tp_mult
    """
    sl_dist = abs(entry_price - sl_price)

    if params.tp_mode == 'fixed_rr':
        if direction == 'bullish':
            return entry_price + sl_dist * params.rr_multiplier
        else:
            return entry_price - sl_dist * params.rr_multiplier

    elif params.tp_mode == 'ema10':
        # atr_val передаётся как ema10 при этом режиме
        return atr_val

    elif params.tp_mode == 'atr':
        if direction == 'bullish':
            return entry_price + atr_val * params.atr_tp_mult
        else:
            return entry_price - atr_val * params.atr_tp_mult

    raise ValueError(f"Неизвестный tp_mode: {params.tp_mode!r}")


# ==============================================================================
# СИМУЛЯЦИЯ СДЕЛКИ
# ==============================================================================

def _simulate_trade(df: pd.DataFrame,
                    entry_idx: int,
                    entry_price: float,
                    sl_price: float,
                    tp_price: float,
                    direction: str,
                    params: StrategyParams) -> Tuple[float, int, str]:
    """
    Симулирует сделку свеча за свечой.

    Поддерживает:
    • Базовый выход по TP / SL / таймауту
    • Трейлинг-стоп (trailing_stop=True)
    • Частичный тейк на 1R (partial_take=True):
        — на 1R закрывается 50%, SL переносится в безубыток
        — остаток идёт к полному TP (или таймауту)

    Возвращает (exit_price, exit_idx, exit_reason).
    exit_reason: 'tp' | 'tp_partial' | 'sl' | 'timeout'
    """
    max_candles   = params.max_candles
    simultaneous  = params.simultaneous_hit
    trailing      = params.trailing_stop
    partial       = params.partial_take

    sl_dist = abs(entry_price - sl_price)
    current_sl = sl_price

    # Уровень частичного тейка (1R от входа)
    if direction == 'bullish':
        partial_tp_price = entry_price + sl_dist          # 1R
    else:
        partial_tp_price = entry_price - sl_dist          # 1R

    partial_hit = False   # флаг: первая половина уже закрыта

    best_price = entry_price  # для трейлинга

    end = min(entry_idx + 1 + max_candles, len(df))

    for j in range(entry_idx + 1, end):
        c = df.iloc[j]
        h = float(c['high'])
        l = float(c['low'])

        # ── Трейлинг-стоп ──────────────────────────────────────────────────
        if trailing and sl_dist > 0:
            step = sl_dist * params.trailing_step_r
            if direction == 'bullish':
                if h > best_price + step:
                    best_price = h
                    # подтягиваем SL, но не ниже исходного
                    new_sl = h - sl_dist
                    current_sl = max(current_sl, new_sl)
            else:
                if l < best_price - step:
                    best_price = l
                    new_sl = l + sl_dist
                    current_sl = min(current_sl, new_sl)

        # ── Частичный тейк на 1R ──────────────────────────────────────────
        if partial and not partial_hit:
            if direction == 'bullish' and h >= partial_tp_price:
                partial_hit = True
                current_sl = entry_price   # SL → безубыток
            elif direction == 'bearish' and l <= partial_tp_price:
                partial_hit = True
                current_sl = entry_price

        # ── Проверка основных уровней ─────────────────────────────────────
        if direction == 'bullish':
            tp_hit = h >= tp_price
            sl_hit = l <= current_sl
        else:
            tp_hit = l <= tp_price
            sl_hit = h >= current_sl

        if tp_hit and sl_hit:
            if simultaneous == 'sl_first':
                reason = 'sl'
                price  = current_sl
            else:
                reason = 'tp_partial' if partial_hit else 'tp'
                price  = tp_price
            return price, j, reason

        if tp_hit:
            return tp_price, j, 'tp_partial' if partial_hit else 'tp'

        if sl_hit:
            # При partial_hit 50% уже закрыто с прибылью; итоговый PnL будет средним
            return current_sl, j, 'sl'

    last_j = min(entry_idx + max_candles, len(df) - 1)
    return float(df.iloc[last_j]['close']), last_j, 'timeout'


def _calc_pnl(entry: float, exit_: float, direction: str,
              partial_hit: bool = False, partial_tp_price: float = 0.0) -> float:
    """
    Считает итоговый PnL в %.

    При partial_take первые 50% закрываются по partial_tp_price,
    вторые 50% — по exit_.
    Это лучше отражает реальную доходность стратегии с частичным выходом.
    """
    if partial_hit and partial_tp_price > 0:
        if direction == 'bullish':
            pnl1 = (partial_tp_price - entry) / entry * 100.0
            pnl2 = (exit_ - entry) / entry * 100.0
        else:
            pnl1 = (entry - partial_tp_price) / entry * 100.0
            pnl2 = (entry - exit_) / entry * 100.0
        return 0.5 * pnl1 + 0.5 * pnl2
    else:
        if direction == 'bullish':
            return (exit_ - entry) / entry * 100.0
        else:
            return (entry - exit_) / entry * 100.0


# ==============================================================================
# МЕТРИКИ ПОРТФЕЛЯ
# ==============================================================================

def _equity_curve(trades: List[Trade]) -> Tuple[List, List[float]]:
    if not trades:
        return [], []
    sorted_trades = sorted(trades, key=lambda t: (t.exit_time, t.exit_idx))
    times  = [t.exit_time for t in sorted_trades]
    cumsum = list(np.cumsum([t.pnl_pct for t in sorted_trades]))
    return times, cumsum


def _drawdown_series(cumulative: List[float]) -> List[float]:
    if not cumulative:
        return []
    arr  = np.array(cumulative)
    peak = np.maximum.accumulate(arr)
    return list(arr - peak)


def _max_drawdown(cumulative: List[float]) -> float:
    dd = _drawdown_series(cumulative)
    return float(min(dd)) if dd else 0.0


def _compute_summary(trades: List[Trade]) -> dict:
    if not trades:
        return {}

    pnls   = [t.pnl_pct   for t in trades]
    rs     = [t.r_multiple for t in trades]
    wins   = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    gross_win  = sum(t.pnl_pct for t in wins)          if wins   else 0.0
    gross_loss = abs(sum(t.pnl_pct for t in losses))   if losses else 1e-9
    pf = round(gross_win / gross_loss, 2) if gross_loss > 1e-9 else float('inf')

    _, cum = _equity_curve(trades)
    max_dd = _max_drawdown(cum)

    sharpe_r = round(float(np.mean(rs)) / float(np.std(rs)), 2) if len(rs) > 1 else 0.0

    avg_atr = round(float(np.mean([t.atr_at_entry for t in trades])), 5)

    return {
        'total_trades'    : len(trades),
        'tp_count'        : sum(1 for t in trades if t.exit_reason in ('tp', 'tp_partial')),
        'sl_count'        : sum(1 for t in trades if t.exit_reason == 'sl'),
        'timeout_count'   : sum(1 for t in trades if t.exit_reason == 'timeout'),
        'win_rate_pct'    : round(len(wins) / len(trades) * 100, 1),
        'avg_pnl_pct'     : round(float(np.mean(pnls)), 3),
        'avg_win_pct'     : round(float(np.mean([t.pnl_pct for t in wins])),   3) if wins   else 0.0,
        'avg_loss_pct'    : round(float(np.mean([t.pnl_pct for t in losses])), 3) if losses else 0.0,
        'avg_r_multiple'  : round(float(np.mean(rs)), 2),
        'median_r'        : round(float(np.median(rs)), 2),
        'profit_factor'   : pf,
        'total_pnl_pct'   : round(float(sum(pnls)), 3),
        'max_drawdown_pct': round(max_dd, 3),
        'sharpe_r'        : sharpe_r,
        'avg_rr_ratio'    : round(float(np.mean([t.rr_ratio for t in trades])), 2),
        'avg_atr_at_entry': avg_atr,
    }


# ==============================================================================
# ОСНОВНОЙ ДВИЖОК БЭКТЕСТА
# ==============================================================================

def run_strategy_backtest(ticker: str,
                          tf: str,
                          days: int = 30,
                          indicator: BaseIndicator = None,
                          params: StrategyParams = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Полный стратегический бэктест с SL/TP/таймаутом.

    Порядок фильтрации
    ──────────────────
    1. PatternDetector находит все паттерны на свече [i]
    2. use_indicator=True  → фильтр индикатором
    3. use_pattern_confirmation=True → подтверждение свечой [i+1]
    4. one_per_candle → дедупликация по направлению
    5. Вход на open[i+offset]
    6. min_rr → пропускаем сделки с плохим теоретическим R:R
    """
    indicator = indicator or NoIndicator()
    params    = params    or StrategyParams()

    tp_desc = {
        'fixed_rr': f"Фикс. {params.rr_multiplier}R",
        'ema10'   : "EMA10",
        'atr'     : f"ATR×{params.atr_tp_mult}",
    }.get(params.tp_mode, params.tp_mode)

    sl_desc = {
        'lookback': f"Lookback {params.sl_lookback} свч",
        'atr'     : f"ATR×{params.atr_sl_mult}",
    }.get(params.sl_mode, params.sl_mode)

    print(f"\n{'='*66}")
    print(f"Стратегия : {ticker} | {tf} | {days} дней")
    print(f"Индикатор : {indicator.plot_label or 'NoIndicator'}")
    print(f"SL        : {sl_desc}   |   TP: {tp_desc}")
    print(f"Таймаут   : {params.max_candles} свечей | simultaneous: {params.simultaneous_hit}")
    print(f"Трейлинг  : {'да (шаг=' + str(params.trailing_step_r) + 'R)' if params.trailing_stop else 'нет'}")
    print(f"Частич.TP : {'да (50% на 1R, SL→BE)' if params.partial_take else 'нет'}")
    print(f"Min R:R   : {params.min_rr} | Подтв. паттерна: {'да' if params.use_pattern_confirmation else 'нет'}")
    print('='*66)

    df_raw = _fetch_data(ticker, tf, days)
    if df_raw.empty:
        print("Нет данных.")
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare_df(df_raw, indicator)
    df = _ensure_atr(df, params.atr_period)          # добавляем ATR

    atr_col = f'atr_{params.atr_period}'

    raw_detector = PatternDetector(ScannerConfig(indicator=NoIndicator()))
    ind_detector = PatternDetector(ScannerConfig(indicator=indicator))

    offset    = 2 if params.use_pattern_confirmation else 1
    start_idx = max(12, params.sl_lookback, params.atr_period)
    end_idx   = len(df) - offset - params.max_candles

    trades: List[Trade] = []

    for i in range(start_idx, end_idx):

        # ── Шаг 1: все паттерны ───────────────────────────────────────────
        all_patterns = raw_detector.get_pattern_at_index(df, i)
        if not all_patterns:
            continue

        # ── Шаг 2: фильтр индикатором ─────────────────────────────────────
        filtered = (set(ind_detector.get_pattern_at_index(df, i))
                    if params.use_indicator else set(all_patterns))
        if not filtered:
            continue

        # ── Шаг 3: подтверждение ──────────────────────────────────────────
        if params.use_pattern_confirmation:
            if i + 1 >= len(df):
                continue
            final = list(filter_confirmed(list(filtered), df, i))
        else:
            final = list(filtered)

        if not final:
            continue

        # ── one_per_candle ─────────────────────────────────────────────────
        if params.one_per_candle:
            seen_dirs: set = set()
            deduped: List[str] = []
            for p in final:
                d = 'bullish' if _is_bullish(p) else 'bearish'
                if d not in seen_dirs:
                    seen_dirs.add(d)
                    deduped.append(p)
            final = deduped

        # ── Общие данные входной свечи ────────────────────────────────────
        entry_idx    = i + offset
        if entry_idx >= len(df):
            continue

        entry_candle = df.iloc[entry_idx]
        entry_price  = float(entry_candle['open'])
        atr_val      = float(df.iloc[i][atr_col])     # ATR на свече паттерна
        ema10        = float(entry_candle.get('ema10', entry_price))

        # ── Обрабатываем каждый паттерн ───────────────────────────────────
        for pattern in final:
            direction = 'bullish' if _is_bullish(pattern) else 'bearish'

            # ── SL ────────────────────────────────────────────────────────
            sl_price = _compute_sl(df, i, params, direction, entry_price)

            # Проверка корректности SL
            if direction == 'bullish' and sl_price >= entry_price:
                continue
            if direction == 'bearish' and sl_price <= entry_price:
                continue

            # Фильтр по размеру риска
            sl_risk_pct = abs(entry_price - sl_price) / entry_price * 100.0
            if sl_risk_pct < params.min_risk_pct or sl_risk_pct > params.max_risk_pct:
                continue

            # ── TP ────────────────────────────────────────────────────────
            # При tp_mode='ema10' передаём ema10 вместо atr_val
            tp_source = ema10 if params.tp_mode == 'ema10' else atr_val
            tp_price  = _compute_tp(entry_price, sl_price, direction, params, tp_source)

            # Проверяем направление TP
            if direction == 'bullish' and tp_price <= entry_price:
                continue
            if direction == 'bearish' and tp_price >= entry_price:
                continue

            # ── Фильтр min_rr ─────────────────────────────────────────────
            tp_dist_pct = abs(tp_price - entry_price) / entry_price * 100.0
            rr_ratio    = round(tp_dist_pct / sl_risk_pct, 2) if sl_risk_pct > 0 else 0.0

            if rr_ratio < params.min_rr:
                continue

            # ── Симуляция ──────────────────────────────────────────────────
            exit_price, exit_idx, exit_reason = _simulate_trade(
                df, entry_idx, entry_price, sl_price, tp_price,
                direction, params
            )

            # При partial_take корректируем PnL: 50% по 1R-цене, 50% по exit
            partial_hit = params.partial_take
            sl_dist     = abs(entry_price - sl_price)
            partial_tp  = (entry_price + sl_dist if direction == 'bullish'
                           else entry_price - sl_dist)

            pnl_pct   = round(_calc_pnl(entry_price, exit_price, direction,
                                         partial_hit=partial_hit,
                                         partial_tp_price=partial_tp), 3)
            r_multiple = round(pnl_pct / sl_risk_pct, 2) if sl_risk_pct > 0 else 0.0

            trades.append(Trade(
                pattern       = pattern,
                direction     = direction,
                pattern_idx   = i,
                entry_idx     = entry_idx,
                exit_idx      = exit_idx,
                entry_time    = df.iloc[entry_idx]['datetime'],
                exit_time     = df.iloc[exit_idx]['datetime'],
                entry_price   = round(entry_price, 4),
                sl_price      = round(sl_price, 4),
                tp_price      = round(tp_price, 4),
                sl_risk_pct   = round(sl_risk_pct, 3),
                tp_dist_pct   = round(tp_dist_pct, 3),
                rr_ratio      = rr_ratio,
                exit_price    = round(exit_price, 4),
                exit_reason   = exit_reason,
                pnl_pct       = pnl_pct,
                r_multiple    = r_multiple,
                atr_at_entry  = round(atr_val, 5),
            ))

    if not trades:
        print("Сделки не найдены. Попробуйте увеличить days или изменить параметры.")
        return pd.DataFrame(), pd.DataFrame()

    trades_df  = pd.DataFrame([asdict(t) for t in trades])
    summary    = _compute_summary(trades)
    summary_df = pd.DataFrame([summary])

    print(f"\n📊 СТРАТЕГИЯ — {len(trades)} сделок:")
    print(f"  Win rate     : {summary['win_rate_pct']}%  "
          f"(TP: {summary['tp_count']} | SL: {summary['sl_count']} | TO: {summary['timeout_count']})")
    print(f"  Avg R        : {summary['avg_r_multiple']}  |  Median R: {summary['median_r']}")
    print(f"  Profit factor: {summary['profit_factor']}")
    print(f"  Total PnL    : {summary['total_pnl_pct']}%")
    print(f"  Max drawdown : {summary['max_drawdown_pct']}%")
    print(f"  Sharpe (R)   : {summary['sharpe_r']}")
    print(f"  Avg R:R      : {summary['avg_rr_ratio']}")
    print(f"  Avg ATR entry: {summary['avg_atr_at_entry']}")

    return trades_df, summary_df


# ==============================================================================
# ВИЗУАЛИЗАЦИЯ
# ==============================================================================

_COLOR_TP      = '#00e676'
_COLOR_SL      = '#ff1744'
_COLOR_TIMEOUT = '#90a4ae'
_COLOR_BULL    = '#26a69a'
_COLOR_BEAR    = '#ef5350'


def _outcome_color(exit_reason: str) -> str:
    if exit_reason in ('tp', 'tp_partial'):
        return _COLOR_TP
    return {'sl': _COLOR_SL, 'timeout': _COLOR_TIMEOUT}.get(exit_reason, _COLOR_TIMEOUT)


def visual_strategy_backtest(ticker: str = 'SBER',
                             tf: str = '15min',
                             days: int = 30,
                             indicator: BaseIndicator = None,
                             params: StrategyParams = None,
                             show_trade_lines: bool = True,
                             max_trades_on_chart: int = 40,
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Полный визуальный стратегический бэктест.

    Панели
    ──────
    1. Свечной график: EMA10 (если есть), маркеры входа/выхода, линии SL/TP
    2. [опц.] Индикаторная панель (1 или 2 при DualConfirmIndicator)
    3. Equity curve
    4. Drawdown

    show_trade_lines = False — убирает линии SL/TP (чище при большом кол-ве сделок)
    max_trades_on_chart — ограничивает кол-во отрисованных сделок (последние N)
    """
    indicator = indicator or NoIndicator()
    params    = params    or StrategyParams()

    trades_df, summary_df = run_strategy_backtest(ticker, tf, days, indicator, params)
    if trades_df.empty:
        return trades_df, summary_df

    df_raw = _fetch_data(ticker, tf, days)
    df     = _prepare_df(df_raw, indicator)
    df     = _ensure_atr(df, params.atr_period)

    trades = [Trade(**row) for row in trades_df.to_dict('records')]

    is_dual  = isinstance(indicator, DualConfirmIndicator)
    is_plain = not isinstance(indicator, NoIndicator)

    if is_dual:
        n_rows  = 5
        heights = [0.40, 0.12, 0.12, 0.20, 0.16]
        titles  = [f"{ticker} | {tf}",
                   indicator.ind1.plot_label, indicator.ind2.plot_label,
                   "Equity curve (PnL %)", "Drawdown (%)"]
    elif is_plain:
        n_rows  = 4
        heights = [0.44, 0.15, 0.24, 0.17]
        titles  = [f"{ticker} | {tf}", indicator.plot_label,
                   "Equity curve (PnL %)", "Drawdown (%)"]
    else:
        n_rows  = 3
        heights = [0.55, 0.27, 0.18]
        titles  = [f"{ticker} | {tf}", "Equity curve (PnL %)", "Drawdown (%)"]

    eq_row = n_rows - 1
    dd_row = n_rows

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        row_heights=heights,
        subplot_titles=titles,
    )

    # ─── Свечи ────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'],    close=df['close'],
        name=ticker,
        increasing_line_color=_COLOR_BULL,
        decreasing_line_color=_COLOR_BEAR,
    ), row=1, col=1)

    # EMA10 (если присутствует в df)
    if 'ema10' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['ema10'],
            line=dict(color='orange', width=1.2),
            name='EMA10', opacity=0.7,
        ), row=1, col=1)

    # ─── Маркеры входа / выхода ───────────────────────────────────────────
    trades_to_show = sorted(trades, key=lambda t: t.entry_idx)[-max_trades_on_chart:]

    entry_x_bull, entry_y_bull = [], []
    entry_x_bear, entry_y_bear = [], []
    exit_groups: Dict[str, Tuple[list, list]] = {
        'tp': ([], []), 'tp_partial': ([], []),
        'sl': ([], []), 'timeout': ([], []),
    }

    for t in trades_to_show:
        if t.entry_idx < len(df):
            entry_dt = df.iloc[t.entry_idx]['datetime']
            if t.direction == 'bullish':
                entry_x_bull.append(entry_dt)
                entry_y_bull.append(t.entry_price)
            else:
                entry_x_bear.append(entry_dt)
                entry_y_bear.append(t.entry_price)

        if t.exit_idx < len(df):
            exit_dt = df.iloc[t.exit_idx]['datetime']
            reason  = t.exit_reason if t.exit_reason in exit_groups else 'timeout'
            exit_groups[reason][0].append(exit_dt)
            exit_groups[reason][1].append(t.exit_price)

    if entry_x_bull:
        fig.add_trace(go.Scatter(
            x=entry_x_bull, y=entry_y_bull, mode='markers',
            marker=dict(symbol='triangle-up', size=10,
                        color=_COLOR_BULL, line=dict(color='white', width=1)),
            name='Вход (бычий)',
        ), row=1, col=1)

    if entry_x_bear:
        fig.add_trace(go.Scatter(
            x=entry_x_bear, y=entry_y_bear, mode='markers',
            marker=dict(symbol='triangle-down', size=10,
                        color=_COLOR_BEAR, line=dict(color='white', width=1)),
            name='Вход (медвежий)',
        ), row=1, col=1)

    exit_labels = {
        'tp'       : 'TP ✓',
        'tp_partial': 'TP½ ✓',
        'sl'       : 'SL ✗',
        'timeout'  : 'Таймаут',
    }
    exit_colors = {
        'tp'        : _COLOR_TP,
        'tp_partial': '#69f0ae',   # светло-зелёный — частичный тейк
        'sl'        : _COLOR_SL,
        'timeout'   : _COLOR_TIMEOUT,
    }
    for reason, (xs, ys) in exit_groups.items():
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(symbol='circle', size=8,
                            color=exit_colors[reason],
                            line=dict(color='white', width=1)),
                name=exit_labels[reason],
            ), row=1, col=1)

    # ─── Линии SL / TP ────────────────────────────────────────────────────
    if show_trade_lines:
        sl_x, sl_y, tp_x, tp_y = [], [], [], []
        for t in trades_to_show:
            if t.entry_idx >= len(df) or t.exit_idx >= len(df):
                continue
            entry_dt = df.iloc[t.entry_idx]['datetime']
            exit_dt  = df.iloc[t.exit_idx]['datetime']
            sl_x += [entry_dt, exit_dt, None]
            sl_y += [t.sl_price, t.sl_price, None]
            tp_x += [entry_dt, exit_dt, None]
            tp_y += [t.tp_price, t.tp_price, None]

        if sl_x:
            fig.add_trace(go.Scatter(
                x=sl_x, y=sl_y, mode='lines',
                line=dict(color=_COLOR_SL, width=1, dash='dot'),
                name='SL уровень', opacity=0.45,
            ), row=1, col=1)
        if tp_x:
            fig.add_trace(go.Scatter(
                x=tp_x, y=tp_y, mode='lines',
                line=dict(color=_COLOR_TP, width=1, dash='dot'),
                name='TP уровень', opacity=0.45,
            ), row=1, col=1)

    # ─── Панели индикаторов ───────────────────────────────────────────────
    def _add_ind_panel(ind: BaseIndicator, row: int):
        col_name = ind.column_name
        if col_name not in df.columns:
            return
        color = '#29b6f6' if row == 2 else '#ce93d8'
        if 'macd' in col_name:
            colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df[col_name].fillna(0)]
            fig.add_trace(go.Bar(x=df['datetime'], y=df[col_name],
                                 marker_color=colors, name=ind.plot_label,
                                 showlegend=False), row=row, col=1)
        else:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[col_name],
                                     line=dict(color=color, width=1.5),
                                     name=ind.plot_label, showlegend=False), row=row, col=1)
        for lv in ind.get_level_lines():
            fig.add_hline(y=lv['value'],
                          line=dict(color=lv['color'], dash=lv['dash'], width=1),
                          row=row, col=1)

    if is_dual:
        _add_ind_panel(indicator.ind1, row=2)
        _add_ind_panel(indicator.ind2, row=3)
    elif is_plain:
        _add_ind_panel(indicator, row=2)

    # ─── Equity curve ─────────────────────────────────────────────────────
    _, cum = _equity_curve(trades)
    trade_nums = list(range(1, len(cum) + 1))
    eq_colors  = [_COLOR_TP if v >= 0 else _COLOR_SL for v in cum]

    fig.add_trace(go.Scatter(
        x=trade_nums, y=cum,
        mode='lines+markers',
        line=dict(color='#42a5f5', width=2),
        marker=dict(size=5, color=eq_colors),
        name='Equity', fill='tozeroy',
        fillcolor='rgba(66,165,245,0.12)',
    ), row=eq_row, col=1)
    fig.add_hline(y=0, line=dict(color='gray', dash='dot', width=1), row=eq_row, col=1)

    if cum:
        fig.add_annotation(
            x=len(cum), y=cum[-1],
            text=f"<b>{cum[-1]:+.2f}%</b>",
            showarrow=False,
            font=dict(color='white', size=11),
            bgcolor='rgba(66,165,245,0.6)',
            row=eq_row, col=1
        )

    # ─── Drawdown ─────────────────────────────────────────────────────────
    dd = _drawdown_series(cum)
    fig.add_trace(go.Scatter(
        x=trade_nums, y=dd,
        mode='lines',
        line=dict(color=_COLOR_SL, width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255,23,68,0.18)',
        name='Drawdown',
    ), row=dd_row, col=1)
    fig.add_hline(y=0, line=dict(color='gray', dash='dot', width=1), row=dd_row, col=1)

    if dd:
        min_dd  = min(dd)
        min_idx = dd.index(min_dd) + 1
        fig.add_annotation(
            x=min_idx, y=min_dd,
            text=f"<b>Max DD: {min_dd:.2f}%</b>",
            showarrow=True, arrowhead=2, arrowcolor=_COLOR_SL,
            font=dict(color='white', size=10),
            bgcolor='rgba(255,23,68,0.6)',
            row=dd_row, col=1
        )

    # ─── Заголовок ────────────────────────────────────────────────────────
    summary  = _compute_summary(trades)
    tp_label = {
        'fixed_rr': f"TP: Фикс.{params.rr_multiplier}R",
        'ema10'   : "TP: EMA10",
        'atr'     : f"TP: ATR×{params.atr_tp_mult}",
    }.get(params.tp_mode, params.tp_mode)
    sl_label = {
        'lookback': f"SL: LB{params.sl_lookback}",
        'atr'     : f"SL: ATR×{params.atr_sl_mult}",
    }.get(params.sl_mode, params.sl_mode)

    subtitle = (
        f"Сделок: {summary['total_trades']} | "
        f"WR: {summary['win_rate_pct']}% | "
        f"Avg R: {summary['avg_r_multiple']} | "
        f"PF: {summary['profit_factor']} | "
        f"PnL: {summary['total_pnl_pct']:+.2f}% | "
        f"MaxDD: {summary['max_drawdown_pct']:.2f}%"
    )

    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        title=dict(
            text=(f"Стратегия: {ticker} | {tf} | {indicator.plot_label or 'Без индикатора'}"
                  f" | {sl_label} | {tp_label} | TO: {params.max_candles} свч"
                  f"<br><sup>{subtitle}</sup>"),
            font=dict(size=13),
        ),
        height=900 + (200 if is_dual else 100 if is_plain else 0),
        showlegend=True,
        legend=dict(orientation='h', y=-0.02, x=0),
        margin=dict(l=50, r=50, t=90, b=60),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#1e1e1e')
    for ax in ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4', 'xaxis5']:
        fig.update_layout(**{ax: dict(rangeslider_visible=False)})

    fig.show()
    return trades_df, summary_df


# ==============================================================================
# СРАВНЕНИЕ КОНФИГУРАЦИЙ
# ==============================================================================

def compare_strategies(ticker: str,
                       tf: str,
                       days: int,
                       configs: Dict[str, Tuple[BaseIndicator, StrategyParams]]
                       ) -> pd.DataFrame:
    """
    Запускает run_strategy_backtest для нескольких конфигураций и выводит
    сводную таблицу.

    Пример:
        compare_strategies('SBER', '15min', days=30, configs={
            'Fixed 1.5R': (NoIndicator(),   StrategyParams(tp_mode='fixed_rr', rr_multiplier=1.5)),
            'Fixed 2.0R': (RSIIndicator(),  StrategyParams(tp_mode='fixed_rr', rr_multiplier=2.0)),
            'ATR TP'    : (AdaptiveMFIIndicator(),
                           StrategyParams(tp_mode='atr', atr_tp_mult=3.0)),
        })
    """
    rows = []
    for label, (ind, prm) in configs.items():
        _, summary_df = run_strategy_backtest(ticker, tf, days, ind, prm)
        if summary_df.empty:
            continue
        row = summary_df.iloc[0].to_dict()
        row['config'] = label
        rows.append(row)

    if not rows:
        print("Ни одна конфигурация не дала сделок.")
        return pd.DataFrame()

    cols_order = [
        'config', 'total_trades', 'win_rate_pct', 'avg_r_multiple', 'median_r',
        'profit_factor', 'total_pnl_pct', 'max_drawdown_pct', 'sharpe_r',
        'avg_rr_ratio', 'tp_count', 'sl_count', 'timeout_count', 'avg_atr_at_entry',
    ]
    df = pd.DataFrame(rows)
    existing_cols = [c for c in cols_order if c in df.columns]
    df = df[existing_cols]

    print("\n📊 СРАВНЕНИЕ СТРАТЕГИЙ:")
    print(df.to_string(index=False))
    return df