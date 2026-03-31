"""
strategy_test.py — полный стратегический бэктест для MorrisBot.

Логика сделки
─────────────
  Вход   : открытие свечи [pattern_idx + 2]
             паттерн на [i] → подтверждение на [i+1] → вход на open[i+2]
  Стоп   : min(low) / max(high) последних sl_lookback свечей
  Тейк   : EMA10 в момент входа (цена возвращается к скользящей средней)
  Выход  : первое касание TP или SL по high/low свечи; таймаут = close последней свечи окна
  Позиции: каждый сигнал открывает независимую позицию (несколько одновременно)

Метрики
───────
  % от входа, R-multiple, equity curve (кумулятивный PnL по времени закрытия),
  max drawdown, profit factor, sharpe (по R)

Использование
─────────────
    from morris_bot.strategy_test import StrategyParams, visual_strategy_backtest

    params = StrategyParams(sl_lookback=5, max_candles=20)
    visual_strategy_backtest('SBER', '15min', days=30, params=params)
"""

import warnings
from dataclasses import dataclass, asdict
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

    sl_lookback             : кол-во свечей назад для локального экстремума SL
    max_candles             : максимум свечей в позиции (таймаут)
    simultaneous_hit        : 'sl_first' | 'tp_first' — что считать исполненным
                              если SL и TP пробиты в одной свече
    use_indicator           : True — применять фильтр индикатора
    use_pattern_confirmation: True — подтверждение следующей свечой (рекомендуется)
    one_per_candle          : True — не более одной позиции одного направления
                              на одну свечу (исключает дублирующие сигналы)
    min_risk_pct            : минимальный риск в % от входа; меньше — пропускаем
    max_risk_pct            : максимальный риск в % от входа; больше — пропускаем
                              (защита от аномально широкого SL)
    """
    sl_lookback: int              = 5
    max_candles: int              = 20
    simultaneous_hit: str         = 'sl_first'
    use_indicator: bool           = True
    use_pattern_confirmation: bool = True
    one_per_candle: bool          = True
    min_risk_pct: float           = 0.05
    max_risk_pct: float           = 5.0


# ==============================================================================
# РЕЗУЛЬТАТ СДЕЛКИ
# ==============================================================================

@dataclass
class Trade:
    """Результат одной сделки."""
    pattern     : str
    direction   : str           # 'bullish' | 'bearish'
    pattern_idx : int
    entry_idx   : int
    exit_idx    : int
    entry_time  : object        # pd.Timestamp
    exit_time   : object        # pd.Timestamp
    entry_price : float
    sl_price    : float
    tp_price    : float
    sl_risk_pct : float         # abs(entry - sl) / entry * 100
    tp_dist_pct : float         # abs(tp - entry) / entry * 100
    rr_ratio    : float         # tp_dist_pct / sl_risk_pct (теоретическое R:R)
    exit_price  : float
    exit_reason : str           # 'tp' | 'sl' | 'timeout'
    pnl_pct     : float         # прибыль/убыток в %
    r_multiple  : float         # pnl_pct / sl_risk_pct (реализованный R)


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ СИМУЛЯЦИИ
# ==============================================================================

def _compute_sl(df: pd.DataFrame, pattern_idx: int,
                sl_lookback: int, direction: str) -> float:
    """
    Локальный экстремум последних sl_lookback свечей (включая свечу паттерна).
    Бычий паттерн → min(low), медвежий → max(high).
    """
    start  = max(0, pattern_idx - sl_lookback + 1)
    window = df.iloc[start : pattern_idx + 1]
    if direction == 'bullish':
        return float(window['low'].min())
    else:
        return float(window['high'].max())


def _simulate_trade(df: pd.DataFrame,
                    entry_idx: int,
                    entry_price: float,
                    sl_price: float,
                    tp_price: float,
                    direction: str,
                    max_candles: int,
                    simultaneous_hit: str) -> Tuple[float, int, str]:
    """
    Симулирует сделку свеча за свечой по high/low (касание = исполнение).

    Возвращает (exit_price, exit_idx, exit_reason).
    exit_reason: 'tp' | 'sl' | 'timeout'
    """
    end = min(entry_idx + 1 + max_candles, len(df))

    for j in range(entry_idx + 1, end):
        c = df.iloc[j]
        h = float(c['high'])
        l = float(c['low'])

        if direction == 'bullish':
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
        else:
            tp_hit = l <= tp_price
            sl_hit = h >= sl_price

        if tp_hit and sl_hit:
            # Оба уровня в одной свече — применяем настройку simultaneous_hit
            if simultaneous_hit == 'sl_first':
                return sl_price, j, 'sl'
            else:
                return tp_price, j, 'tp'

        if tp_hit:
            return tp_price, j, 'tp'
        if sl_hit:
            return sl_price, j, 'sl'

    # Таймаут: закрываемся по close последней свечи окна
    last_j = min(entry_idx + max_candles, len(df) - 1)
    return float(df.iloc[last_j]['close']), last_j, 'timeout'


def _calc_pnl(entry: float, exit_: float, direction: str) -> float:
    if direction == 'bullish':
        return (exit_ - entry) / entry * 100.0
    else:
        return (entry - exit_) / entry * 100.0


# ==============================================================================
# МЕТРИКИ ПОРТФЕЛЯ
# ==============================================================================

def _equity_curve(trades: List[Trade]) -> Tuple[List, List[float]]:
    """
    Кумулятивный PnL, отсортированный по времени закрытия сделки.
    Поскольку позиции могут быть открыты одновременно, используем
    накопление по времени выхода — стандартная практика для паттерновых стратегий.
    """
    if not trades:
        return [], []
    sorted_trades = sorted(trades, key=lambda t: (t.exit_time, t.exit_idx))
    times  = [t.exit_time for t in sorted_trades]
    cumsum = list(np.cumsum([t.pnl_pct for t in sorted_trades]))
    return times, cumsum


def _drawdown_series(cumulative: List[float]) -> List[float]:
    """Просадка в каждой точке equity curve (всегда ≤ 0)."""
    if not cumulative:
        return []
    arr  = np.array(cumulative)
    peak = np.maximum.accumulate(arr)
    return list(arr - peak)


def _max_drawdown(cumulative: List[float]) -> float:
    dd = _drawdown_series(cumulative)
    return float(min(dd)) if dd else 0.0


def _compute_summary(trades: List[Trade]) -> dict:
    """Агрегированная статистика по списку сделок."""
    if not trades:
        return {}

    pnls   = [t.pnl_pct    for t in trades]
    rs     = [t.r_multiple  for t in trades]
    wins   = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    gross_win  = sum(t.pnl_pct for t in wins)   if wins   else 0.0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 1e-9
    pf = round(gross_win / gross_loss, 2) if gross_loss > 1e-9 else float('inf')

    _, cum   = _equity_curve(trades)
    max_dd   = _max_drawdown(cum)

    # Sharpe по R-multiple (более стабилен чем по %-PnL при разных размерах позиций)
    sharpe_r = round(float(np.mean(rs)) / float(np.std(rs)), 2) if len(rs) > 1 else 0.0

    return {
        'total_trades'    : len(trades),
        'tp_count'        : sum(1 for t in trades if t.exit_reason == 'tp'),
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

    Порядок фильтрации сигналов
    ───────────────────────────
    1. Детектор находит все паттерны на свече [i] (без фильтров)
    2. Если use_indicator=True  → оставляем только подтверждённые индикатором
    3. Если use_pattern_confirmation=True → оставляем только те, у которых
       свеча [i+1] подтверждает (Harami, Hammer и др.)
    4. Вход на open[i+2]

    Управление позицией
    ───────────────────
    • SL  = min/max last sl_lookback свечей (по low/high)
    • TP  = EMA10 на свече входа
    • Каждый прошедший фильтры паттерн → независимая позиция

    Возвращает
    ──────────
    trades_df  : строка на каждую сделку
    summary_df : агрегированная статистика
    """
    indicator = indicator or NoIndicator()
    params    = params    or StrategyParams()

    print(f"\n{'='*62}")
    print(f"Стратегия : {ticker} | {tf} | {days} дней")
    print(f"Индикатор : {indicator.plot_label or 'NoIndicator'}")
    print(f"SL        : last {params.sl_lookback} свечей | TP: EMA10")
    print(f"Таймаут   : {params.max_candles} свечей | simultaneous: {params.simultaneous_hit}")
    print(f"Подтверждение паттерна: {'да' if params.use_pattern_confirmation else 'нет'}")
    print(f"Фильтр индикатора     : {'да' if params.use_indicator else 'нет'}")
    print('='*62)

    df_raw = _fetch_data(ticker, tf, days)
    if df_raw.empty:
        print("Нет данных.")
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare_df(df_raw, indicator)

    # Два детектора: один без фильтра, другой с индикатором
    raw_detector = PatternDetector(ScannerConfig(indicator=NoIndicator()))
    ind_detector = PatternDetector(ScannerConfig(indicator=indicator))

    # Смещение: 2 свечи (подтверждение[i+1] + вход на open[i+2])
    # или 1 без подтверждения
    offset = 2 if params.use_pattern_confirmation else 1

    # Минимальный старт: 12 (контекст PatternDetector) + sl_lookback
    start_idx = max(12, params.sl_lookback)

    # Дальняя граница: оставляем место для offset + forward window
    end_idx = len(df) - offset - params.max_candles

    trades: List[Trade] = []

    for i in range(start_idx, end_idx):

        # ── Шаг 1: все паттерны без фильтра ──────────────────────────────
        all_patterns = raw_detector.get_pattern_at_index(df, i)
        if not all_patterns:
            continue

        # ── Шаг 2: фильтр индикатором ─────────────────────────────────────
        if params.use_indicator:
            filtered = set(ind_detector.get_pattern_at_index(df, i))
        else:
            filtered = set(all_patterns)

        if not filtered:
            continue

        # ── Шаг 3: подтверждение следующей свечой ─────────────────────────
        if params.use_pattern_confirmation:
            # Нужна закрытая свеча i+1
            if i + 1 >= len(df):
                continue
            final = list(filter_confirmed(list(filtered), df, i))
        else:
            final = list(filtered)

        if not final:
            continue

        # ── one_per_candle: не более одного сигнала каждого направления ───
        if params.one_per_candle:
            seen_dirs: set = set()
            deduped: List[str] = []
            for p in final:
                d = 'bullish' if _is_bullish(p) else 'bearish'
                if d not in seen_dirs:
                    seen_dirs.add(d)
                    deduped.append(p)
            final = deduped

        # ── Вход ──────────────────────────────────────────────────────────
        entry_idx = i + offset
        if entry_idx >= len(df):
            continue

        entry_candle = df.iloc[entry_idx]
        entry_price  = float(entry_candle['open'])
        tp_price     = float(entry_candle['ema10'])  # TP = EMA10

        for pattern in final:
            direction = 'bullish' if _is_bullish(pattern) else 'bearish'

            # ── Проверяем, что TP в правильном направлении ─────────────────
            if direction == 'bullish' and tp_price <= entry_price:
                # EMA10 ниже или равна входу — паттерн недалеко от EMA, TP некорректен
                continue
            if direction == 'bearish' and tp_price >= entry_price:
                continue

            # ── SL: локальный экстремум ────────────────────────────────────
            sl_price = _compute_sl(df, i, params.sl_lookback, direction)

            # Проверяем корректность SL
            if direction == 'bullish' and sl_price >= entry_price:
                continue   # SL выше входа — аномалия
            if direction == 'bearish' and sl_price <= entry_price:
                continue

            # ── Фильтр по размеру риска ────────────────────────────────────
            sl_risk_pct = abs(entry_price - sl_price) / entry_price * 100.0
            if sl_risk_pct < params.min_risk_pct:
                continue   # риск слишком мал — вероятно дубль или флэт
            if sl_risk_pct > params.max_risk_pct:
                continue   # риск слишком велик — выброс цены

            tp_dist_pct = abs(tp_price - entry_price) / entry_price * 100.0
            rr_ratio    = round(tp_dist_pct / sl_risk_pct, 2) if sl_risk_pct > 0 else 0.0

            # ── Симуляция ──────────────────────────────────────────────────
            exit_price, exit_idx, exit_reason = _simulate_trade(
                df, entry_idx, entry_price, sl_price, tp_price,
                direction, params.max_candles, params.simultaneous_hit
            )

            pnl_pct   = round(_calc_pnl(entry_price, exit_price, direction), 3)
            r_multiple = round(pnl_pct / sl_risk_pct, 2) if sl_risk_pct > 0 else 0.0

            trades.append(Trade(
                pattern      = pattern,
                direction    = direction,
                pattern_idx  = i,
                entry_idx    = entry_idx,
                exit_idx     = exit_idx,
                entry_time   = df.iloc[entry_idx]['datetime'],
                exit_time    = df.iloc[exit_idx]['datetime'],
                entry_price  = round(entry_price, 4),
                sl_price     = round(sl_price, 4),
                tp_price     = round(tp_price, 4),
                sl_risk_pct  = round(sl_risk_pct, 3),
                tp_dist_pct  = round(tp_dist_pct, 3),
                rr_ratio     = rr_ratio,
                exit_price   = round(exit_price, 4),
                exit_reason  = exit_reason,
                pnl_pct      = pnl_pct,
                r_multiple   = r_multiple,
            ))

    if not trades:
        print("Сделки не найдены. Попробуйте увеличить days или изменить параметры.")
        return pd.DataFrame(), pd.DataFrame()

    trades_df  = pd.DataFrame([asdict(t) for t in trades])
    summary    = _compute_summary(trades)
    summary_df = pd.DataFrame([summary])

    # ── Вывод ──────────────────────────────────────────────────────────────
    print(f"\n📊 СТРАТЕГИЯ — {len(trades)} сделок:")
    print(f"  Win rate     : {summary['win_rate_pct']}%  "
          f"(TP: {summary['tp_count']} | SL: {summary['sl_count']} | TO: {summary['timeout_count']})")
    print(f"  Avg R        : {summary['avg_r_multiple']}  |  Median R: {summary['median_r']}")
    print(f"  Profit factor: {summary['profit_factor']}")
    print(f"  Total PnL    : {summary['total_pnl_pct']}%")
    print(f"  Max drawdown : {summary['max_drawdown_pct']}%")
    print(f"  Sharpe (R)   : {summary['sharpe_r']}")
    print(f"  Avg R:R      : {summary['avg_rr_ratio']}")

    return trades_df, summary_df


# ==============================================================================
# ВИЗУАЛИЗАЦИЯ
# ==============================================================================

_COLOR_TP      = '#00e676'   # зелёный — TP
_COLOR_SL      = '#ff1744'   # красный — SL
_COLOR_TIMEOUT = '#90a4ae'   # серый   — таймаут
_COLOR_BULL    = '#26a69a'
_COLOR_BEAR    = '#ef5350'


def _outcome_color(exit_reason: str) -> str:
    return {
        'tp'     : _COLOR_TP,
        'sl'     : _COLOR_SL,
        'timeout': _COLOR_TIMEOUT,
    }.get(exit_reason, _COLOR_TIMEOUT)


def visual_strategy_backtest(ticker: str = 'SBER',
                             tf: str = '15min',
                             days: int = 30,
                             indicator: BaseIndicator = None,
                             params: StrategyParams = None,
                             show_trade_lines: bool = True,
                             max_trades_on_chart: int = 40
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Полный визуальный стратегический бэктест.

    Панели
    ──────
    1. Свечной график: EMA10, маркеры входа/выхода, линии SL/TP по каждой сделке
    2. Equity curve: кумулятивный PnL (% сумма по сделкам, отсортированным по выходу)
    3. Drawdown: просадка от текущего пика equity curve

    Маркеры на свечном графике
    ──────────────────────────
    △ / ▽   вход в позицию (бычий / медвежий)
    ●       выход: зелёный=TP, красный=SL, серый=таймаут
    ─ ─ ─   пунктирные линии SL (красный) и TP (зелёный) от входа до выхода

    show_trade_lines = False убирает линии SL/TP (чище при большом кол-ве сделок)
    max_trades_on_chart ограничивает кол-во отрисованных сделок (последние N)
    """
    indicator = indicator or NoIndicator()
    params    = params    or StrategyParams()

    trades_df, summary_df = run_strategy_backtest(ticker, tf, days, indicator, params)

    if trades_df.empty:
        return trades_df, summary_df

    df_raw = _fetch_data(ticker, tf, days)
    df     = _prepare_df(df_raw, indicator)

    trades = [Trade(**row) for row in trades_df.to_dict('records')]

    # ─── Layout ───────────────────────────────────────────────────────────────
    is_dual  = isinstance(indicator, DualConfirmIndicator)
    is_plain = not isinstance(indicator, NoIndicator)

    if is_dual:
        n_rows  = 5   # свечи + ind1 + ind2 + equity + drawdown
        heights = [0.40, 0.12, 0.12, 0.20, 0.16]
        titles  = [f"{ticker} | {tf}",
                   indicator.ind1.plot_label, indicator.ind2.plot_label,
                   "Equity curve (PnL %)", "Drawdown (%)"]
    elif is_plain:
        n_rows  = 4   # свечи + индикатор + equity + drawdown
        heights = [0.44, 0.15, 0.24, 0.17]
        titles  = [f"{ticker} | {tf}", indicator.plot_label,
                   "Equity curve (PnL %)", "Drawdown (%)"]
    else:
        n_rows  = 3   # свечи + equity + drawdown
        heights = [0.55, 0.27, 0.18]
        titles  = [f"{ticker} | {tf}", "Equity curve (PnL %)", "Drawdown (%)"]

    eq_row = n_rows - 1   # строка equity curve
    dd_row = n_rows        # строка drawdown

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=False,   # equity/dd по индексу сделки, не по дате
        vertical_spacing=0.04,
        row_heights=heights,
        subplot_titles=titles,
    )

    # ─── Панель 1: Свечи + EMA10 ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'],    close=df['close'],
        name=ticker,
        increasing_line_color=_COLOR_BULL,
        decreasing_line_color=_COLOR_BEAR,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['ema10'],
        line=dict(color='orange', width=1.5),
        name='EMA10'
    ), row=1, col=1)

    # ─── Торговые маркеры и линии ─────────────────────────────────────────
    # Ограничиваем кол-во сделок на графике (последние N по времени входа)
    trades_to_show = sorted(trades, key=lambda t: t.entry_idx)[-max_trades_on_chart:]

    # Группируем маркеры по типу исхода для минимизации traces
    entry_x_bull, entry_y_bull = [], []
    entry_x_bear, entry_y_bear = [], []
    exit_groups: Dict[str, Tuple[list, list]] = {
        'tp': ([], []), 'sl': ([], []), 'timeout': ([], [])
    }

    for t in trades_to_show:
        # Точки входа
        if t.entry_idx < len(df):
            entry_dt = df.iloc[t.entry_idx]['datetime']
            if t.direction == 'bullish':
                entry_x_bull.append(entry_dt)
                entry_y_bull.append(t.entry_price)
            else:
                entry_x_bear.append(entry_dt)
                entry_y_bear.append(t.entry_price)

        # Точки выхода
        if t.exit_idx < len(df):
            exit_dt = df.iloc[t.exit_idx]['datetime']
            exit_groups[t.exit_reason][0].append(exit_dt)
            exit_groups[t.exit_reason][1].append(t.exit_price)

    # Маркеры входа (треугольники)
    if entry_x_bull:
        fig.add_trace(go.Scatter(
            x=entry_x_bull, y=entry_y_bull,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10,
                        color=_COLOR_BULL, line=dict(color='white', width=1)),
            name='Вход (бычий)', showlegend=True,
        ), row=1, col=1)

    if entry_x_bear:
        fig.add_trace(go.Scatter(
            x=entry_x_bear, y=entry_y_bear,
            mode='markers',
            marker=dict(symbol='triangle-down', size=10,
                        color=_COLOR_BEAR, line=dict(color='white', width=1)),
            name='Вход (медвежий)', showlegend=True,
        ), row=1, col=1)

    # Маркеры выхода (круги)
    exit_labels = {'tp': 'TP ✓', 'sl': 'SL ✗', 'timeout': 'Таймаут'}
    exit_colors = {'tp': _COLOR_TP, 'sl': _COLOR_SL, 'timeout': _COLOR_TIMEOUT}
    for reason, (xs, ys) in exit_groups.items():
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(symbol='circle', size=8,
                            color=exit_colors[reason],
                            line=dict(color='white', width=1)),
                name=exit_labels[reason], showlegend=True,
            ), row=1, col=1)

    # Линии SL и TP (пунктир от входа до выхода)
    if show_trade_lines:
        sl_x, sl_y = [], []
        tp_x, tp_y = [], []

        for t in trades_to_show:
            if t.entry_idx >= len(df) or t.exit_idx >= len(df):
                continue
            entry_dt = df.iloc[t.entry_idx]['datetime']
            exit_dt  = df.iloc[t.exit_idx]['datetime']

            # SL линия
            sl_x += [entry_dt, exit_dt, None]
            sl_y += [t.sl_price, t.sl_price, None]

            # TP линия
            tp_x += [entry_dt, exit_dt, None]
            tp_y += [t.tp_price, t.tp_price, None]

        if sl_x:
            fig.add_trace(go.Scatter(
                x=sl_x, y=sl_y,
                mode='lines',
                line=dict(color=_COLOR_SL, width=1, dash='dot'),
                name='SL уровень', opacity=0.5, showlegend=True,
            ), row=1, col=1)

        if tp_x:
            fig.add_trace(go.Scatter(
                x=tp_x, y=tp_y,
                mode='lines',
                line=dict(color=_COLOR_TP, width=1, dash='dot'),
                name='TP уровень', opacity=0.5, showlegend=True,
            ), row=1, col=1)

    # ─── Панели индикаторов (если есть) ───────────────────────────────────
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

    # Цвет линии equity: зелёный выше нуля, красный ниже
    eq_colors = [_COLOR_TP if v >= 0 else _COLOR_SL for v in cum]

    fig.add_trace(go.Scatter(
        x=trade_nums, y=cum,
        mode='lines+markers',
        line=dict(color='#42a5f5', width=2),
        marker=dict(size=5, color=eq_colors),
        name='Equity', fill='tozeroy',
        fillcolor='rgba(66,165,245,0.12)',
    ), row=eq_row, col=1)

    fig.add_hline(y=0, line=dict(color='gray', dash='dot', width=1), row=eq_row, col=1)

    # Аннотация итогового PnL
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

    # Аннотация максимальной просадки
    if dd:
        min_dd = min(dd)
        min_idx = dd.index(min_dd) + 1
        fig.add_annotation(
            x=min_idx, y=min_dd,
            text=f"<b>Max DD: {min_dd:.2f}%</b>",
            showarrow=True, arrowhead=2, arrowcolor=_COLOR_SL,
            font=dict(color='white', size=10),
            bgcolor='rgba(255,23,68,0.6)',
            row=dd_row, col=1
        )

    # ─── Заголовок со статистикой ─────────────────────────────────────────
    summary = _compute_summary(trades)
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
                  f" | SL: {params.sl_lookback} свч | TO: {params.max_candles} свч"
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

    # Отключаем rangeslider на всех осях
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
    сводную таблицу для сравнения.

    configs = {
        'RSI':  (RSIIndicator(), StrategyParams()),
        'Dual': (DualConfirmIndicator(...), StrategyParams(sl_lookback=7)),
    }
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
        'avg_rr_ratio', 'tp_count', 'sl_count', 'timeout_count',
    ]
    df = pd.DataFrame(rows)
    existing_cols = [c for c in cols_order if c in df.columns]
    df = df[existing_cols]

    print("\n📊 СРАВНЕНИЕ СТРАТЕГИЙ:")
    print(df.to_string(index=False))
    return df
