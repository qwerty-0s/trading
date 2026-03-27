"""
backtest.py — модуль бэктеста для MorrisBot.

Импортирует из trading_test.py без каких-либо изменений в нём.
Добавляет три новых индикатора и статистический движок бэктеста.

Новые индикаторы
────────────────
BollingerPercentBIndicator  — Void Lines из TV_ALGO (%B + фильтр узких полос)
AdaptiveMFIIndicator        — K-MEAN.pine (MFI + K-means кластеризация)
DualConfirmIndicator        — обёртка над любыми двумя BaseIndicator;
                              подтверждает сигнал только если ОБА согласны;
                              прозрачна для PatternDetector через bit-encoding

Использование
─────────────
    from backtest import (
        BollingerPercentBIndicator, AdaptiveMFIIndicator,
        DualConfirmIndicator, visual_backtest_dual, run_backtest,
    )

    ind = DualConfirmIndicator(
        BollingerPercentBIndicator(),
        AdaptiveMFIIndicator(),
    )
    visual_backtest_dual('NGJ6', '15min', days=30, indicator=ind)
"""

from trading_test import (
    BaseIndicator, NoIndicator, RSIIndicator, MACDIndicator, StochasticIndicator,
    ScannerConfig, PatternDetector, ChartVisualizer, TelegramRouter, MorrisBot,
)

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from moexalgo import Ticker
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

warnings.filterwarnings("ignore")

# ─────────────────────── определяем направление паттерна ───────────────────────

_BULLISH_KEYWORDS = {'bull', 'hammer', 'morning', 'soldier', 'piercing'}

def _is_bullish(pattern: str) -> bool:
    p = pattern.lower()
    return any(k in p for k in _BULLISH_KEYWORDS)


# ==============================================================================
# НОВЫЕ ИНДИКАТОРЫ
# ==============================================================================

class BollingerPercentBIndicator(BaseIndicator):
    """
    Bollinger %B — порт Void Lines из TV_ALGO (HomelessLemon).

        basis = SMA(close, period)
        upper = basis + mult × σ        ← "Upper 200%" / upper3
        lower = basis − mult × σ        ← "Lower 200%" / lower3
        %B    = (close − lower) / (upper − lower)

    Диапазон: 0 = нижняя полоса, 1 = верхняя, >1 / <0 = выход за полосу.

    Фильтр min_bandwidth_pct
    ────────────────────────
    В боковике BB сжимаются: (upper − lower) / basis < threshold.
    Экстремум %B в таком режиме — шум, а не перекупленность.
    Для таких свечей compute() возвращает NaN → PatternDetector не блокирует
    сигнал (NaN трактуется как «нет данных»), но и не использует ложный экстремум.
    """

    def __init__(self,
                 period: int = 20,
                 mult: float = 2.0,
                 oversold_band: float = 0.2,
                 overbought_band: float = 0.8,
                 min_bandwidth_pct: float = 0.01):
        self.period = period
        self.mult = mult
        self.oversold_band = oversold_band
        self.overbought_band = overbought_band
        self.min_bandwidth_pct = min_bandwidth_pct

    @property
    def column_name(self) -> str:
        return f"bb_pctB_{self.period}"

    @property
    def plot_label(self) -> str:
        return f"Bollinger %B ({self.period}, {self.mult}σ)"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        basis = df['close'].rolling(self.period).mean()
        std   = df['close'].rolling(self.period).std()
        upper = basis + self.mult * std
        lower = basis - self.mult * std
        band  = upper - lower

        too_narrow = band < (basis * self.min_bandwidth_pct)
        pct_b = (df['close'] - lower) / band.replace(0, np.nan)
        pct_b[too_narrow] = np.nan
        return pct_b

    def confirms_bullish(self, value: float) -> bool:
        return value <= self.oversold_band

    def confirms_bearish(self, value: float) -> bool:
        return value >= self.overbought_band

    def get_level_lines(self) -> List[dict]:
        return [
            {"value": self.overbought_band, "color": "red",   "dash": "dash"},
            {"value": self.oversold_band,   "color": "green", "dash": "dash"},
            {"value": 0.5,                  "color": "gray",  "dash": "dot"},
        ]


class AdaptiveMFIIndicator(BaseIndicator):
    """
    Adaptive MFI с K-means кластеризацией — порт K-MEAN.pine (AlgoAlpha).

    Алгоритм
    ────────
    1. Вычисляется классический MFI(mfi_len) на hlc3 × volume.
    2. На скользящем окне training_size свечей запускается K-means (3 кластера):
       overbought (ob), neutral (ne), oversold (os).
       Начальные центроиды — фиксированные 80/50/20, затем сходятся за iterations шагов.
    3. Текущий MFI нормализуется: position = 100 × (mfi − os) / (ob − os).
       Диапазон 0..100 с adaptive уровнями (не фиксированными 80/20).

    Подтверждение:
      Бычий  ← position ≤ 20 (у адаптивного os)
      Медвежий ← position ≥ 80 (у адаптивного ob)
    """

    def __init__(self,
                 mfi_len: int = 14,
                 training_size: int = 300,
                 iterations: int = 5,
                 init_ob: float = 80.0,
                 init_ne: float = 50.0,
                 init_os: float = 20.0):
        self.mfi_len = mfi_len
        self.training_size = training_size
        self.iterations = iterations
        self.init_ob = init_ob
        self.init_ne = init_ne
        self.init_os = init_os

    @property
    def column_name(self) -> str:
        return "adaptive_mfi"

    @property
    def plot_label(self) -> str:
        return f"AI MFI K-Means({self.mfi_len})"

    def _raw_mfi(self, df: pd.DataFrame) -> pd.Series:
        tp       = (df['high'] + df['low'] + df['close']) / 3
        raw_mf   = tp * df['volume']
        pos_flow = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_flow = raw_mf.where(tp < tp.shift(1), 0.0)
        pos_sum  = pos_flow.rolling(self.mfi_len).sum()
        neg_sum  = neg_flow.rolling(self.mfi_len).sum()
        return 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mfi    = self._raw_mfi(df)
        n      = len(mfi)
        result = pd.Series(np.nan, index=df.index)

        for i in range(self.mfi_len + 1, n):
            window = mfi.iloc[max(0, i - self.training_size):i + 1].dropna().values
            if len(window) < 10:
                continue

            a, b, c = self.init_ob, self.init_ne, self.init_os
            for _ in range(self.iterations):
                ob_v, ne_v, os_v = [], [], []
                for v in window:
                    d_a, d_b, d_c = abs(v - a), abs(v - b), abs(v - c)
                    if d_b < d_a and d_b < d_c:
                        ne_v.append(v)
                    elif d_a < d_b and d_a < d_c:
                        ob_v.append(v)
                    else:
                        os_v.append(v)
                if ob_v: a = float(np.mean(ob_v))
                if ne_v: b = float(np.mean(ne_v))
                if os_v: c = float(np.mean(os_v))

            cur = mfi.iloc[i]
            if not np.isnan(cur) and (a - c) != 0:
                result.iloc[i] = 100.0 * (cur - c) / (a - c)

        return result

    def confirms_bullish(self, value: float) -> bool:
        return value <= 20.0

    def confirms_bearish(self, value: float) -> bool:
        return value >= 80.0

    def get_level_lines(self) -> List[dict]:
        return [
            {"value": 80, "color": "red",   "dash": "dash"},
            {"value": 20, "color": "green", "dash": "dash"},
            {"value": 50, "color": "gray",  "dash": "dot"},
        ]


class DualConfirmIndicator(BaseIndicator):
    """
    Composite-индикатор: подтверждает сигнал только если ОБА внутренних
    индикатора согласны. Прозрачен для PatternDetector — ведёт себя как один.

    Архитектурное решение: bit-encoding
    ────────────────────────────────────
    Интерфейс BaseIndicator предполагает один float из compute().
    DualConfirmIndicator кодирует состояние обоих инд. в битовую маску:

        bit 0 (1):  ind1 подтверждает бычий
        bit 1 (2):  ind2 подтверждает бычий
        bit 2 (4):  ind1 подтверждает медвежий
        bit 3 (8):  ind2 подтверждает медвежий

        confirms_bullish → bits 0 и 1 → value & 3  == 3
        confirms_bearish → bits 2 и 3 → value & 12 == 12

    Визуализация
    ────────────
    visual_backtest_dual определяет DualConfirmIndicator через isinstance() и
    рисует отдельные панели для ind1 и ind2 (не бессмысленный bit-encoded график).
    prepare_df при обнаружении DualConfirmIndicator также записывает в df
    колонки sub-индикаторов для этих панелей.

    Пример
    ──────
        ind = DualConfirmIndicator(
            BollingerPercentBIndicator(period=20, mult=2.0),
            AdaptiveMFIIndicator(mfi_len=14),
        )
        config = ScannerConfig(indicator=ind)   # без изменений в PatternDetector
    """

    def __init__(self, ind1: BaseIndicator, ind2: BaseIndicator):
        self.ind1 = ind1
        self.ind2 = ind2

    @property
    def column_name(self) -> str:
        return f"dual__{self.ind1.column_name}__{self.ind2.column_name}"

    @property
    def plot_label(self) -> str:
        return f"{self.ind1.plot_label} ∩ {self.ind2.plot_label}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        s1 = self.ind1.compute(df)
        s2 = self.ind2.compute(df)

        valid = s1.notna() & s2.notna()

        b0 = s1.map(lambda v: int(self.ind1.confirms_bullish(v)) if pd.notna(v) else 0)
        b1 = s2.map(lambda v: int(self.ind2.confirms_bullish(v)) if pd.notna(v) else 0)
        b2 = s1.map(lambda v: int(self.ind1.confirms_bearish(v)) if pd.notna(v) else 0)
        b3 = s2.map(lambda v: int(self.ind2.confirms_bearish(v)) if pd.notna(v) else 0)

        encoded = (b0 + b1 * 2 + b2 * 4 + b3 * 8).astype(float)
        encoded[~valid] = np.nan
        return encoded

    def confirms_bullish(self, value: float) -> bool:
        if np.isnan(value):
            return True   # нет данных → не блокируем
        return (int(value) & 3) == 3

    def confirms_bearish(self, value: float) -> bool:
        if np.isnan(value):
            return True
        return (int(value) & 12) == 12

    def get_level_lines(self) -> List[dict]:
        return []   # визуализируем sub-панели, не bit-серию


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def fetch_data(ticker: str, tf: str, days: int = 30) -> pd.DataFrame:
    """
    Загрузка свечей через moexalgo.

    Исправление: moexalgo трактует `end` как исключающую правую границу.
    end = now + 2 дня гарантирует включение сегодняшних свечей
    (запас на UTC-сдвиг и ночной запуск).
    """
    try:
        t     = Ticker(ticker)
        end   = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        data = t.candles(start=start, end=end, period=tf)
        df   = pd.DataFrame(data)

        if df.empty:
            print(f"[fetch_data] Пустой ответ для {ticker} ({start} → {end})")
            return df

        first = pd.to_datetime(df['begin'].min()).strftime('%Y-%m-%d %H:%M')
        last  = pd.to_datetime(df['begin'].max()).strftime('%Y-%m-%d %H:%M')
        print(f"[fetch_data] {ticker} {tf}: {len(df)} свечей  |  {first}  →  {last}")
        return df

    except Exception as e:
        print(f"[fetch_data] Ошибка {ticker}: {e}")
        return pd.DataFrame()


def prepare_df(df_raw: pd.DataFrame, indicator: BaseIndicator) -> pd.DataFrame:
    """
    Добавляет datetime, EMA10 и колонки индикатора.
    Для DualConfirmIndicator дополнительно вычисляет колонки sub-индикаторов
    (нужны для панелей визуализации).
    """
    df = df_raw.copy()
    df['datetime'] = pd.to_datetime(df['begin'])
    df['ema10']    = df['close'].ewm(span=10, adjust=False).mean()
    df[indicator.column_name] = indicator.compute(df)

    if isinstance(indicator, DualConfirmIndicator):
        df[indicator.ind1.column_name] = indicator.ind1.compute(df)
        df[indicator.ind2.column_name] = indicator.ind2.compute(df)

    return df.reset_index(drop=True)


# ==============================================================================
# СТАТИСТИЧЕСКИЙ ДВИЖОК БЭКТЕСТА
# ==============================================================================

def run_backtest(ticker: str,
                 tf: str,
                 days: int = 30,
                 indicator: BaseIndicator = None,
                 forward_candles: int = 10,
                 min_move_pct: float = 0.3,
                 cooldown_candles: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Статистический бэктест с одним индикатором (может быть DualConfirmIndicator).

    Параметры
    ─────────
    indicator       : любой BaseIndicator, включая DualConfirmIndicator
    forward_candles : горизонт оценки (свечей вперёд)
    min_move_pct    : минимальное движение (%) для «успешного» сигнала
    cooldown_candles: не считать одинаковый паттерн повторно в пределах N свечей
                      (0 = отключено). Убирает каскадные дубликаты.

    Возвращает
    ──────────
    signals_df : детальная таблица каждого сигнала
    stats_df   : агрегация: без фильтра / с индикатором / по паттернам
    """
    indicator = indicator or NoIndicator()

    print(f"\n{'='*60}")
    print(f"Бэктест: {ticker} | {tf} | {days} дней")
    print(f"Индикатор : {indicator.plot_label or 'NoIndicator'}")
    print(f"Горизонт  : {forward_candles} свечей | мин. движение: {min_move_pct}%")
    if cooldown_candles:
        print(f"Cooldown  : {cooldown_candles} свечей")
    print('='*60)

    df_raw = fetch_data(ticker, tf, days)
    if df_raw.empty:
        print("Нет данных.")
        return pd.DataFrame(), pd.DataFrame()

    df = prepare_df(df_raw, indicator)

    # Два детектора: без фильтра (все паттерны) и с индикатором (только подтверждённые)
    no_filter   = PatternDetector(ScannerConfig(indicator=NoIndicator()))
    with_filter = PatternDetector(ScannerConfig(indicator=indicator))

    results: List[dict] = []
    # cooldown: {pattern_name: last_fired_idx}
    cooldown_tracker: Dict[str, int] = {}

    for i in range(12, len(df) - forward_candles):
        all_patterns = no_filter.get_pattern_at_index(df, i)
        if not all_patterns:
            continue

        confirmed_set = set(with_filter.get_pattern_at_index(df, i))
        row           = df.iloc[i]
        entry_price   = float(row['close'])
        future        = df.iloc[i + 1: i + 1 + forward_candles]

        for pattern in all_patterns:
            confirmed = pattern in confirmed_set

            # Cooldown фильтр: пропускаем если тот же паттерн был недавно
            if cooldown_candles and confirmed:
                last = cooldown_tracker.get(pattern, -999)
                if i - last < cooldown_candles:
                    confirmed = False
                else:
                    cooldown_tracker[pattern] = i

            is_bull = _is_bullish(pattern)

            if is_bull:
                max_fav = float((future['high'].max() - entry_price) / entry_price * 100)
                max_adv = float((entry_price - future['low'].min()) / entry_price * 100)
            else:
                max_fav = float((entry_price - future['low'].min()) / entry_price * 100)
                max_adv = float((future['high'].max() - entry_price) / entry_price * 100)

            wins = {}
            for n in [1, 3, 5, 10]:
                if n > forward_candles:
                    break
                fc = future['close'].values[:n]
                if is_bull:
                    wins[n] = bool((fc.max() - entry_price) / entry_price * 100 >= min_move_pct)
                else:
                    wins[n] = bool((entry_price - fc.min()) / entry_price * 100 >= min_move_pct)

            results.append({
                'datetime'    : row['datetime'],
                'pattern'     : pattern,
                'direction'   : 'bullish' if is_bull else 'bearish',
                'confirmed'   : confirmed,
                'entry_price' : round(entry_price, 4),
                'max_fav_%'   : round(max_fav, 2),
                'max_adv_%'   : round(max_adv, 2),
                **{f'win_c{n}': v for n, v in wins.items()},
            })

    if not results:
        print("Сигналы не найдены.")
        return pd.DataFrame(), pd.DataFrame()

    signals_df = pd.DataFrame(results)

    # ─── Агрегированная статистика ───
    horizon_col = f'win_c{min(10, forward_candles)}'
    if horizon_col not in signals_df.columns:
        horizon_col = [c for c in signals_df.columns if c.startswith('win_c')][-1]

    def _stats(mask: pd.Series, label: str) -> dict:
        sub = signals_df[mask]
        if sub.empty:
            return {'группа': label, 'кол-во': 0, 'WR%': '-',
                    'avg_fav%': '-', 'avg_adv%': '-', 'expectancy': '-'}
        total = len(sub)
        wins  = sub[horizon_col].sum()
        wr    = wins / total * 100
        w_sub = sub[sub[horizon_col]]
        l_sub = sub[~sub[horizon_col]]
        ag  = w_sub['max_fav_%'].mean() if len(w_sub) else 0.0
        al  = l_sub['max_adv_%'].mean() if len(l_sub) else 0.0
        exp = (wr / 100 * ag) - ((1 - wr / 100) * al)
        return {
            'группа'    : label,
            'кол-во'    : total,
            'WR%'       : round(wr, 1),
            'avg_fav%'  : round(sub['max_fav_%'].mean(), 2),
            'avg_adv%'  : round(sub['max_adv_%'].mean(), 2),
            'expectancy': round(exp, 2),
        }

    conf = signals_df['confirmed']
    rows = [
        _stats(pd.Series([True] * len(signals_df)), "Без фильтра"),
        _stats(conf, f"С индикатором"),
        _stats(conf & (signals_df['direction'] == 'bullish'), "  ↳ бычьи"),
        _stats(conf & (signals_df['direction'] == 'bearish'), "  ↳ медвежьи"),
    ]
    for pname in signals_df[conf]['pattern'].unique():
        rows.append(_stats(conf & (signals_df['pattern'] == pname), f"    ↳ {pname}"))

    stats_df = pd.DataFrame(rows)
    print("\n📊 СТАТИСТИКА:")
    print(stats_df.to_string(index=False))

    return signals_df, stats_df


# ==============================================================================
# ВИЗУАЛИЗАЦИЯ (1 / 2 / 3 панели в зависимости от типа индикатора)
# ==============================================================================

def _add_indicator_panel(fig: go.Figure,
                         df: pd.DataFrame,
                         ind: BaseIndicator,
                         row: int) -> None:
    """Добавляет панель одного индикатора в subplot."""
    col_name = ind.column_name
    if col_name not in df.columns:
        return

    if "macd" in col_name:
        colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df[col_name].fillna(0)]
        fig.add_trace(go.Bar(x=df.datetime, y=df[col_name],
                             marker_color=colors, name=ind.plot_label), row=row, col=1)
    else:
        color = '#29b6f6' if row == 2 else '#ce93d8'
        fig.add_trace(go.Scatter(x=df.datetime, y=df[col_name],
                                 line=dict(color=color, width=1.5),
                                 name=ind.plot_label), row=row, col=1)

    for lv in ind.get_level_lines():
        fig.add_hline(y=lv["value"],
                      line=dict(color=lv["color"], dash=lv["dash"], width=1),
                      row=row, col=1)


def visual_backtest_dual(ticker: str = 'SBER',
                         tf: str = '15min',
                         days: int = 20,
                         indicator: BaseIndicator = None,
                         forward_candles: int = 10,
                         min_move_pct: float = 0.3,
                         cooldown_candles: int = 0,
                         show_all: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Визуальный бэктест.

    Число панелей определяется автоматически:
      DualConfirmIndicator → 3 панели (свечи + ind1 + ind2)
      любой другой         → 2 панели (свечи + индикатор)
      NoIndicator          → 1 панель  (только свечи)

    show_all = False → рисует только подтверждённые (✓) сигналы
    show_all = True  → все сигналы; цвет показывает степень подтверждения

    Параметр по умолчанию — DualConfirmIndicator(BB%B, AdaptiveMFI).
    """
    if indicator is None:
        indicator = DualConfirmIndicator(
            BollingerPercentBIndicator(),
            AdaptiveMFIIndicator(),
        )

    signals_df, stats_df = run_backtest(
        ticker, tf, days, indicator, forward_candles, min_move_pct, cooldown_candles
    )

    df_raw = fetch_data(ticker, tf, days)
    if df_raw.empty:
        return signals_df, stats_df

    df = prepare_df(df_raw, indicator)

    # ─── Layout ───
    is_dual  = isinstance(indicator, DualConfirmIndicator)
    is_plain = not isinstance(indicator, NoIndicator)

    if is_dual:
        n_rows, heights = 3, [0.55, 0.225, 0.225]
        titles = [f"{ticker} | {tf}  —  двойное подтверждение",
                  indicator.ind1.plot_label, indicator.ind2.plot_label]
    elif is_plain:
        n_rows, heights = 2, [0.70, 0.30]
        titles = [f"{ticker} | {tf}", indicator.plot_label]
    else:
        n_rows, heights = 1, [1.0]
        titles = [f"{ticker} | {tf}"]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=heights,
                        subplot_titles=titles)

    # ─── Свечи ───
    fig.add_trace(go.Candlestick(
        x=df.datetime, open=df.open, high=df.high, low=df.low, close=df.close,
        name=ticker,
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.datetime, y=df.ema10,
        line=dict(color='orange', width=1.5), name='EMA10'
    ), row=1, col=1)

    # ─── Аннотации паттернов ───
    for _, sig in (signals_df.iterrows() if not signals_df.empty else []):
        if not show_all and not sig['confirmed']:
            continue

        is_bull = sig['direction'] == 'bullish'
        if sig['confirmed']:
            color = '#00e676' if is_bull else '#ff1744'
            label = sig['pattern'] + ' ✓'
        else:
            color = '#ffeb3b'
            label = sig['pattern']

        mask = df['datetime'] == sig['datetime']
        if not mask.any():
            continue
        i = df.index[mask][0]

        y_val  = float(df.loc[i, 'low'])  if is_bull else float(df.loc[i, 'high'])
        ay_val = -35 if is_bull else 35

        fig.add_annotation(
            x=sig['datetime'], y=y_val,
            text=label, showarrow=True, arrowhead=2,
            arrowcolor=color, bgcolor=color,
            font=dict(color='black', size=9),
            ay=ay_val, row=1, col=1
        )

    # ─── Панели индикаторов ───
    if is_dual:
        _add_indicator_panel(fig, df, indicator.ind1, row=2)
        _add_indicator_panel(fig, df, indicator.ind2, row=3)
    elif is_plain:
        _add_indicator_panel(fig, df, indicator, row=2)

    # ─── Подзаголовок со статистикой ───
    subtitle = ''
    if not stats_df.empty:
        row_ind = stats_df[stats_df['группа'] == 'С индикатором']
        if not row_ind.empty:
            r = row_ind.iloc[0]
            subtitle = (f"Подтверждённых сигналов: {r['кол-во']} | "
                        f"WR: {r['WR%']}% | "
                        f"avg max: {r['avg_fav%']}% | "
                        f"Горизонт: {forward_candles} свечей, мин {min_move_pct}%")

    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        title=dict(
            text=(f"Backtest {ticker} | {tf} | {indicator.plot_label}"
                  f"<br><sup>{subtitle}</sup>"),
            font=dict(size=13),
        ),
        height=820,
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#1e1e1e')

    fig.show()
    return signals_df, stats_df


# ==============================================================================
# ТОЧКА ВХОДА
# ==============================================================================

if __name__ == '__main__':

    # ── Dual: BB%B + Adaptive MFI (по умолчанию) ──────────────────────────
    visual_backtest_dual(
        ticker='NGJ6',
        tf='15min',
        days=30,
        indicator=AdaptiveMFIIndicator(),
        forward_candles=10,
        min_move_pct=0.5,
        cooldown_candles=3,   # не повторять тот же паттерн в пределах 3 свечей
        show_all=False,
    )

    # ── Только статистика (без графика) ───────────────────────────────────
    # signals_df, stats_df = run_backtest(
    #     ticker='SBER',
    #     tf='1h',
    #     days=60,
    #     indicator=DualConfirmIndicator(
    #         BollingerPercentBIndicator(),
    #         AdaptiveMFIIndicator(),
    #     ),
    #     forward_candles=10,
    #     min_move_pct=0.5,
    #     cooldown_candles=5,
    # )

    # ── Одиночный индикатор (2 панели) ────────────────────────────────────
    # visual_backtest_dual('SBER', '15min', days=20,
    #                      indicator=RSIIndicator(14, oversold=30, overbought=70))

    # ── Без индикатора (1 панель, только паттерны) ─────────────────────────
    # visual_backtest_dual('BRJ6', '15min', days=5, indicator=NoIndicator())

    # ── Сравнение нескольких вариантов ────────────────────────────────────
    # for label, ind in [
    #     ('No filter',   NoIndicator()),
    #     ('RSI',         RSIIndicator()),
    #     ('BB%B',        BollingerPercentBIndicator()),
    #     ('AI MFI',      AdaptiveMFIIndicator()),
    #     ('Dual',        DualConfirmIndicator(BollingerPercentBIndicator(),
    #                                          AdaptiveMFIIndicator())),
    # ]:
    #     print(f'\n>>> {label}')
    #     run_backtest('NGJ6', '15min', 30, ind, forward_candles=10)