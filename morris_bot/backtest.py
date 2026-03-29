"""
backtest.py — инструменты визуализации и статистического бэктеста для MorrisBot.

Публичный API
─────────────
visual_backtest(...)       — быстрый визуальный просмотр паттернов на графике
run_backtest(...)          — статистический движок: signals_df + stats_df
visual_backtest_dual(...)  — полный бэктест с графиком (1 / 2 / 3 панели)
test_telegram(...)         — проверка роутинга Telegram
test_all_topics()          — тест всех топиков из .env
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from morris_bot.config import ScannerConfig
from morris_bot.indicators.base import BaseIndicator, NoIndicator
from morris_bot.indicators.dual import DualConfirmIndicator
from morris_bot.indicators.bollinger import BollingerPercentBIndicator
from morris_bot.indicators.mfi import AdaptiveMFIIndicator
from morris_bot.patterns.detector import PatternDetector
from morris_bot.patterns.confirmation import filter_confirmed, needs_confirmation
from morris_bot.bot.morris_bot import MorrisBot
from morris_bot.bot.router import TelegramRouter

load_dotenv()
warnings.filterwarnings("ignore")

_BULLISH_KEYWORDS = {'bull', 'hammer', 'morning', 'soldier', 'piercing'}


def _is_bullish(pattern: str) -> bool:
    return any(k in pattern.lower() for k in _BULLISH_KEYWORDS)


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def _fetch_data(ticker: str, tf: str, days: int = 30) -> pd.DataFrame:
    """Загрузка через MorrisBot.fetch_data (с фиксом end+2 дня)."""
    return MorrisBot("", "").fetch_data(ticker, tf, days)


def _prepare_df(df_raw: pd.DataFrame, indicator: BaseIndicator) -> pd.DataFrame:
    """
    Добавляет datetime, EMA10 и колонки индикатора.
    Для DualConfirmIndicator дополнительно вычисляет колонки sub-индикаторов
    (нужны для отдельных панелей визуализации).
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
# БЫСТРЫЙ ВИЗУАЛЬНЫЙ БЭКТЕСТ (без статистики)
# ==============================================================================

def visual_backtest(ticker: str = 'SBER',
                    tf: str = '15min',
                    days_back: int = 10,
                    indicator: Optional[BaseIndicator] = None,
                    use_confirmation: bool = True):
    """
    Визуальный бэктест с отрисовкой паттернов, EMA10 и индикатора.

    Args:
        ticker:           Тикер инструмента.
        tf:               Таймфрейм.
        days_back:        Глубина истории в днях.
        indicator:        Опциональный индикатор. Если None — берётся из get_detector().
        use_confirmation: True — показывать только подтверждённые паттерны.
    """
    bot = MorrisBot("", "")

    if indicator:
        bot.detectors[ticker] = PatternDetector(ScannerConfig(indicator=indicator))

    df_raw = bot.fetch_data(ticker, tf, days=days_back)
    if df_raw.empty:
        print("Нет данных для бэктеста.")
        return

    detector = bot.get_detector(ticker)
    df       = bot._prepare_df(df_raw, detector)
    ind      = detector.config.indicator
    ind_col  = ind.column_name
    has_ind  = ind_col in df.columns and not isinstance(ind, NoIndicator)

    rows        = 2 if has_ind else 1
    row_heights = [0.7, 0.3] if has_ind else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[f"Backtest {ticker} {tf}", ind.plot_label if has_ind else ""]
    )

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

    for i in range(12, len(df)):
        raw_patterns = detector.get_pattern_at_index(df, i)
        patterns     = filter_confirmed(raw_patterns, df, i) if use_confirmation else raw_patterns

        for p in patterns:
            is_bull = _is_bullish(p)
            color   = "lime" if is_bull else "red"
            y_val   = df.loc[i, 'low'] if is_bull else df.loc[i, 'high']
            ay_val  = -30 if is_bull else 30
            label   = f"{'✅ ' if use_confirmation and needs_confirmation(p) else ''}{p}"

            fig.add_annotation(
                x=df.loc[i, 'datetime'], y=y_val,
                text=label, showarrow=True, arrowhead=2,
                arrowcolor=color, bgcolor=color,
                font=dict(color="black", size=9),
                ay=ay_val, row=1, col=1
            )

    if has_ind:
        if "macd" in ind_col:
            bar_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df[ind_col]]
            fig.add_trace(go.Bar(
                x=df.datetime, y=df[ind_col],
                marker_color=bar_colors, name=ind.plot_label
            ), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df.datetime, y=df[ind_col],
                line=dict(color='#7c4dff', width=1.5), name=ind.plot_label
            ), row=2, col=1)

        for level in ind.get_level_lines():
            fig.add_hline(
                y=level["value"],
                line=dict(color=level["color"], dash=level["dash"], width=1),
                row=2, col=1
            )

    confirm_title = "с подтверждением" if use_confirmation else "без подтверждения"
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        title=f"Backtest {ticker} | {tf} | {confirm_title}",
        height=700,
        showlegend=True,
    )
    fig.show()


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

    df_raw = _fetch_data(ticker, tf, days)
    if df_raw.empty:
        print("Нет данных.")
        return pd.DataFrame(), pd.DataFrame()

    df = _prepare_df(df_raw, indicator)

    no_filter   = PatternDetector(ScannerConfig(indicator=NoIndicator()))
    with_filter = PatternDetector(ScannerConfig(indicator=indicator))

    results: List[dict] = []
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
        ag    = w_sub['max_fav_%'].mean() if len(w_sub) else 0.0
        al    = l_sub['max_adv_%'].mean() if len(l_sub) else 0.0
        exp   = (wr / 100 * ag) - ((1 - wr / 100) * al)
        return {
            'группа'    : label,
            'кол-во'    : total,
            'WR%'       : round(wr, 1),
            'avg_fav%'  : round(sub['max_fav_%'].mean(), 2),
            'avg_adv%'  : round(sub['max_adv_%'].mean(), 2),
            'expectancy': round(exp, 2),
        }

    conf      = signals_df['confirmed']
    rows_stat = [
        _stats(pd.Series([True] * len(signals_df)), "Без фильтра"),
        _stats(conf, "С индикатором"),
        _stats(conf & (signals_df['direction'] == 'bullish'), "  ↳ бычьи"),
        _stats(conf & (signals_df['direction'] == 'bearish'), "  ↳ медвежьи"),
    ]
    for pname in signals_df[conf]['pattern'].unique():
        rows_stat.append(_stats(conf & (signals_df['pattern'] == pname), f"    ↳ {pname}"))

    stats_df = pd.DataFrame(rows_stat)
    print("\n📊 СТАТИСТИКА:")
    print(stats_df.to_string(index=False))

    return signals_df, stats_df


# ==============================================================================
# ПОЛНЫЙ ВИЗУАЛЬНЫЙ БЭКТЕСТ (1 / 2 / 3 панели)
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
    Полный визуальный бэктест со статистикой.

    Число панелей определяется автоматически:
      DualConfirmIndicator → 3 панели (свечи + ind1 + ind2)
      любой другой         → 2 панели (свечи + индикатор)
      NoIndicator          → 1 панель  (только свечи)

    show_all = False → рисует только подтверждённые (✓) сигналы
    show_all = True  → все сигналы; жёлтый — неподтверждённые

    По умолчанию — DualConfirmIndicator(BB%B, AdaptiveMFI).
    """
    if indicator is None:
        indicator = DualConfirmIndicator(
            BollingerPercentBIndicator(),
            AdaptiveMFIIndicator(),
        )

    signals_df, stats_df = run_backtest(
        ticker, tf, days, indicator, forward_candles, min_move_pct, cooldown_candles
    )

    df_raw = _fetch_data(ticker, tf, days)
    if df_raw.empty:
        return signals_df, stats_df

    df = _prepare_df(df_raw, indicator)

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

    if is_dual:
        _add_indicator_panel(fig, df, indicator.ind1, row=2)
        _add_indicator_panel(fig, df, indicator.ind2, row=3)
    elif is_plain:
        _add_indicator_panel(fig, df, indicator, row=2)

    subtitle = ''
    if not stats_df.empty:
        row_ind = stats_df[stats_df['группа'] == 'С индикатором']
        if not row_ind.empty:
            r = row_ind.iloc[0]
            subtitle = (f"Подтверждённых: {r['кол-во']} | "
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
# УТИЛИТЫ TELEGRAM
# ==============================================================================

def test_telegram(ticker: str, tf: str):
    """Отправляет тестовое сообщение в тему группы для пары ticker × tf."""
    token    = os.getenv("TG_BOT_TOKEN", "")
    group_id = os.getenv("TG_GROUP_ID", "")

    if not token or token == "your_bot_token_here":
        print("TG_BOT_TOKEN не задан в .env")
        return
    if not group_id or group_id == "-1001234567890":
        print("TG_GROUP_ID не задан в .env")
        return

    router    = TelegramRouter(token, group_id)
    key       = f"{ticker.upper()}_{tf.upper()}"
    thread_id = router._topics.get(key)

    print(f"[test_telegram] {ticker} | {tf} | {key} | thread_id: {thread_id}")

    if thread_id is None:
        print(f"Ключ '{key}' не найден. Нужна строка: TOPIC_{key}=<thread_id>")
        print(f"Доступные: {list(router._topics.keys())}")
        return

    router.send_message(ticker, tf,
        f"*Тест подключения*\nТикер: `{ticker}` | ТФ: `{tf}`\n"
        f"Ключ: `{key}` thread\\_id: `{thread_id}`\nБот работает корректно"
    )
    print(f"Отправлено в thread_id={thread_id}")


def test_all_topics():
    """Отправляет тестовое сообщение во все темы из .env."""
    token    = os.getenv("TG_BOT_TOKEN", "")
    group_id = os.getenv("TG_GROUP_ID", "")
    router   = TelegramRouter(token, group_id)

    if not router._topics:
        print("Нет топиков в .env")
        return

    for key, thread_id in router._topics.items():
        ticker, tf = key.split("_", 1)
        router.send_message(ticker, tf.lower(),
            f"✅ *MorrisBot запущен*\n`{ticker}` | `{tf.lower()}` | thread\\_id: `{thread_id}`")
        print(f"Отправлено: {key} → thread_id={thread_id}")
