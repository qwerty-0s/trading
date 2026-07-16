"""
backtest_engine.py
==================
Архитектура: Strategy → Signal → BacktestEngine → Portfolio → Report

Принципы:
  - Strategy не знает о движке. Возвращает Signal или None.
  - BacktestEngine не знает о логике входов. Только исполняет сигналы.
  - Position / Trade — иммутабельные dataclass-ы. Никакой бизнес-логики.
  - Portfolio — единственный владелец состояния.
  - Visualizer — отдельный класс, ничего не считает.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import ASSETS, TIMEFRAMES, ScannerConfig
from patterns.detector import PatternDetector

try:
    from load_data import BULLISH_PATTERNS, BEARISH_PATTERNS
except ImportError:
    BULLISH_PATTERNS = {
        "Hammer (Молот)", "Inverted Hammer (Перевернутый молот)",
        "Bullish Engulfing (Бычье поглощение)", "Bullish Harami (Бычье Харами)",
        "Bullish Harami Cross (Бычий Крест Харами)", "Piercing Line (Просвет в облаках)",
        "Morning Star (Утренняя звезда)", "Three White Soldiers (Три белых солдата)",
    }
    BEARISH_PATTERNS = {
        "Hanging Man (Висельник)", "Shooting Star (Падающая звезда)",
        "Bearish Engulfing (Медвежье поглощение)", "Bearish Harami (Медвежье Харами)",
        "Bearish Harami Cross (Медвежий Крест Харами)", "Dark Cloud Cover (Темные облака)",
        "Evening Star (Вечерняя звезда)", "Three Black Crows (Три черные вороны)",
    }


# ==============================================================================
# ТИПЫ ДАННЫХ
# ==============================================================================

SignalType = Literal["LONG", "SHORT"]


@dataclass
class Signal:
    """Торговый сигнал, возвращаемый стратегией."""
    type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    pattern_name: str = ""
    bar_index: int = 0
    bar_time: Optional[datetime] = None

    @property
    def risk(self) -> float:
        if self.type == "LONG":
            return self.entry_price - self.stop_loss
        return self.stop_loss - self.entry_price

    @property
    def reward(self) -> float:
        if self.type == "LONG":
            return self.take_profit - self.entry_price
        return self.entry_price - self.take_profit

    @property
    def rr_ratio(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0.0


@dataclass
class Position:
    """Открытая позиция."""
    signal: Signal
    size: float
    open_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def type(self) -> SignalType:
        return self.signal.type

    @property
    def entry(self) -> float:
        return self.signal.entry_price

    @property
    def sl(self) -> float:
        return self.signal.stop_loss

    @property
    def tp(self) -> float:
        return self.signal.take_profit


@dataclass
class Trade:
    """Закрытая сделка."""
    position: Position
    exit_price: float
    exit_time: datetime
    exit_bar: int
    exit_reason: Literal["SL", "TP", "SIGNAL", "FORCED"]
    pnl_points: float
    pnl_money: float

    @property
    def is_winner(self) -> bool:
        return self.pnl_money > 0

    @property
    def duration_bars(self) -> int:
        return self.exit_bar - self.position.signal.bar_index


# ==============================================================================
# ПОРТФЕЛЬ
# ==============================================================================

class Portfolio:
    def __init__(self, initial_balance: float = 100_000.0):
        self.initial_balance  = initial_balance
        self.balance          = initial_balance
        self.open_position: Optional[Position] = None
        self.trades: List[Trade] = []

    def open(self, signal: Signal, size: float = 1.0) -> Position:
        if self.open_position is not None:
            raise RuntimeError("Попытка открыть позицию, когда уже есть открытая.")
        pos = Position(signal=signal, size=size, open_time=signal.bar_time or datetime.utcnow())
        self.open_position = pos
        return pos

    def close(self, exit_price: float, exit_time: datetime,
              exit_bar: int, reason: Literal["SL", "TP", "SIGNAL", "FORCED"]) -> Trade:
        if self.open_position is None:
            raise RuntimeError("Нет открытой позиции для закрытия.")
        pos = self.open_position

        if pos.type == "LONG":
            pnl_points = exit_price - pos.entry
        else:
            pnl_points = pos.entry - exit_price

        pnl_money = pnl_points * pos.size
        self.balance += pnl_money

        trade = Trade(
            position=pos,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_bar=exit_bar,
            exit_reason=reason,
            pnl_points=pnl_points,
            pnl_money=pnl_money,
        )
        self.trades.append(trade)
        self.open_position = None
        return trade

    @property
    def is_flat(self) -> bool:
        return self.open_position is None

    @property
    def total_pnl(self) -> float:
        return self.balance - self.initial_balance


# ==============================================================================
# АБСТРАКТНАЯ СТРАТЕГИЯ
# ==============================================================================

class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self,
               df: pd.DataFrame,
               i: int,
               htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        ...

    def should_exit(self, position: Position, row: pd.Series) -> bool:
        if position.type == "LONG":
            return row['low'] <= position.sl or row['high'] >= position.tp
        else:
            return row['high'] >= position.sl or row['low'] <= position.tp

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


# ==============================================================================
# СТРАТЕГИЯ 1: Паттерн + AiMFI + HTF-тренд
# ==============================================================================

class CandlePatternStrategy(BaseStrategy):
    def __init__(self,
                 detector,
                 atr_sl_mult: float = 1.5,
                 rr_ratio: float = 2.0,
                 mfi_long_thr: float = 45.0,
                 mfi_short_thr: float = 55.0,
                 use_htf_filter: bool = True):
        self.detector       = detector
        self.atr_sl_mult    = atr_sl_mult
        self.rr_ratio       = rr_ratio
        self.mfi_long_thr   = mfi_long_thr
        self.mfi_short_thr  = mfi_short_thr
        self.use_htf_filter = use_htf_filter

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()

        hl  = df['high'] - df['low']
        hcp = (df['high'] - df['close'].shift()).abs()
        lcp = (df['low']  - df['close'].shift()).abs()
        df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()

        df['aimfi'] = self._simple_mfi(df, period=14)
        return df

    @staticmethod
    def _simple_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tp  = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        pos = rmf.where(tp > tp.shift(), 0.0)
        neg = rmf.where(tp < tp.shift(), 0.0)
        pos_sum = pos.rolling(period).sum()
        neg_sum = neg.rolling(period).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))

    def on_bar(self,
               df: pd.DataFrame,
               i: int,
               htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:

        if i < 50:
            return None

        row = df.iloc[i]

        if pd.isna(row.get('atr', np.nan)) or pd.isna(row.get('aimfi', np.nan)):
            return None

        htf_trend = self._get_htf_trend(htf_df, row['time']) if (self.use_htf_filter and htf_df is not None) else 0

        patterns = self.detector.get_pattern_at_index(df, i)
        if not patterns:
            return None

        p_name     = patterns[0]
        stop_dist  = row['atr'] * self.atr_sl_mult
        tp_dist    = stop_dist * self.rr_ratio

        is_bull_pattern = p_name in BULLISH_PATTERNS
        is_bear_pattern = p_name in BEARISH_PATTERNS

        if is_bull_pattern and row['aimfi'] < self.mfi_long_thr:
            if not self.use_htf_filter or htf_trend >= 0:
                return Signal(
                    type="LONG",
                    entry_price=row['close'],
                    stop_loss=row['close'] - stop_dist,
                    take_profit=row['close'] + tp_dist,
                    pattern_name=patterns[0],
                    bar_index=i,
                    bar_time=row['time'],
                )

        if is_bear_pattern and row['aimfi'] > self.mfi_short_thr:
            if not self.use_htf_filter or htf_trend <= 0:
                return Signal(
                    type="SHORT",
                    entry_price=row['close'],
                    stop_loss=row['close'] + stop_dist,
                    take_profit=row['close'] - tp_dist,
                    pattern_name=patterns[0],
                    bar_index=i,
                    bar_time=row['time'],
                )

        return None

    @staticmethod
    def _get_htf_trend(htf_df: pd.DataFrame, current_time: datetime) -> int:
        time_col = 'close_time' if 'close_time' in htf_df.columns else 'time'
        subset = htf_df[htf_df[time_col] <= current_time]
        if subset.empty:
            return 0
        last = subset.iloc[-1]
        if last['close'] > last['ema10_htf']:
            return 1
        if last['close'] < last['ema10_htf']:
            return -1
        return 0


# ==============================================================================
# СТРАТЕГИЯ 2: БЕЗ свечных паттернов (импульс AiMFI + HTF-тренд)
# ==============================================================================

class NoCandleTrendMfiStrategy(BaseStrategy):
    """
    Стратегия БЕЗ свечных паттернов для A/B тестирования.
    
    Логика триггера (вместо паттернов):
      - LONG: AiMFI находился в зоне перепроданности (< mfi_long_thr) 
              и развернулся вверх (пересек уровень снизу вверх).
      - SHORT: AiMFI находился в зоне перегретости (> mfi_short_thr) 
               и развернулся вниз (пересек уровень сверху вниз).
    """

    def __init__(self,
                 atr_sl_mult: float = 1.5,
                 rr_ratio: float = 2.0,
                 mfi_period: int = 14,
                 mfi_long_thr: float = 40.0,
                 mfi_short_thr: float = 60.0,
                 use_htf_filter: bool = True):
        self.atr_sl_mult    = atr_sl_mult
        self.rr_ratio       = rr_ratio
        self.mfi_period     = mfi_period
        self.mfi_long_thr   = mfi_long_thr
        self.mfi_short_thr  = mfi_short_thr
        self.use_htf_filter = use_htf_filter

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()

        hl  = df['high'] - df['low']
        hcp = (df['high'] - df['close'].shift()).abs()
        lcp = (df['low']  - df['close'].shift()).abs()
        df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()

        df['aimfi'] = self._simple_mfi(df, period=self.mfi_period)
        return df

    @staticmethod
    def _simple_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tp  = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        pos = rmf.where(tp > tp.shift(), 0.0)
        neg = rmf.where(tp < tp.shift(), 0.0)
        pos_sum = pos.rolling(period).sum()
        neg_sum = neg.rolling(period).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))

    def on_bar(self,
               df: pd.DataFrame,
               i: int,
               htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:

        if i < 50:
            return None

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        if (pd.isna(row.get('atr', np.nan)) or 
            pd.isna(row.get('aimfi', np.nan)) or 
            pd.isna(prev_row.get('aimfi', np.nan))):
            return None

        htf_trend = self._get_htf_trend(htf_df, row['time']) if (self.use_htf_filter and htf_df is not None) else 0

        stop_dist = row['atr'] * self.atr_sl_mult
        tp_dist   = stop_dist * self.rr_ratio

        curr_mfi = row['aimfi']
        prev_mfi = prev_row['aimfi']

        # Триггер входа по развороту импульса AiMFI через пороговые зоны
        is_long_trigger  = (prev_mfi < self.mfi_long_thr) and (curr_mfi >= self.mfi_long_thr)
        is_short_trigger = (prev_mfi > self.mfi_short_thr) and (curr_mfi <= self.mfi_short_thr)

        # --- LONG ---
        if is_long_trigger:
            if not self.use_htf_filter or htf_trend >= 0:
                return Signal(
                    type="LONG",
                    entry_price=row['close'],
                    stop_loss=row['close'] - stop_dist,
                    take_profit=row['close'] + tp_dist,
                    pattern_name="AiMFI_Cross_Up",
                    bar_index=i,
                    bar_time=row['time'],
                )

        # --- SHORT ---
        if is_short_trigger:
            if not self.use_htf_filter or htf_trend <= 0:
                return Signal(
                    type="SHORT",
                    entry_price=row['close'],
                    stop_loss=row['close'] + stop_dist,
                    take_profit=row['close'] - tp_dist,
                    pattern_name="AiMFI_Cross_Down",
                    bar_index=i,
                    bar_time=row['time'],
                )

        return None

    @staticmethod
    def _get_htf_trend(htf_df: pd.DataFrame, current_time: datetime) -> int:
        time_col = 'close_time' if 'close_time' in htf_df.columns else 'time'
        subset = htf_df[htf_df[time_col] <= current_time]
        if subset.empty:
            return 0
        last = subset.iloc[-1]
        if last['close'] > last['ema10_htf']:
            return 1
        if last['close'] < last['ema10_htf']:
            return -1
        return 0


# ==============================================================================
# ДВИЖОК БЭКТЕСТА
# ==============================================================================

class BacktestEngine:
    def __init__(self,
                 strategy: BaseStrategy,
                 initial_balance: float = 100_000.0,
                 position_size: float = 1.0,
                 warmup_bars: int = 50):
        self.strategy      = strategy
        self.portfolio     = Portfolio(initial_balance)
        self.position_size = position_size
        self.warmup_bars   = warmup_bars

    def run(self,
            df: pd.DataFrame,
            htf_df: Optional[pd.DataFrame] = None) -> Portfolio:
        df = self.strategy.prepare(df)

        if htf_df is not None:
            htf_df = htf_df.copy()
            htf_df['ema10_htf'] = htf_df['close'].ewm(span=10, adjust=False).mean()
            htf_bar_duration    = htf_df['time'].diff().median()
            htf_df['close_time'] = htf_df['time'] + htf_bar_duration

        for i in range(self.warmup_bars, len(df)):
            row = df.iloc[i]
            exited_by_signal = False

            if not self.portfolio.is_flat:
                pos = self.portfolio.open_position

                if self.strategy.should_exit(pos, row):
                    if pos.type == "LONG":
                        if row['low'] <= pos.sl:
                            reason, exit_p = "SL", pos.sl
                        elif row['high'] >= pos.tp:
                            reason, exit_p = "TP", pos.tp
                        else:
                            reason, exit_p = "SIGNAL", row['close']
                            exited_by_signal = True
                    else:
                        if row['high'] >= pos.sl:
                            reason, exit_p = "SL", pos.sl
                        elif row['low'] <= pos.tp:
                            reason, exit_p = "TP", pos.tp
                        else:
                            reason, exit_p = "SIGNAL", row['close']
                            exited_by_signal = True

                    self.portfolio.close(
                        exit_price=exit_p,
                        exit_time=row['time'],
                        exit_bar=i,
                        reason=reason,
                    )

            if self.portfolio.is_flat and not exited_by_signal:
                signal = self.strategy.on_bar(df, i, htf_df)
                if signal is not None:
                    self.portfolio.open(signal, size=self.position_size)

        if not self.portfolio.is_flat:
            last_row = df.iloc[-1]
            self.portfolio.close(
                exit_price=last_row['close'],
                exit_time=last_row['time'],
                exit_bar=len(df) - 1,
                reason="FORCED",
            )

        return self.portfolio

    def report(self) -> pd.DataFrame:
        trades = self.portfolio.trades
        p      = self.portfolio

        if not trades:
            print("❌ Сделок нет. Попробуй смягчить фильтры или увеличить период.")
            return pd.DataFrame()

        df_t = pd.DataFrame([{
            "type":         t.position.type,
            "entry":        t.position.entry,
            "exit":         t.exit_price,
            "sl":           t.position.sl,
            "tp":           t.position.tp,
            "pnl_pts":      round(t.pnl_points, 4),
            "pnl_money":    round(t.pnl_money, 2),
            "reason":       t.exit_reason,
            "pattern":      t.position.signal.pattern_name,
            "duration":     t.duration_bars,
            "entry_time":   t.position.open_time,
            "exit_time":    t.exit_time,
        } for t in trades])

        wins      = df_t['pnl_money'] > 0
        losses    = df_t['pnl_money'] < 0
        win_rate  = wins.mean() * 100
        avg_win   = df_t.loc[wins,  'pnl_money'].mean() if wins.any()   else 0
        avg_loss  = df_t.loc[losses,'pnl_money'].mean() if losses.any() else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        max_dd    = self._max_drawdown(df_t['pnl_money'])
        profit_factor = (
            df_t.loc[wins,   'pnl_money'].sum() /
            abs(df_t.loc[losses,'pnl_money'].sum())
            if losses.any() and df_t.loc[losses,'pnl_money'].sum() != 0 else float('inf')
        )

        print("=" * 55)
        print(f"  ИТОГИ БЭКТЕСТА")
        print("=" * 55)
        print(f"  Сделок:          {len(trades)}")
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Avg win:         {avg_win:.2f}")
        print(f"  Avg loss:        {avg_loss:.2f}")
        print(f"  Profit factor:   {profit_factor:.2f}")
        print(f"  Expectancy:      {expectancy:.2f} / сделку")
        print(f"  Max drawdown:    {max_dd:.2f}")
        print(f"  Итоговый PnL:    {p.total_pnl:.2f}  ({p.total_pnl/p.initial_balance*100:.1f}%)")
        print(f"  SL hits:         {(df_t['reason'] == 'SL').sum()}")
        print(f"  TP hits:         {(df_t['reason'] == 'TP').sum()}")
        print("=" * 55)

        return df_t

    @staticmethod
    def _max_drawdown(pnl_series: pd.Series) -> float:
        equity = pnl_series.cumsum()
        roll_max = equity.cummax()
        dd = (equity - roll_max)
        return dd.min()

    def visualize(self, df: pd.DataFrame, df_trades: Optional[pd.DataFrame] = None):
        trades = self.portfolio.trades
        if df_trades is None:
            df_trades = self.report()

        has_mfi = 'aimfi' in df.columns
        rows        = 2 if has_mfi else 1
        row_heights = [0.7, 0.3] if has_mfi else [1.0]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=["Price", "AiMFI"] if has_mfi else ["Price"],
        )

        fig.add_trace(go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name='Price',
        ), row=1, col=1)

        if 'ema10' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['ema10'],
                line=dict(color='orange', width=1.5), name='EMA10'
            ), row=1, col=1)

        for t in trades:
            sig   = t.position.signal
            color = '#00e676' if sig.type == 'LONG' else '#ff1744'
            arrow = 'triangle-up' if sig.type == 'LONG' else 'triangle-down'

            fig.add_trace(go.Scatter(
                x=[sig.bar_time], y=[sig.entry_price],
                mode='markers+text',
                marker=dict(symbol=arrow, size=12, color=color),
                text=[sig.pattern_name], textposition='top center',
                textfont=dict(size=8, color=color),
                name='',
                showlegend=False,
            ), row=1, col=1)

            exit_color = '#00e676' if t.is_winner else '#ff1744'
            fig.add_trace(go.Scatter(
                x=[t.exit_time], y=[t.exit_price],
                mode='markers',
                marker=dict(symbol='x', size=10, color=exit_color),
                name='',
                showlegend=False,
            ), row=1, col=1)

            fig.add_shape(
                type='line',
                x0=sig.bar_time, x1=t.exit_time,
                y0=sig.entry_price, y1=sig.entry_price,
                line=dict(color=color, width=1, dash='dot'),
                row=1, col=1
            )

        if has_mfi:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['aimfi'],
                line=dict(color='#7c4dff', width=1.5), name='AiMFI'
            ), row=2, col=1)
            for level, color, dash in [(80, 'red', 'dash'), (20, 'green', 'dash'), (50, 'gray', 'dot')]:
                fig.add_hline(y=level, line=dict(color=color, dash=dash, width=1), row=2, col=1)

        fig.update_layout(
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=850,
            showlegend=False,
            title='Backtest Results',
        )
        fig.show()


# ==============================================================================
# DATA LOADER
# ==============================================================================

async def fetch_candles(figi: str,
                        token: str,
                        interval,
                        days: int = 30) -> pd.DataFrame:
    try:
        from t_tech.invest import AsyncClient
        from t_tech.invest.utils import quotation_to_decimal
    except ImportError:
        raise ImportError("Установи tinkoff-investments: pip install tinkoff-investments")

    to_dt   = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days)
    rows    = []

    async with AsyncClient(token) as client:
        async for c in client.get_all_candles(figi=figi, from_=from_dt, to=to_dt, interval=interval):
            rows.append({
                "time":   c.time,
                "open":   float(quotation_to_decimal(c.open)),
                "high":   float(quotation_to_decimal(c.high)),
                "low":    float(quotation_to_decimal(c.low)),
                "close":  float(quotation_to_decimal(c.close)),
                "volume": c.volume,
            })

    return pd.DataFrame(rows)


# ==============================================================================
# ТОЧКА ВХОДА (A/B ТЕСТИРОВАНИЕ)
# ==============================================================================

async def main():
    import os
    from config import TINKOFF_TOKEN, ASSETS, ScannerConfig
    from patterns.detector import PatternDetector
    from data_loader.loader import InstrumentResolver
    from t_tech.invest import CandleInterval

    # Импортируем нашу выделенную стратегию K-Means AiMFI из модуля strategies
    from strategies.aimfi_htf_no_candles import AimfiHtfNoCandlesStrategy

    # 1. Получение токена и настройка актива
    TOKEN = os.getenv("TINKOFF_TOKEN", TINKOFF_TOKEN)

    asset = ASSETS[4]
    print(f"⏳ Резолвим FIGI для {asset.ticker}...")
    await InstrumentResolver.resolve_all([asset])
    if not asset.figi:
        print("❌ Не удалось найти FIGI. Проверь токен или тикер.")
        return
    FIGI = asset.figi
    print(f"✅ FIGI: {FIGI}")

    # 2. Загрузка данных (10m и 1h)
    print("⏳ Загружаем данные...")
    df_15m = await fetch_candles(FIGI, TOKEN, CandleInterval.CANDLE_INTERVAL_15_MIN, days=180)
    df_1h  = await fetch_candles(FIGI, TOKEN, CandleInterval.CANDLE_INTERVAL_HOUR,   days=180)
    
    if df_15m.empty or df_1h.empty:
        print("❌ Ошибка загрузки данных: один из датафреймов пуст.")
        return

    print(f"   15m: {len(df_15m)} свечей | 1h: {len(df_1h)} свечей")

    detector = PatternDetector(ScannerConfig())

    # ==========================================================================
    # 🧪 ТЕСТ A: СТРАТЕГИЯ СО СВЕЧНЫМИ ПАТТЕРНАМИ
    # ==========================================================================
    print("\n" + "=" * 60)
    print("📊 1. [ТЕСТ A] CandlePatternStrategy (Свечные паттерны + MFI)")
    print("=" * 60)
    strat_candles = CandlePatternStrategy(
        detector=detector,
        atr_sl_mult=1.5,
        rr_ratio=2.0,
        mfi_long_thr=45.0,
        mfi_short_thr=55.0,
        use_htf_filter=True,
    )
    engine_a = BacktestEngine(strat_candles, initial_balance=100_000, position_size=10)
    engine_a.run(df_15m, htf_df=df_1h)
    df_trades_a = engine_a.report()

    # ==========================================================================
    # 🧪 ТЕСТ B: СТРАТЕГИЯ БЕЗ СВЕЧЕЙ (Pure K-Means AiMFI + HTF)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("📊 2. [ТЕСТ B] AimfiHtfNoCandlesStrategy (K-Means AiMFI из strategies/)")
    print("=" * 60)
    strat_no_candles = AimfiHtfNoCandlesStrategy(
        atr_sl_mult=1.5,
        rr_ratio=2.0,
        mfi_period=14,
        mfi_training_size=300,  # Размер окна для K-Means кластеризации
        mfi_long_thr=40.0,
        mfi_short_thr=60.0,
        use_htf_filter=True,
    )
    engine_b = BacktestEngine(strat_no_candles, initial_balance=100_000, position_size=10)
    engine_b.run(df_15m, htf_df=df_1h)
    df_trades_b = engine_b.report()

    # Визуализация результатов ТЕСТА B в Plotly
    engine_b.visualize(df_15m, df_trades_b)


if __name__ == "__main__":
    asyncio.run(main())
