"""
backtest_engine.py
==================
Чистый движок бэктестинга без внешних зависимостей от бота.
Архитектура: Strategy → Signal → BacktestEngine → Portfolio → Report
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==============================================================================
# 1. ТИПЫ ДАННЫХ
# ==============================================================================

SignalType = Literal["LONG", "SHORT"]


@dataclass
class Signal:
    """Торговый сигнал, генерируемый стратегией."""
    type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    pattern_name: str = "ML_Signal"
    bar_index: int = 0
    bar_time: Optional[datetime] = None

    @property
    def risk(self) -> float:
        return (self.entry_price - self.stop_loss) if self.type == "LONG" else (self.stop_loss - self.entry_price)

    @property
    def reward(self) -> float:
        return (self.take_profit - self.entry_price) if self.type == "LONG" else (self.entry_price - self.take_profit)

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
# 2. ПОРТФЕЛЬ
# ==============================================================================

class Portfolio:
    def __init__(self, initial_balance: float = 100_000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.open_position: Optional[Position] = None
        self.trades: List[Trade] = []

    def open(self, signal: Signal, size: float = 1.0) -> Position:
        if self.open_position is not None:
            raise RuntimeError("Уже есть открытая позиция!")
        pos = Position(signal=signal, size=size, open_time=signal.bar_time or datetime.utcnow())
        self.open_position = pos
        return pos

    def close(self, exit_price: float, exit_time: datetime, exit_bar: int,
              reason: Literal["SL", "TP", "SIGNAL", "FORCED"]) -> Trade:
        if self.open_position is None:
            raise RuntimeError("Нет позиции для закрытия!")
        pos = self.open_position

        pnl_points = (exit_price - pos.entry) if pos.type == "LONG" else (pos.entry - exit_price)
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
# 3. БАЗОВЫЙ КЛАСС СТРАТЕГИИ
# ==============================================================================

class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self, df: pd.DataFrame, i: int, htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """Расчет сигнала на каждом баре."""
        ...

    def should_exit(self, position: Position, row: pd.Series) -> bool:
        """Проверка достижения SL или TP."""
        if position.type == "LONG":
            return row['low'] <= position.sl or row['high'] >= position.tp
        return row['high'] >= position.sl or row['low'] <= position.tp

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предварительный расчет необходимых индикаторов."""
        return df


# ==============================================================================
# 4. ДВИЖОК БЭКТЕСТА
# ==============================================================================

class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, initial_balance: float = 100_000.0,
                 position_size: float = 1.0, warmup_bars: int = 50):
        self.strategy = strategy
        self.portfolio = Portfolio(initial_balance)
        self.position_size = position_size
        self.warmup_bars = warmup_bars

    def run(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> Portfolio:
        df = self.strategy.prepare(df)

        if htf_df is not None:
            htf_df = htf_df.copy()
            htf_df['ema10_htf'] = htf_df['close'].ewm(span=10, adjust=False).mean()
            htf_bar_duration = htf_df['time'].diff().median()
            htf_df['close_time'] = htf_df['time'] + htf_bar_duration

        for i in range(self.warmup_bars, len(df)):
            row = df.iloc[i]
            exited_by_signal = False

            # Проверяем исполнение SL/TP текущей позиции
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

                    self.portfolio.close(exit_price=exit_p, exit_time=row['time'], exit_bar=i, reason=reason)

            # Проверяем входы, если позиция свободна
            if self.portfolio.is_flat and not exited_by_signal:
                signal = self.strategy.on_bar(df, i, htf_df)
                if signal is not None:
                    self.portfolio.open(signal, size=self.position_size)

        # Принудительное закрытие в конце истории
        if not self.portfolio.is_flat:
            last_row = df.iloc[-1]
            self.portfolio.close(
                exit_price=last_row['close'],
                exit_time=last_row['time'],
                exit_bar=len(df) - 1,
                reason="FORCED"
            )

        return self.portfolio

    def report(self) -> pd.DataFrame:
        trades = self.portfolio.trades
        p = self.portfolio

        if not trades:
            print("❌ Сделок не совершено.")
            return pd.DataFrame()

        df_t = pd.DataFrame([{
            "type": t.position.type,
            "entry": t.position.entry,
            "exit": t.exit_price,
            "sl": t.position.sl,
            "tp": t.position.tp,
            "pnl_pts": round(t.pnl_points, 4),
            "pnl_money": round(t.pnl_money, 2),
            "reason": t.exit_reason,
            "pattern": t.position.signal.pattern_name,
            "duration": t.duration_bars,
            "entry_time": t.position.open_time,
            "exit_time": t.exit_time,
        } for t in trades])

        wins = df_t['pnl_money'] > 0
        losses = df_t['pnl_money'] < 0
        win_rate = wins.mean() * 100
        avg_win = df_t.loc[wins, 'pnl_money'].mean() if wins.any() else 0
        avg_loss = df_t.loc[losses, 'pnl_money'].mean() if losses.any() else 0
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

        max_dd = self._max_drawdown(df_t['pnl_money'])
        profit_factor = (
            df_t.loc[wins, 'pnl_money'].sum() / abs(df_t.loc[losses, 'pnl_money'].sum())
            if losses.any() and df_t.loc[losses, 'pnl_money'].sum() != 0 else float('inf')
        )

        print("=" * 55)
        print("  ИТОГИ БЭКТЕСТА (ML Lab)")
        print("=" * 55)
        print(f"  Сделок:          {len(trades)}")
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Avg win:         {avg_win:.2f}")
        print(f"  Avg loss:        {avg_loss:.2f}")
        print(f"  Profit factor:   {profit_factor:.2f}")
        print(f"  Expectancy:      {expectancy:.2f} / сделку")
        print(f"  Max drawdown:    {max_dd:.2f}")
        print(f"  Итоговый PnL:    {p.total_pnl:.2f}  ({p.total_pnl/p.initial_balance*100:.1f}%)")
        print(f"  SL / TP hits:    {(df_t['reason'] == 'SL').sum()} / {(df_t['reason'] == 'TP').sum()}")
        print("=" * 55)

        return df_t

    @staticmethod
    def _max_drawdown(pnl_series: pd.Series) -> float:
        equity = pnl_series.cumsum()
        roll_max = equity.cummax()
        return (equity - roll_max).min()

    def visualize(self, df: pd.DataFrame, df_trades: Optional[pd.DataFrame] = None):
        trades = self.portfolio.trades
        has_mfi = 'aimfi' in df.columns

        fig = make_subplots(
            rows=2 if has_mfi else 1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3] if has_mfi else [1.0],
            subplot_titles=["Price", "AiMFI"] if has_mfi else ["Price"],
        )

        fig.add_trace(go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            name='Price'
        ), row=1, col=1)

        for t in trades:
            sig = t.position.signal
            color = '#00e676' if sig.type == 'LONG' else '#ff1744'
            arrow = 'triangle-up' if sig.type == 'LONG' else 'triangle-down'

            fig.add_trace(go.Scatter(
                x=[sig.bar_time], y=[sig.entry_price],
                mode='markers', marker=dict(symbol=arrow, size=10, color=color),
                showlegend=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[t.exit_time], y=[t.exit_price],
                mode='markers', marker=dict(symbol='x', size=8, color='#ff1744' if not t.is_winner else '#00e676'),
                showlegend=False
            ), row=1, col=1)

        if has_mfi:
            fig.add_trace(go.Scatter(x=df['time'], y=df['aimfi'], line=dict(color='#7c4dff'), name='AiMFI'), row=2, col=1)

        fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=800)
        fig.show()


# ==============================================================================
# 5. УТИЛИТА ЗАГРУЗКИ (T-INVEST API)
# ==============================================================================

async def fetch_candles(figi: str, token: str, interval, days: int = 30) -> pd.DataFrame:
    """Загрузка свечей из Т-Инвестиций."""
    try:
        from t_tech.invest import AsyncClient
        from t_tech.invest.utils import quotation_to_decimal
    except ImportError:
        try:
            from tinkoff.invest import AsyncClient
            from tinkoff.invest.utils import quotation_to_decimal
        except ImportError:
            raise ImportError("Установите пакет tinkoff-investments: pip install tinkoff-investments")

    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days)
    rows = []

    async with AsyncClient(token) as client:
        async for c in client.get_all_candles(figi=figi, from_=from_dt, to=to_dt, interval=interval):
            rows.append({
                "time": c.time,
                "open": float(quotation_to_decimal(c.open)),
                "high": float(quotation_to_decimal(c.high)),
                "low": float(quotation_to_decimal(c.low)),
                "close": float(quotation_to_decimal(c.close)),
                "volume": c.volume,
            })

    return pd.DataFrame(rows)