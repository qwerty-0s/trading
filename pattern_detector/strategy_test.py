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
        """Расстояние до стопа в пунктах."""
        if self.type == "LONG":
            return self.entry_price - self.stop_loss
        return self.stop_loss - self.entry_price

    @property
    def reward(self) -> float:
        """Расстояние до TP в пунктах."""
        if self.type == "LONG":
            return self.take_profit - self.entry_price
        return self.entry_price - self.take_profit

    @property
    def rr_ratio(self) -> float:
        """Risk / Reward ratio."""
        return self.reward / self.risk if self.risk > 0 else 0.0


@dataclass
class Position:
    """Открытая позиция."""
    signal: Signal
    size: float                        # Кол-во лотов / контрактов
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
    pnl_points: float                  # PnL в пунктах
    pnl_money: float                   # PnL в деньгах (с учётом size)

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
    """
    Единственный владелец состояния: баланс, открытые позиции, история.
    Не принимает торговых решений.
    """

    def __init__(self, initial_balance: float = 100_000.0):
        self.initial_balance  = initial_balance
        self.balance          = initial_balance
        self.open_position: Optional[Position] = None
        self.trades: List[Trade] = []

    def open(self, signal: Signal, size: float = 1.0) -> Position:
        if self.open_position is not None:
            raise RuntimeError("Попытка открыть позицию, когда уже есть открытая.")
        pos = Position(signal=signal, size=size)
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
    """
    Интерфейс стратегии.

    Движок вызывает:
      - on_bar(df, i, htf_df) → Optional[Signal]   при каждой свече
      - should_exit(position, row) → bool           пока позиция открыта

    Стратегия НЕ знает о Portfolio и BacktestEngine.
    """

    @abstractmethod
    def on_bar(self,
               df: pd.DataFrame,
               i: int,
               htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:
        """
        Вернуть Signal если нужно открыть позицию, иначе None.
        Вызывается только когда портфель флэт.
        """
        ...

    def should_exit(self, position: Position, row: pd.Series) -> bool:
        """
        Проверить стоп и тейк. Переопределяй если нужна кастомная логика выхода
        (трейлинг-стоп, time-exit и т.д.).
        """
        if position.type == "LONG":
            return row['low'] <= position.sl or row['high'] >= position.tp
        else:
            return row['high'] >= position.sl or row['low'] <= position.tp

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Хук для добавления индикаторов в датафрейм перед запуском теста.
        Переопредели в своей стратегии.
        """
        return df


# ==============================================================================
# ПРИМЕР СТРАТЕГИИ: Паттерн + AiMFI + HTF-тренд
# ==============================================================================

class CandlePatternStrategy(BaseStrategy):
    """
    Входы по свечным паттернам с фильтрами:
      - AiMFI (адаптивный Money Flow Index) для подтверждения перегрева/перепроданности
      - HTF-тренд (EMA10 на старшем таймфрейме)
      - ATR-стоп и настраиваемый R:R

    Параметры специально сделаны мягче, чем в исходном файле:
      - mfi_long_thr  = 45  (было 35)
      - mfi_short_thr = 55  (было 65)
      - rr_ratio      = 2.0 (было 1.0)
    """

    def __init__(self,
                 detector,                    # твой PatternDetector
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

    # ------------------------------------------------------------------ prepare
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMA10
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()

        # ATR(14)
        hl  = df['high'] - df['low']
        hcp = (df['high'] - df['close'].shift()).abs()
        lcp = (df['low']  - df['close'].shift()).abs()
        df['atr'] = pd.concat([hl, hcp, lcp], axis=1).max(axis=1).rolling(14).mean()

        # AiMFI — адаптивная версия MFI (если нет своего модуля)
        # Заменяй на: df['aimfi'] = AdaptiveMFIIndicator().compute(df)
        df['aimfi'] = self._simple_mfi(df, period=14)

        return df

    @staticmethod
    def _simple_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Классический MFI как fallback если нет AdaptiveMFIIndicator."""
        tp  = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        pos = rmf.where(tp > tp.shift(), 0.0)
        neg = rmf.where(tp < tp.shift(), 0.0)
        pos_sum = pos.rolling(period).sum()
        neg_sum = neg.rolling(period).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))

    # ------------------------------------------------------------------ on_bar
    def on_bar(self,
               df: pd.DataFrame,
               i: int,
               htf_df: Optional[pd.DataFrame] = None) -> Optional[Signal]:

        if i < 50:
            return None

        row = df.iloc[i]

        if pd.isna(row.get('atr', np.nan)) or pd.isna(row.get('aimfi', np.nan)):
            return None

        # HTF тренд
        htf_trend = self._get_htf_trend(htf_df, row['time']) if (self.use_htf_filter and htf_df is not None) else 0

        patterns = self.detector.get_pattern_at_index(df, i)
        if not patterns:
            return None

        p_name     = patterns[0].lower()
        stop_dist  = row['atr'] * self.atr_sl_mult
        tp_dist    = stop_dist * self.rr_ratio

        # LONG
        is_bull_pattern = any(k in p_name for k in ['bull', 'hammer', 'morning', 'piercing', 'soldier'])
        if is_bull_pattern and row['aimfi'] < self.mfi_long_thr:
            if not self.use_htf_filter or htf_trend >= 0:   # 0 = нейтральный, разрешаем
                return Signal(
                    type="LONG",
                    entry_price=row['close'],
                    stop_loss=row['close'] - stop_dist,
                    take_profit=row['close'] + tp_dist,
                    pattern_name=patterns[0],
                    bar_index=i,
                    bar_time=row['time'],
                )

        # SHORT
        is_bear_pattern = any(k in p_name for k in ['bear', 'star', 'hanging', 'shooting', 'cloud', 'crow'])
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
        """1 = бычий, -1 = медвежий, 0 = нет данных."""
        subset = htf_df[htf_df['time'] <= current_time]
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
    """
    Исполняет сигналы стратегии на исторических данных.
    Не содержит никакой торговой логики — только механику.

    Поддерживает:
      - Один открытый инструмент одновременно (можно расширить)
      - Позиционирование по фиксированному лоту (size) или % от баланса
      - HTF-датафрейм для передачи в стратегию
    """

    def __init__(self,
                 strategy: BaseStrategy,
                 initial_balance: float = 100_000.0,
                 position_size: float = 1.0,
                 warmup_bars: int = 50):
        self.strategy      = strategy
        self.portfolio     = Portfolio(initial_balance)
        self.position_size = position_size
        self.warmup_bars   = warmup_bars

    # ---------------------------------------------------------------- run
    def run(self,
            df: pd.DataFrame,
            htf_df: Optional[pd.DataFrame] = None) -> Portfolio:
        """
        Основной цикл. Возвращает Portfolio с историей всех сделок.

        df     — основной таймфрейм (уже с индикаторами после prepare())
        htf_df — старший таймфрейм (опционально)
        """
        df = self.strategy.prepare(df)

        if htf_df is not None:
            htf_df = htf_df.copy()
            htf_df['ema10_htf'] = htf_df['close'].ewm(span=10, adjust=False).mean()

        for i in range(self.warmup_bars, len(df)):
            row = df.iloc[i]

            # --- Обработка открытой позиции ---
            exited_by_signal = False  # флаг: вышли по сигналу на этой свече

            if not self.portfolio.is_flat:
                pos = self.portfolio.open_position

                if self.strategy.should_exit(pos, row):
                    # Определяем причину и цену выхода
                    if pos.type == "LONG":
                        if row['low'] <= pos.sl:
                            reason, exit_p = "SL", pos.sl
                        elif row['high'] >= pos.tp:
                            reason, exit_p = "TP", pos.tp
                        else:
                            # Выход по сигналу — цена закрытия свечи
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

            # --- Поиск входа (только когда флэт И не закрылись только что по сигналу) ---
            # exited_by_signal=True блокирует немедленный реэнтри на той же свече
            if self.portfolio.is_flat and not exited_by_signal:
                signal = self.strategy.on_bar(df, i, htf_df)
                if signal is not None:
                    self.portfolio.open(signal, size=self.position_size)

        # Принудительно закрываем незакрытую позицию
        if not self.portfolio.is_flat:
            last_row = df.iloc[-1]
            self.portfolio.close(
                exit_price=last_row['close'],
                exit_time=last_row['time'],
                exit_bar=len(df) - 1,
                reason="FORCED",
            )

        return self.portfolio

    # ---------------------------------------------------------------- report
    def report(self) -> pd.DataFrame:
        """Печатает статистику и возвращает DataFrame сделок."""
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

    # ---------------------------------------------------------------- visualize
    def visualize(self, df: pd.DataFrame, df_trades: Optional[pd.DataFrame] = None):
        """Рисует свечи + EMA + AiMFI + точки входов/выходов."""
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

        # Свечи
        fig.add_trace(go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name='Price',
        ), row=1, col=1)

        # EMA10
        if 'ema10' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['ema10'],
                line=dict(color='orange', width=1.5), name='EMA10'
            ), row=1, col=1)

        # Входы и выходы
        for t in trades:
            sig   = t.position.signal
            color = '#00e676' if sig.type == 'LONG' else '#ff1744'
            arrow = 'triangle-up' if sig.type == 'LONG' else 'triangle-down'

            # Вход
            fig.add_trace(go.Scatter(
                x=[sig.bar_time], y=[sig.entry_price],
                mode='markers+text',
                marker=dict(symbol=arrow, size=12, color=color),
                text=[sig.pattern_name], textposition='top center',
                textfont=dict(size=8, color=color),
                name='',
                showlegend=False,
            ), row=1, col=1)

            # Выход
            exit_color = '#00e676' if t.is_winner else '#ff1744'
            fig.add_trace(go.Scatter(
                x=[t.exit_time], y=[t.exit_price],
                mode='markers',
                marker=dict(symbol='x', size=10, color=exit_color),
                name='',
                showlegend=False,
            ), row=1, col=1)

            # Линия позиции
            fig.add_shape(
                type='line',
                x0=sig.bar_time, x1=t.exit_time,
                y0=sig.entry_price, y1=sig.entry_price,
                line=dict(color=color, width=1, dash='dot'),
                row=1, col=1
            )

        # AiMFI
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
# DATA LOADER (async, Tinkoff)
# ==============================================================================

async def fetch_candles(figi: str,
                        token: str,
                        interval,
                        days: int = 30) -> pd.DataFrame:
    """
    Загружает свечи через Tinkoff Invest API.
    Замени импорты на свои, если используешь другой брокер.
    """
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
# ТОЧКА ВХОДА
# ==============================================================================

async def main():
    """
    Пример запуска. Замени token, figi и импорты PatternDetector на свои.
    """
    import os
    # from config import TINKOFF_TOKEN
    # from patterns.detector import PatternDetector
    from data_loader.loader import InstrumentResolver

    TOKEN = os.getenv("TINKOFF_TOKEN", "")

    asset = ASSETS[0]
    print(f"⏳ Резолвим FIGI для {asset.ticker}...")
    await InstrumentResolver.resolve_all([asset])
    if not asset.figi:
        print("❌ Не удалось найти FIGI. Проверь токен или тикер.")
        return
    FIGI = asset.figi
    print(f"✅ FIGI: {FIGI}")

    # Загрузка данных
    from t_tech.invest import CandleInterval
    print("⏳ Загружаем данные...")
    df_10m = await fetch_candles(FIGI, TOKEN, CandleInterval.CANDLE_INTERVAL_10_MIN, days=60)
    df_1h  = await fetch_candles(FIGI, TOKEN, CandleInterval.CANDLE_INTERVAL_HOUR,   days=60)
    print(f"   10m: {len(df_10m)} свечей | 1h: {len(df_1h)} свечей")

    # Создаём стратегию
    """detector = PatternDetector(ScannerConfig())
    
    strategy = CandlePatternStrategy(
         detector=detector,
         atr_sl_mult=1.5,
         rr_ratio=2.0,       # <— R:R 1:2, было 1:1
         mfi_long_thr=45.0,  # <— было 35
         mfi_short_thr=55.0, # <— было 65
         use_htf_filter=True,
     )

    # Запускаем движок
    engine = BacktestEngine(strategy, initial_balance=100_000, position_size=800.0)
    portfolio = engine.run(df_10m, htf_df=df_1h)
    df_trades = engine.report()
    engine.visualize(df_10m)
    """
    """
    from strategies.wavetrend_strategy import WaveTrendStrategy

    strategy = WaveTrendStrategy(
    adx_threshold=22.0,
    use_di_filter=True,
    fixed_exit=False,      # выход по противоположному сигналу + ATR стоп
    atr_sl_mult=1.5,
    )
    engine = BacktestEngine(strategy, initial_balance=100_000)
    portfolio = engine.run(df_10m, htf_df=df_1h)
    engine.report()
    engine.visualize(df_10m)
    
    #print("\n✅ Шаблон готов. Раскомментируй нужные строки и подключи свои модули.")
    """
    
    
    from strategies.wavetrend_strategy_v2 import WaveTrendStrategyV2, PARAM_GRID

    strategy = WaveTrendStrategyV2(**PARAM_GRID['balanced'])

    engine = BacktestEngine(strategy, initial_balance=100_000)
    df_prep = strategy.prepare(df_10m)          # подготовить df заранее
    portfolio = engine.run(df_prep, htf_df=df_1h)
    df_trades = engine.report()
    engine.visualize(df_prep, df_trades)
    
    df_prep = strategy.prepare(df_10m)
    print(df_prep[['time','wt1','wt2','adx','_last_signal']].dropna().tail(20))
    print("Кроссоверов всего:", df_prep['_last_signal'].notna().sum())

if __name__ == "__main__":
    asyncio.run(main())