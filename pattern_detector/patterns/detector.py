"""
PatternDetector — оригинальная логика без изменений.
Импорты адаптированы под новую структуру проекта.
"""
from typing import List
import pandas as pd
import numpy as np

from config import ScannerConfig


class PatternDetector:
    def __init__(self, config: ScannerConfig):
        self.config = config
        self.avg_body = 0

    def _update_context(self, df: pd.DataFrame, idx: int):
        past_bodies = (df['close'].iloc[idx-10:idx] - df['open'].iloc[idx-10:idx]).abs()
        self.avg_body = past_bodies.mean()

    def is_long(self, body_size: float) -> bool:
        return body_size > (self.avg_body * self.config.long_body_coeff)

    def is_short(self, body_size: float) -> bool:
        return body_size < (self.avg_body * self.config.short_body_coeff)

    def is_dodji(self, candle) -> bool:
        full_range = candle.high - candle.low
        body       = abs(candle.close - candle.open)
        if full_range == 0:
            return False
        return body <= (full_range * 0.15)

    def get_pattern_at_index(self, df: pd.DataFrame, idx: int) -> List[str]:
        if idx < 12:
            return []
        self._update_context(df, idx)

        c  = df.iloc[idx]
        p  = df.iloc[idx - 1]
        pp = df.iloc[idx - 2]

        c_body         = abs(c.close - c.open)
        c_range        = c.high - c.low
        c_top          = max(c.open, c.close)
        c_bottom       = min(c.open, c.close)
        c_upper_shadow = c.high - c_top
        c_lower_shadow = c_bottom - c.low

        p_body   = abs(p.close - p.open)
        p_top    = max(p.open, p.close)
        p_bottom = min(p.open, p.close)
        p_mid    = (p.open + p.close) / 2

        pp_body   = abs(pp.close - pp.open)
        pp_mid    = (pp.open + pp.close) / 2
        pp_top    = max(pp.open, pp.close)
        pp_bottom = min(pp.open, pp.close)

        c_is_white, c_is_black   = c.close > c.open, c.close < c.open
        p_is_white, p_is_black   = p.close > p.open, p.close < p.open
        pp_is_white, pp_is_black = pp.close > pp.open, pp.close < pp.open

        signals   = []
        ema       = c.ema10
        ind_col   = self.config.indicator.column_name
        ind_value = c[ind_col] if ind_col in df.columns else None

        def bullish_confirmed() -> bool:
            if ind_value is None or np.isnan(ind_value):
                return True
            return self.config.indicator.confirms_bullish(ind_value)

        def bearish_confirmed() -> bool:
            if ind_value is None or np.isnan(ind_value):
                return True
            return self.config.indicator.confirms_bearish(ind_value)

        # === ТРЕНД ВНИЗ — бычьи развороты ===
        if c.close < ema:

            if bullish_confirmed():

                if c_lower_shadow >= (c_body * 2) and c_upper_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Hammer (Молот)")

                if c_upper_shadow >= (c_body * 2) and c_lower_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Inverted Hammer (Перевернутый молот)")

                if c_is_white and p_is_black and c_top >= p_top and c_bottom <= p_bottom and p_body <= (c_body * 0.8):
                    signals.append("Bullish Engulfing (Бычье поглощение)")

                if c_is_white and p_is_black and c_body <= (p_body * 0.8) and c_top <= p_top and c_bottom >= p_bottom:
                    signals.append("Bullish Harami (Бычье Харами)")

                if self.is_dodji(c) and p_is_black and p_body > self.avg_body and c_top <= p_top and c_bottom >= p_bottom:
                    signals.append("Bullish Harami Cross (Бычий Крест Харами)")

                if self.is_long(p_body) and p_is_black and c_is_white:
                    if c.open < p.close and c.close > p_mid:
                        signals.append("Piercing Line (Просвет в облаках)")

                if pp_is_black and self.is_long(pp_body) and self.is_short(p_body):
                    if p_top <= pp_bottom and c_is_white and c.close >= pp_mid:
                        signals.append("Morning Star (Утренняя звезда)")

                if all([pp_is_white, p_is_white, c_is_white]):
                    if all([self.is_long(b) for b in [pp_body, p_body, c_body]]):
                        signals.append("Three White Soldiers (Три белых солдата)")

        # === ТРЕНД ВВЕРХ — медвежьи развороты ===
        elif c.close > ema:

            if bearish_confirmed():

                if c_lower_shadow >= (c_body * 2) and c_upper_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Hanging Man (Висельник)")

                if c_upper_shadow >= (c_body * 2) and c_lower_shadow <= (c_range * 0.1) and c_body > 0:
                    signals.append("Shooting Star (Падающая звезда)")

                if c_is_black and p_is_white and p_body <= (c_body * 0.8) and c_top >= p_top and c_bottom <= p_bottom:
                    signals.append("Bearish Engulfing (Медвежье поглощение)")

                if c_is_black and p_is_white and c_body <= (p_body * 0.8) and c_top <= p_top and c_bottom >= p_bottom:
                    signals.append("Bearish Harami (Медвежье Харами)")

                if self.is_dodji(c) and p_is_white and p_body > self.avg_body and c_top <= p_top and c_bottom >= p_bottom:
                    signals.append("Bearish Harami Cross (Медвежий Крест Харами)")

                if self.is_long(p_body) and p_is_white and c_is_black:
                    if c.open > p.close and c.close < p_mid:
                        signals.append("Dark Cloud Cover (Темные облака)")

                if pp_is_white and self.is_long(pp_body) and self.is_short(p_body):
                    if p_bottom >= pp_top and c_is_black and c.close < pp_mid:
                        signals.append("Evening Star (Вечерняя звезда)")

                if all([pp_is_black, p_is_black, c_is_black]):
                    if all([self.is_long(b) for b in [pp_body, p_body, c_body]]):
                        signals.append("Three Black Crows (Три черные вороны)")

        return signals
