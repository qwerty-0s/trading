from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np


class BaseIndicator(ABC):
    """Абстрактный базовый класс для всех индикаторов-фильтров."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Вычислить значения индикатора по всему датафрейму."""
        pass

    @abstractmethod
    def confirms_bullish(self, value: float) -> bool:
        """Подтверждает ли значение индикатора бычий сигнал?"""
        pass

    @abstractmethod
    def confirms_bearish(self, value: float) -> bool:
        """Подтверждает ли значение индикатора медвежий сигнал?"""
        pass

    @property
    @abstractmethod
    def column_name(self) -> str:
        """Имя колонки, которую индикатор добавляет в df."""
        pass

    @property
    @abstractmethod
    def plot_label(self) -> str:
        """Подпись для оси Y на графике."""
        pass

    def get_level_lines(self) -> List[dict]:
        """
        Горизонтальные уровни для отрисовки на графике.
        Возвращает список dict: {"value": float, "color": str, "dash": str}
        """
        return []


class NoIndicator(BaseIndicator):
    """
    Индикатор-заглушка — отключает фильтрацию полностью.
    Используется для чистого теста паттернов без подтверждения.
    На графике ничего не рисует.
    """

    @property
    def column_name(self) -> str:
        return "_no_indicator"

    @property
    def plot_label(self) -> str:
        return ""

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([np.nan] * len(df), index=df.index)

    def confirms_bullish(self, value: float) -> bool:
        return True

    def confirms_bearish(self, value: float) -> bool:
        return True
