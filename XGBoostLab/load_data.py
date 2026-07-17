import os
import asyncio
import pandas as pd
from datetime import datetime
from typing import Tuple

# Импортируем CandleInterval из T-Invest API (с fallback на старый пакет tinkoff-investments)
try:
    from t_tech.invest import CandleInterval
except ImportError:
    from tinkoff.invest import CandleInterval

# Импортируем готовую функцию загрузки из вашего backtest_engine
from backtest_engine import fetch_candles


async def get_tbank_market_data(
    figi: str,
    token: str,
    days: int = 180,
    cache_dir: str = "data",
    force_reload: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загрузка рабочих (10m/15m) и старших (1h) свечей из Т-Инвестиций
    с автоматическим кэшированием в Parquet.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_10m = os.path.join(cache_dir, "df_10m.parquet")
    file_1h = os.path.join(cache_dir, "df_1h.parquet")

    # 1. Если кэш есть и не затребовано принудительное обновление — читаем из файла
    if not force_reload and os.path.exists(file_10m) and os.path.exists(file_1h):
        print(f"📦 Данные загружены из локального кэша ({cache_dir}/)")
        df_10m = pd.read_parquet(file_10m)
        df_1h = pd.read_parquet(file_1h)
        return df_10m, df_1h

    # 2. Иначе скачиваем из Т-Инвестиций через fetch_candles
    print(f"📡 Загрузка свежих данных из Т-Инвестиций за последние {days} дней...")
    
    # Загружаем рабочий (10m) и старший (1h) таймфреймы
    df_10m = await fetch_candles(
        figi=figi, 
        token=token, 
        interval=CandleInterval.CANDLE_INTERVAL_10_MIN, 
        days=days
    )
    
    df_1h = await fetch_candles(
        figi=figi, 
        token=token, 
        interval=CandleInterval.CANDLE_INTERVAL_HOUR, 
        days=days
    )

    if df_10m.empty or df_1h.empty:
        raise ValueError("❌ Ошибка: T-Investments вернул пустой датасет свечей!")

    # 3. Приведение временных меток к единому формату (убираем timezone для Pandas/XGBoost)
    for df in [df_10m, df_1h]:
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

    # 4. Сохраняем в Parquet кэш
    df_10m.to_parquet(file_10m, index=False)
    df_1h.to_parquet(file_1h, index=False)
    print(f"✅ Данные успешно кешированы (10m: {len(df_10m)} баров | 1h: {len(df_1h)} баров)")

    return df_10m, df_1h


def load_data_sync(figi: str, token: str, days: int = 180, force_reload: bool = False):
    """Синхронная обертка для удобного вызова в обычном скрипте."""
    return asyncio.run(get_tbank_market_data(
        figi=figi,
        token=token,
        days=days,
        force_reload=force_reload
    ))