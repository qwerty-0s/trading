# test_logic_injection.py
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.candle_aggregator import Candle
from core.asset_worker import AssetWorker
from config import ASSETS, ScannerConfig

async def test_pattern_below_ema():
    asset = ASSETS[0]
    # Создаем мок-воркер без реальных соединений
    worker = AssetWorker(asset, None, None, ScannerConfig(), None)
    ticker = asset.ticker
    
    base_time = datetime.now(timezone.utc) - timedelta(hours=5)
    
    # 1. Формируем "медвежий тренд", чтобы EMA10 была выше цены
    # Цена падает со 150 до 100
    history = []
    for i in range(20):
        price = 150 - i * 2
        history.append(Candle(
            ticker=ticker, timeframe=1,
            open_time=base_time + timedelta(minutes=i),
            open=price + 1, high=price + 2, low=price - 1, close=price,
            volume=100
        ))
        
    # 2. Добавляем паттерн "Бычье поглощение" на лоях (около 100)
    # Свеча А: маленькая черная
    history.append(Candle(
        ticker=ticker, timeframe=1,
        open_time=base_time + timedelta(minutes=20),
        open=102, high=103, low=99, close=100, volume=100
    ))
    # Свеча Б: большая белая, поглощающая А
    history.append(Candle(
        ticker=ticker, timeframe=1,
        open_time=base_time + timedelta(minutes=21),
        open=99, high=115, low=98, close=110, volume=500
    ))

    print(f"🧪 Тестируем {ticker}. Инъекция 22 свечей...")
    
    for c in history:
        # Пропускаем через агрегатор воркера
        worker._aggregator.push(c)
        
    # 3. Проверяем результат для таймфрейма 10m (или любого активного)
    tf = 10
    df = worker._aggregator.to_dataframe(tf)
    df = worker.scanner_config.indicator.compute(df)
    
    last_idx = len(df) - 1
    patterns = worker.detector.get_pattern_at_index(df, last_idx)
    
    ema = df['ema10'].iloc[last_idx]
    close = df['close'].iloc[last_idx]
    
    print(f"\nРезультаты для TF {tf}m:")
    print(f"Последняя цена: {close} | EMA10: {ema:.2f}")
    print(f"Обнаруженные паттерны: {patterns}")
    
    if close < ema:
        print("📉 Цена ниже EMA10 — сигнал должен быть отфильтрован (если так задумано).")
    else:
        print("📈 Цена выше EMA10.")

if __name__ == "__main__":
    asyncio.run(test_pattern_below_ema())