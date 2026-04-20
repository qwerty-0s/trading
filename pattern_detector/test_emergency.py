import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import ASSETS, ScannerConfig, TELEGRAM_BOT_TOKEN
from core.asset_worker import AssetWorker
from core.candle_aggregator import Candle
from core.topic_manager import TopicManager
from visualization.visualisation import ChartVisualizer
from sending.telegram_router import TelegramRouter

async def test_logic_with_filters():
    print("🧪 Тестирование логики: Поглощение + Тренд + EMA10")
    
    asset = ASSETS[0]
    tf = 10
    scanner_cfg = ScannerConfig(long_body_coeff=1.2) # Немного снизим порог для теста
    
    router = TelegramRouter(TELEGRAM_BOT_TOKEN)
    visualizer = ChartVisualizer()
    topic_manager = TopicManager(TELEGRAM_BOT_TOKEN, [tf])
    await topic_manager.setup([asset])
    
    worker = AssetWorker(asset, router, visualizer, scanner_cfg, topic_manager, timeframes=[tf])

    # Генерируем данные так, чтобы цена была под EMA10
    # Начинаем с высоких цен, которые быстро падают
    now = datetime.now(timezone.utc)
    history_candles = []
    
    # 1. Заполняем 13 падающих свечей (создаем медвежий тренд)
    for i in range(13):
        price = 150 - i  # Цена падает со 150 до 137
        history_candles.append(Candle(
            ticker=asset.ticker, timeframe=tf,
            open_time=now - timedelta(minutes=(15-i)*tf),
            open=price + 0.5, high=price + 1, low=price - 1, close=price, volume=100
        ))
    
    # 2. Пред-последняя свеча: Маленькая красная (индекс 13)
    history_candles.append(Candle(
        ticker=asset.ticker, timeframe=tf,
        open_time=now - timedelta(minutes=tf),
        open=136.5, high=137.0, low=135.5, close=136.0, volume=100
    ))
    
    # 3. Сигнальная свеча: Большая зеленая (индекс 14)
    # Она поглощает 136.5-136.0, и её close (139) должен быть ниже EMA10
    # (EMA10 при таком падении будет в районе 142-143)
    history_candles.append(Candle(
        ticker=asset.ticker, timeframe=tf,
        open_time=now,
        open=135.8, high=140.0, low=135.0, close=139.5, volume=500
    ))

    # Инъекция и расчет
    worker._aggregator._history[tf] = history_candles
    df = worker._aggregator.to_dataframe(tf)
    
    # Проверка EMA10 в тесте
    last_close = df['close'].iloc[-1]
    last_ema = df['ema10'].iloc[-1]
    print(f"DEBUG: Close={last_close:.2f}, EMA10={last_ema:.2f}")
    
    if last_close >= last_ema:
        print("⚠️ Предупреждение: Свеча закрылась ВЫШЕ EMA10, паттерн может быть отфильтрован.")

    # Детекция (через словарь _detectors)
    detector = worker._detectors[tf]
    patterns = detector.get_pattern_at_index(df, len(df) - 1)
    
    print(f"📊 Найдено паттернов: {patterns}")
    
    if patterns:
        print("✅ Успех! Паттерн обнаружен с учетом фильтров.")
        # Для проверки отправки:
        thread_id = topic_manager.get_thread_id(asset.tg_chat_id, tf)
        await worker._send_pattern(df, history_candles[-1], f"{tf}min", patterns[0], now, scanner_cfg.indicator, thread_id)
    else:
        print("❌ Паттерн не найден. Проверь условия is_long или расчет EMA.")

if __name__ == "__main__":
    asyncio.run(test_logic_with_filters())