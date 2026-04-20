# verify_data.py
import asyncio
import pandas as pd
from config import ASSETS, TIMEFRAMES, ScannerConfig
from data_loader.loader import InstrumentResolver, StreamLoader
from core.asset_worker import AssetWorker

async def verify():
    # 1. Резолвим FIGI
    await InstrumentResolver.resolve_all(ASSETS)
    asset = ASSETS[5] # Берем первый актив для теста (например, Si)
    
    # 2. Создаем временный воркер
    worker = AssetWorker(asset, None, None, ScannerConfig(), None)
    workers = {asset.figi: worker}
    
    loader = StreamLoader(assets=[asset], workers=workers, timeframes=TIMEFRAMES)
    
    # 3. Загружаем историю
    print(f"--- Загрузка истории для {asset.ticker} ---")
    await loader.prefill_history()
    
    # 4. Проверяем каждый таймфрейм
    for tf in TIMEFRAMES:
        df = worker._aggregator.to_dataframe(tf)
        if df.empty:
            print(f"❌ TF {tf}m: ДАННЫЕ ОТСУТСТВУЮТ")
            continue
            
        last_time = df.index[-1]
        count = len(df)
        print(f"✅ TF {tf:3}m: {count:4} свечей | Последняя: {last_time} | DF(последние 10 свечей): {df.tail(10)}")
        
        # Проверка на пропуски (только для 1м, если есть в списке)
        if tf == 1:
            diffs = df.index.to_series().diff().dropna()
            gaps = diffs[diffs > pd.Timedelta(minutes=1)]
            if not gaps.empty:
                print(f"⚠️ Найдено {len(gaps)} пропусков в данных 1m!")

if __name__ == "__main__":
    asyncio.run(verify())