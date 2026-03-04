import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from moexalgo import Ticker
from datetime import datetime, timedelta
import requests
import time

# --- НАСТРОЙКИ ---
TOKEN = ""
CHAT_ID = ""

def create_screenshot(df, ticker, tf, pattern_name, signal_time):
    """
    Создает скриншот: 12 свечей ДО паттерна и ВСЕ свечи ПОСЛЕ.
    """
    try:
        idx = df.index[df['datetime'] == signal_time].tolist()[0]
    except IndexError:
        return None

    # Берем 12 свечей ДО и все доступные свечи ПОСЛЕ
    start_idx = max(0, idx - 20)
    end = min(len(df), idx + 20)
    plot_df = df.iloc[start_idx:end].copy() 
    
    fig = go.Figure(data=[go.Candlestick(
        x=plot_df['datetime'],
        open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'], close=plot_df['close'],
        name='Свечи'
    )])

    fig.add_trace(go.Scatter(
        x=plot_df['datetime'], y=plot_df['ema10'],
        mode='lines', line=dict(color='orange', width=1.5), name='EMA 10'
    ))

    fig.add_vline(x=signal_time, line_width=1, line_dash="dash", line_color="white")

    is_bullish = any(x in pattern_name.lower() for x in ['bull', 'hammer', 'piercing', 'inv_hammer'])
    y_pos = plot_df.loc[idx, 'low'] if is_bullish else plot_df.loc[idx, 'high']
    ay = 40 if is_bullish else -40
    color = "green" if is_bullish else "red"

    fig.add_annotation(
        x=signal_time, y=y_pos, text=pattern_name,
        showarrow=True, arrowhead=2, arrowcolor=color,
        ax=0, ay=ay, bgcolor=color, font=dict(color="white")
    )

    fig.update_layout(
        title=f"{ticker} {tf} | Разрез паттерна",
        template="plotly_dark", xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    file_path = f"alert_{ticker}_{tf}.png"
    fig.write_image(file_path, scale=2)
    return file_path

def analyze_morris_patterns(df):
    if len(df) < 20: 
        return [], None, None
    
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    curr = df.iloc[-2].copy()
    prev = df.iloc[-3].copy()
    
    c_close, ema, c_time = curr['close'], curr['ema10'], curr['datetime']
    c_open, c_high, c_low = curr['open'], curr['high'], curr['low']
    
    c_body_size = abs(c_close - c_open)
    c_range = c_high - c_low
    c_body_top, c_body_bottom = max(c_open, c_close), min(c_open, c_close)
    c_upper_shadow = c_high - c_body_top
    c_lower_shadow = c_body_bottom - c_low
    
    p_open, p_close = prev['open'], prev['close']
    p_body_top, p_body_bottom = max(p_open, p_close), min(p_open, p_close)
    p_body_size = abs(p_close - p_open)
    p_midpoint = (p_open + p_close) / 2

    signals = []

    # --- 1. ТРЕНД ВНИЗ (Ищем бычьи развороты под EMA) ---
    if c_close < ema:
        if c_lower_shadow >= (c_body_size * 2) and c_upper_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Hammer (Молот)")
            
        if c_upper_shadow >= (c_body_size * 2) and c_lower_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Inverted Hammer (Перевернутый молот)")
            
        if c_close > c_open and p_close < p_open and c_body_top >= p_body_top and c_body_bottom <= p_body_bottom:
            signals.append("Bullish Engulfing (Бычье поглощение)")
            
        if p_close < p_open and c_body_top <= p_body_top and c_body_bottom >= p_body_bottom and p_body_size > c_body_size:
            signals.append("Bullish Harami (Бычье Харами)")
            
        if p_close < p_open and c_close > c_open and c_open < p_close and c_close > p_midpoint:
            signals.append("Piercing Line (Просвет в облаках)")

    # --- 2. ТРЕНД ВВЕРХ (Ищем медвежьи развороты над EMA) ---
    elif c_close > ema:
        if c_lower_shadow >= (c_body_size * 2) and c_upper_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Hanging Man (Висельник)")
            
        if c_upper_shadow >= (c_body_size * 2) and c_lower_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Shooting Star (Падающая звезда)")
            
        if c_close < c_open and p_close > p_open and c_body_top >= p_body_top and c_body_bottom <= p_body_bottom:
            signals.append("Bearish Engulfing (Медвежье поглощение)")
            
        if p_close > p_open and c_body_top <= p_body_top and c_body_bottom >= p_body_bottom and p_body_size > c_body_size:
            signals.append("Bearish Harami (Медвежье Харами)")
            
        if p_close > p_open and c_close < c_open and c_open > p_close and c_close < p_midpoint:
            signals.append("Dark Cloud Cover (Завеса из темных облаков)")

    # --- 3. НЕЙТРАЛЬНЫЕ ---
    if c_body_size <= (c_range * 0.1) and c_range > 0:
        signals.append("Doji (Доджи)")

    return signals, c_time, c_close

def analyze_morris_patterns_at_index(df, idx):
    """
    Анализирует паттерны на конкретном индексе 'idx'. 
    Добавлена логика длинных/коротких дней и строгие разрывы (гэпы) по Моррису.
    """
    # Нам нужно как минимум 10 свечей истории для расчета среднего тела
    if idx < 10: 
        return []
    
    curr = df.iloc[idx].copy()
    prev = df.iloc[idx-1].copy()
    
    c_close, ema = curr['close'], curr['ema10']
    c_open, c_high, c_low = curr['open'], curr['high'], curr['low']
    c_body_size = abs(c_close - c_open)
    c_range = c_high - c_low
    c_body_top, c_body_bottom = max(c_open, c_close), min(c_open, c_close)
    c_upper_shadow = c_high - c_body_top
    c_lower_shadow = c_body_bottom - c_low
    
    # Цвета текущей свечи
    c_is_white = c_close > c_open
    c_is_black = c_close < c_open
    
    p_open, p_close = prev['open'], prev['close']
    p_high, p_low = prev['high'], prev['low']
    p_body_top, p_body_bottom = max(p_open, p_close), min(p_open, p_close)
    p_body_size = abs(p_close - p_open)
    p_midpoint = (p_open + p_close) / 2
    
    # Цвета предыдущей свечи
    p_is_white = p_close > p_open
    p_is_black = p_close < p_open

    # --- ЛОГИКА ДЛИННЫХ И КОРОТКИХ ДНЕЙ ПО МОРРИСУ ---
    # Считаем средний размер тела за последние 10 дней (до текущей свечи)
    past_bodies = abs(df['close'].iloc[idx-10:idx] - df['open'].iloc[idx-10:idx])
    avg_body = past_bodies.mean()
    
    # "Длинный" день: тело на 30% больше среднего
    p_is_long = p_body_size > (avg_body * 1.3)
    # "Короткий" день: тело меньше среднего
    c_is_short = c_body_size < avg_body 

    signals = []

    # === 1. ТРЕНД ВНИЗ (Ищем БЫЧЬИ развороты под EMA) ===
    if c_close < ema:
        # Молот (Hammer)
        if c_lower_shadow >= (c_body_size * 2) and c_upper_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Hammer (Молот)")
            
        # Перевернутый молот (Inverted Hammer)
        if c_upper_shadow >= (c_body_size * 2) and c_lower_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Inverted Hammer (Перевернутый молот)")
            
        # Бычье поглощение (Bullish Engulfing)
        if c_is_white and p_is_black and c_body_top >= p_body_top and c_body_bottom <= p_body_bottom and c_body_size > p_body_size:
            signals.append("Bullish Engulfing (Бычье поглощение)")
            
        # Бычье Харами (Bullish Harami)
        # 1. Предшествует тренд (c_close < ema)
        # 2. Первый день длинный и отражает тренд (черный)
        # 3. Второй день короткий, цвет противоположный (белый)
        # 4. Тело полностью внутри первого
        if p_is_long and c_is_short and p_is_black and c_is_white:
            if c_body_top <= p_body_top and c_body_bottom >= p_body_bottom and p_body_size > c_body_size:
                signals.append("Bullish Harami (Бычье Харами)")
                
        # Пронизывающая линия (Piercing Line)
        # 1. Первый день длинный черный
        # 2. Второй открывается НИЖЕ МИНИМУМА первого
        # 3. Второй закрывается выше середины первого
        if p_is_long and p_is_black and c_is_white:
            if c_open <= p_close and c_close > p_midpoint and c_close <= p_body_top:
                signals.append("Piercing Line (Просвет в облаках)")

    # === 2. ТРЕНД ВВЕРХ (Ищем МЕДВЕЖЬИ развороты над EMA) ===
    elif c_close > ema:
        # Висельник (Hanging Man)
        if c_lower_shadow >= (c_body_size * 2) and c_upper_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Hanging Man (Висельник)")
            
        # Падающая звезда (Shooting Star)
        if c_upper_shadow >= (c_body_size * 2) and c_lower_shadow <= (c_range * 0.1) and c_body_size > 0:
            signals.append("Shooting Star (Падающая звезда)")
            
        # Медвежье поглощение (Bearish Engulfing)
        if c_is_black and p_is_white and c_body_top >= p_body_top and c_body_bottom <= p_body_bottom and c_body_size > p_body_size:
            signals.append("Bearish Engulfing (Медвежье поглощение)")
            
        # Медвежье Харами (Bearish Harami)
        if p_is_long and c_is_short and p_is_white and c_is_black:
            if c_body_top <= p_body_top and c_body_bottom >= p_body_bottom and p_body_size > c_body_size:
                signals.append("Bearish Harami (Медвежье Харами)")
                
        # Темные облака (Dark Cloud Cover)
        # 1. Первый день длинный белый
        # 2. Второй открывается ВЫШЕ МАКСИМУМА первого
        # 3. Второй закрывается ниже середины первого
        if p_is_long and p_is_white and c_is_black:
            if c_open >= p_close and c_close < p_midpoint and c_close >= p_body_bottom:
                signals.append("Dark Cloud Cover (Темные облака)")

    # === 3. НЕЙТРАЛЬНЫЕ ===
    if c_body_size <= (c_range * 0.1) and c_range > 0:
        signals.append("Doji (Доджи)")

    return signals

# --- БЕЗОПАСНАЯ ЗАГРУЗКА ДАННЫХ (RETRY МЕХАНИЗМ) ---
def get_safe_candles(ticker_name, tf, days_back=4, retries=3):
    start_dt = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_dt = datetime.now().strftime('%Y-%m-%d')
    
    for attempt in range(retries):
        try:
            t = Ticker(ticker_name)
            data = t.candles(start=start_dt, end=end_dt, period=tf)
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            err_msg = str(e)
            if "isoformat" in err_msg or "NoneType" in err_msg:
                print(f"[!] Тикер {ticker_name} неактивен или данные недоступны.")
                return pd.DataFrame()
            elif "SSL" in err_msg or "EOF" in err_msg or "Connection" in err_msg:
                print(f"[*] Сбой сети при загрузке {ticker_name}. Попытка {attempt + 1}/{retries}...")
                time.sleep(2)
            else:
                print(f"[*] Неизвестная ошибка загрузки {ticker_name}: {err_msg}")
                return pd.DataFrame()
                
    return pd.DataFrame() 

def run_scanner():
    TICKERS = ['SBER'] 
    TIMEFRAMES = ['15min']
    last_alerts = {}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Сканер запущен. Мониторинг: {TICKERS}")

    while True:
        for ticker in TICKERS:
            for tf in TIMEFRAMES:
                df = get_safe_candles(ticker, tf)
                
                if df.empty or 'begin' not in df.columns: 
                    continue
                
                try:
                    df['begin'] = pd.to_datetime(df['begin'])
                    df.rename(columns={'begin': 'datetime'}, inplace=True)
                    
                    found_patterns, candle_time, last_price = analyze_morris_patterns(df)
                    
                    if not found_patterns or candle_time is None:
                        continue

                    for pattern in found_patterns:
                        alert_key = (ticker, tf, pattern)
                        if last_alerts.get(alert_key) != candle_time:
                            
                            img_path = create_screenshot(df, ticker, tf, pattern, candle_time)
                            
                            if img_path:
                                text = (f"🎯 *{pattern}*\n"
                                        f"📊 `{ticker}` | `{tf}`\n"
                                        f"💰 Цена: `{last_price}`\n"
                                        f"⏰ Свеча: {candle_time.strftime('%H:%M')}")
                                
                                url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
                                
                                # Отправляем фото
                                with open(img_path, 'rb') as f:
                                    requests.post(url, data={'chat_id': CHAT_ID, 'caption': text, 'parse_mode': 'Markdown'}, files={'photo': f})
                                
                                # Удаляем файл после отправки, чтобы не засорять диск
                                try:
                                    os.remove(img_path)
                                except OSError as e:
                                    print(f"Ошибка при удалении {img_path}: {e}")
                                
                                last_alerts[alert_key] = candle_time
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Отправлен сигнал: {ticker} {tf} {pattern}")

                except Exception as e:
                    print(f"Ошибка логики анализа {ticker} {tf}: {e}")
        
        # Ждем перед следующим циклом
        time.sleep(60)


def test_run(days_back=3):
    """
    Проходит по всей истории за последние дни и шлет скрины найденных паттернов.
    """
    TICKERS = ['SBER'] # Тестовый список
    TIMEFRAMES = ['15min']
    
    print(f"--- ЗАПУСК ТЕСТА ЗА {days_back} ДНЯ ---")

    for ticker in TICKERS:
        for tf in TIMEFRAMES:
            df = get_safe_candles(ticker, tf, days_back=days_back)
            if df.empty: continue
            
            df['begin'] = pd.to_datetime(df['begin'])
            df.rename(columns={'begin': 'datetime'}, inplace=True)
            df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()

            # Идем по истории (пропуская первые 10 свечей для EMA)
            for i in range(10, len(df)):
                found_patterns = analyze_morris_patterns_at_index(df, i)
                
                if found_patterns:
                    candle_time = df.loc[i, 'datetime']
                    last_price = df.loc[i, 'close']
                    
                    for pattern in found_patterns:
                        print(f"Найдено в истории: {ticker} {tf} {pattern} в {candle_time}")
                        
                        img_path = create_screenshot(df, ticker, tf, pattern, candle_time)
                        
                        if img_path:
                            text = (f"🧪 *ТЕСТОВЫЙ СИГНАЛ*\n"
                                    f"🎯 *{pattern}*\n"
                                    f"📊 `{ticker}` | `{tf}`\n"
                                    f"💰 Цена: `{last_price}`\n"
                                    f"⏰ Время: {candle_time.strftime('%d.%m %H:%M')}")
                            
                            url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
                            with open(img_path, 'rb') as f:
                                requests.post(url, data={'chat_id': CHAT_ID, 'caption': text, 'parse_mode': 'Markdown'}, files={'photo': f})
                            
                            if os.path.exists(img_path): os.remove(img_path)
                            time.sleep(1) # Защита от спам-фильтра Telegram

    print("--- ТЕСТ ЗАВЕРШЕН ---")

# --- БЭКТЕСТ ЛОКАЛЬНОЙ ЛОГИКИ (ПРОХОД ПО ИСТОРИИ И ОТПРАВКА СИГНАЛОВ) ---

def visual_backtest(ticker='SBER', tf='15min', days_back=5):
    """
    Сканирует историю, собирает все сигналы и выводит их на один интерактивный график.
    Никакого спама в Telegram — только визуальный анализ.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Загрузка данных для визуального теста ({days_back} дней)...")
    
    df = get_safe_candles(ticker, tf, days_back=days_back)
    
    if df.empty:
        print("Данные не получены.")
        return

    # Подготовка данных
    if 'begin' in df.columns:
        df['datetime'] = pd.to_datetime(df['begin'])
    
    # Обязательная сортировка по времени
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Расчет EMA 10 для всего датафрейма
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()

    # Списки для хранения координат маркеров
    bull_x, bull_y, bull_text = [], [], []
    bear_x, bear_y, bear_text = [], [], []
    doji_x, doji_y, doji_text = [], [], []

    print("Анализ паттернов...")
    # Пробегаем по всей истории
    for i in range(10, len(df)):
        patterns = analyze_morris_patterns_at_index(df, i)
        
        if patterns:
            dt = df.loc[i, 'datetime']
            high = df.loc[i, 'high']
            low = df.loc[i, 'low']
            
            for p in patterns:
                # Распределяем паттерны по группам для правильной отрисовки
                if 'Doji' in p:
                    doji_x.append(dt)
                    doji_y.append(high + (high * 0.001)) # Чуть выше свечи
                    doji_text.append(p)
                elif any(x in p.lower() for x in ['bull', 'hammer', 'piercing', 'inv_hammer']):
                    bull_x.append(dt)
                    bull_y.append(low - (low * 0.002)) # Под свечой
                    bull_text.append(p)
                else:
                    bear_x.append(dt)
                    bear_y.append(high + (high * 0.002)) # Над свечой
                    bear_text.append(p)

    print(f"Найдено бычьих: {len(bull_x)}, медвежьих: {len(bear_x)}, доджи: {len(doji_x)}")
    print("Отрисовка графика...")

    # Создаем основной график
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Свечи'
    )])

    # Добавляем линию EMA
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['ema10'],
        mode='lines', line=dict(color='orange', width=1.5), name='EMA 10'
    ))

    # Добавляем бычьи сигналы (Зеленые треугольники снизу)
    if bull_x:
        fig.add_trace(go.Scatter(
            x=bull_x, y=bull_y, mode='markers+text',
            marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(width=1, color='black')),
            text=bull_text, textposition="bottom center", textfont=dict(color='lime', size=10),
            name='Бычьи паттерны'
        ))

    # Добавляем медвежьи сигналы (Красные треугольники сверху)
    if bear_x:
        fig.add_trace(go.Scatter(
            x=bear_x, y=bear_y, mode='markers+text',
            marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='black')),
            text=bear_text, textposition="top center", textfont=dict(color='red', size=10),
            name='Медвежьи паттерны'
        ))

    # Добавляем доджи (Серые крестики)
    if doji_x:
        fig.add_trace(go.Scatter(
            x=doji_x, y=doji_y, mode='markers+text',
            marker=dict(symbol='x', size=8, color='gray'),
            text=doji_text, textposition="top center", textfont=dict(color='gray', size=9),
            name='Доджи'
        ))

    fig.update_layout(
        title=f"Полный Бэктест: {ticker} {tf} | Дней: {days_back}",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=900 # Делаем график высоким для удобства
    )
    
    # Открываем график в браузере
    fig.show()


if __name__ == "__main__":
    # Запускаем визуальный тест за 5 дней
    visual_backtest(ticker='SiH6', tf='15min', days_back=14)

#if __name__ == "__main__":
#    test_run(days_back=2)
