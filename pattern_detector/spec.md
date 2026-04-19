# Pattern Detector Bot — Спецификация

## Назначение
Торговый бот реального времени на базе T-Invest gRPC API.
Мониторит фьючи MOEX, обнаруживает свечные паттерны и отправляет
сигналы в Telegram-супергруппу каждого актива.

## Архитектура

```
main.py
  │
  ├── InstrumentResolver      — резолвит FIGI фьючей при старте
  │
  ├── StreamLoader            — gRPC MarketDataStream (1min, waiting_close=True)
  │     └── asyncio.Queue     — одна очередь на актив
  │
  └── AssetWorker × N         — по одному на каждый актив
        │
        ├── CandleAggregator  — агрегирует 1min → [10,15,30,60,120,180,240]m
        │     └── to_dataframe() + ema10 + indicator
        │
        ├── PatternDetector   — свечные паттерны по EMA10 + индикатор
        │
        ├── ChartVisualizer   — Plotly Dark график + аннотация паттерна
        │
        └── TelegramRouter    — отправка фото/текста в нужную супергруппу
```

## Активы и Telegram-группы

| Тикер | Telegram chat_id  |
|-------|-------------------|
| SiM   | -3916417055       |
| KC    | -3716674818       |
| BR    | -3750564099       |
| CC    | -3944653501       |
| SV    | -3994847530       |
| GD    | -3949836612       |

## Таймфреймы
10, 15, 30, 60, 120, 180, 240 минут.
Все получаются агрегацией из 1-минутного gRPC стрима.

## Паттерны (PatternDetector)

### Бычьи (c.close < ema10)
- Hammer (Молот)
- Inverted Hammer (Перевернутый молот)
- Bullish Engulfing (Бычье поглощение)
- Bullish Harami (Бычье Харами)
- Bullish Harami Cross (Бычий Крест Харами)
- Piercing Line (Просвет в облаках)
- Morning Star (Утренняя звезда)
- Three White Soldiers (Три белых солдата)

### Медвежьи (c.close > ema10)
- Hanging Man (Висельник)
- Shooting Star (Падающая звезда)
- Bearish Engulfing (Медвежье поглощение)
- Bearish Harami (Медвежье Харами)
- Bearish Harami Cross (Медвежий Крест Харами)
- Dark Cloud Cover (Тёмные облака)
- Evening Star (Вечерняя звезда)
- Three Black Crows (Три чёрные вороны)

## Индикаторы (indicators/base.py)

| Класс            | Использование                             |
|------------------|-------------------------------------------|
| `NoIndicator`    | Без фильтра (по умолчанию)               |
| `RSIIndicator`   | `RSIIndicator(period=14, bullish_below=40, bearish_above=60)` |
| `MACDIndicator`  | `MACDIndicator(fast=12, slow=26, signal=9)` |

Индикатор меняется в `config.py` → `ScannerConfig`.

## Переменные окружения (.env)

```
TINKOFF_TOKEN=...        # T-Invest токен (sandbox или production)
TELEGRAM_BOT_TOKEN=...   # токен @BotFather
```

## Установка и запуск

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env      # заполнить токены
python main.py
```

## Особенности

- **FIGI фьючей** резолвятся автоматически при каждом старте —
  после экспирации контракта достаточно перезапустить бота.
- **Переподключение** при обрыве gRPC стрима — автоматически
  через `STREAM_RECONNECT_DELAY` секунд.
- **Параллелизм**: все TF одного актива обрабатываются через
  `asyncio.gather()`, CPU-операции (pandas/plotly) — в `asyncio.to_thread()`.
- **Graceful shutdown**: SIGINT / SIGTERM корректно отменяют все задачи.

## Темы внутри супергрупп (Forum Topics)

Каждая супергруппа должна иметь включённый режим **Topics** (Forum).
Бот создаёт темы автоматически при старте через `createForumTopic`.

| Таймфрейм | Название темы | Цвет иконки |
|-----------|---------------|-------------|
| 10 min    | 10 min        | 🔵 синий    |
| 15 min    | 15 min        | 🟢 зелёный  |
| 30 min    | 30 min        | 🟡 жёлтый   |
| 1 hour    | 1 hour        | 🟣 фиолетовый |
| 2 hours   | 2 hours       | 🩷 розовый  |
| 3 hours   | 3 hours       | 🔴 красный  |
| 4 hours   | 4 hours       | ⚫ серый    |

### Требования к боту в каждой супергруппе
1. Бот добавлен как **администратор**
2. Выдано право **"Manage Topics"** (`can_manage_topics`)
3. В настройках группы включён режим **Topics**

### Логика TopicManager
- При старте получает список существующих тем (`getForumTopics`)
- Создаёт только недостающие (`createForumTopic`) — повторный запуск безопасен
- Хранит маппинг `chat_id → {tf_minutes → message_thread_id}` в памяти
- Сигналы отправляются с `message_thread_id` → попадают в нужную тему
