# test_signal.py
import asyncio, json, os
from telegram import Bot
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = "topics_cache.json"

async def main():
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])

    with open(CACHE_FILE) as f:
        cache = json.load(f)

    tf_names = {
        "10": "10 min", "15": "15 min", "30": "30 min",
        "60": "1 hour", "120": "2 hours", "180": "3 hours", "240": "4 hours"
    }

    for chat_id_str, topics in cache.items():
        chat_id = int(chat_id_str)
        for tf_str, thread_id in topics.items():
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=f"✅ Тест темы <b>{tf_names.get(tf_str, tf_str)}</b> — OK",
                    parse_mode=ParseMode.HTML,
                )
                print(f"OK  chat={chat_id} tf={tf_str} thread={thread_id}")
            except Exception as e:
                print(f"ERR chat={chat_id} tf={tf_str} thread={thread_id} → {e}")

asyncio.run(main())