import os
from typing import Dict, Optional

import requests


class TelegramRouter:
    """
    Определяет нужный message_thread_id по паре (ticker, timeframe).
    ID загружаются из .env по паттерну: TOPIC_{TICKER}_{TF}
    Пример: TOPIC_SBER_15MIN=123
    """

    def __init__(self, bot_token: str, group_id: str):
        self.bot_token = bot_token
        self.group_id  = group_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
        self._topics: Dict[str, int] = self._load_topics()

    def _load_topics(self) -> Dict[str, int]:
        topics = {}
        for key, value in os.environ.items():
            if key.startswith("TOPIC_"):
                route_key = key[len("TOPIC_"):].upper()
                try:
                    topics[route_key] = int(value)
                except ValueError:
                    print(f"[TelegramRouter] Некорректный thread_id для {key}: {value}")
        return topics

    def _get_thread_id(self, ticker: str, tf: str) -> Optional[int]:
        key = f"{ticker.upper()}_{tf.upper()}"
        thread_id = self._topics.get(key)
        if thread_id is None:
            print(f"[TelegramRouter] Тема не найдена для ключа: {key}. Отправляю в общий чат.")
        return thread_id

    def send_message(self, ticker: str, tf: str, text: str):
        thread_id = self._get_thread_id(ticker, tf)
        payload = {
            "chat_id":    self.group_id,
            "text":       text,
            "parse_mode": "Markdown",
        }
        if thread_id:
            payload["message_thread_id"] = thread_id

        try:
            resp = requests.post(f"{self._base_url}/sendMessage", data=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[TelegramRouter] Ошибка sendMessage: {e}")

    def send_photo(self, ticker: str, tf: str, caption: str, img_path: str):
        thread_id = self._get_thread_id(ticker, tf)
        payload = {
            "chat_id":    self.group_id,
            "caption":    caption,
            "parse_mode": "Markdown",
        }
        if thread_id:
            payload["message_thread_id"] = thread_id

        try:
            with open(img_path, 'rb') as f:
                resp = requests.post(
                    f"{self._base_url}/sendPhoto",
                    data=payload,
                    files={"photo": f},
                    timeout=60
                )
                resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[TelegramRouter] Ошибка sendPhoto: {e}")
        except FileNotFoundError:
            print(f"[TelegramRouter] Файл не найден: {img_path}")
