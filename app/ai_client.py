import os

from dotenv import load_dotenv
from openai import OpenAI


def get_openai_client() -> OpenAI:
    """
    Создает настроенный OpenAI client.

    API-ключ загружается из переменной
    окружения OPENAI_API_KEY.
    """

    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI()