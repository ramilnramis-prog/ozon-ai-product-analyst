import pytest

from app.ai_client import get_openai_client


def test_missing_api_key_raises_clear_error(
    monkeypatch,
):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="OPENAI_API_KEY is not configured",
    ):
        get_openai_client()