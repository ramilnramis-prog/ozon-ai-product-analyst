import pandas as pd

from app.explanation import (
    build_explanation_context,
    build_explanation_prompt,
)


def test_explanation_uses_grounded_facts():
    row = pd.Series(
        {
            "product_name": "Тестовый товар",
            "opportunity_score": 82.5,
            "score_coverage": 70.0,
            "eligibility_status": (
                "insufficient_competition_data"
            ),
            "competition_score": pd.NA,
            "growth_rate": 0.25,
            "trend_direction": "growing",
            "niche_key_confidence": "very_low",
        }
    )

    context = build_explanation_context(
        row
    )

    assert context[
        "product_name"
    ] == "Тестовый товар"

    assert context[
        "opportunity_score"
    ] == 82.5

    assert context[
        "competition_score"
    ] is None

    prompt = build_explanation_prompt(
        context
    )

    assert "Тестовый товар" in prompt
    assert '"competition_score": null' in prompt
    assert "Не пересчитывай Opportunity Score" in prompt
    assert "Не придумывай отсутствующие показатели" in prompt

from types import SimpleNamespace

from app.explanation import generate_explanation


def test_generate_explanation_uses_responses_api():
    calls = []

    class FakeResponses:
        def create(
            self,
            *,
            model,
            input,
        ):
            calls.append(
                {
                    "model": model,
                    "input": input,
                }
            )

            return SimpleNamespace(
                output_text="Тестовое объяснение"
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    context = {
        "product_name": "Тестовый товар",
        "opportunity_score": 82.5,
        "score_coverage": 70.0,
    }

    result = generate_explanation(
        context=context,
        client=fake_client,
        model="test-model",
    )

    assert result == "Тестовое объяснение"

    assert len(calls) == 1

    assert calls[0][
        "model"
    ] == "test-model"

    assert "Тестовый товар" in calls[0][
        "input"
    ]

    assert "82.5" in calls[0][
        "input"
    ]

def test_generate_explanation_uses_default_model():
    calls = []

    class FakeResponses:
        def create(
            self,
            *,
            model,
            input,
        ):
            calls.append(
                {
                    "model": model,
                    "input": input,
                }
            )

            return SimpleNamespace(
                output_text="Тестовое объяснение"
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    context = {
        "product_name": "Тестовый товар",
        "opportunity_score": 80.0,
    }

    generate_explanation(
        context=context,
        client=fake_client,
    )

    assert calls[0][
        "model"
    ] == "gpt-5-nano"