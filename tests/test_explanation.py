import pandas as pd

from app.explanation import (
    build_explanation_context,
    build_explanation_prompt,
)


def test_explanation_uses_grounded_facts():
    row = pd.Series(
        {
            "product_name": "Тестовый товар",
            "opportunity_score": 82.56789,
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
    ] == 82.57

    assert context[
        "competition_score"
    ] is None

    prompt = build_explanation_prompt(
        context
    )
    assert (
        "воспроизводи точно"
        in prompt
    )

    assert (
        "Не расширяй смысл метрик"
        in prompt
    )

    assert (
        "Не добавляй новые направления анализа"
        in prompt
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
            text,
        ):
            calls.append(
                {
                    "model": model,
                    "input": input,
                    "text": text,
                }
            )

            return SimpleNamespace(
                output_text=(
                    '{"summary":"Тестовое объяснение",'
                    '"strengths":[],'
                    '"risks":[],'
                    '"next_checks":[]}'
                )
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

    assert result == {
        "summary": "Тестовое объяснение",
        "strengths": [],
        "risks": [],
        "next_checks": [],
    }

    assert len(calls) == 1

    assert calls[0]["model"] == "test-model"

    assert "Тестовый товар" in calls[0]["input"]

    assert calls[0]["text"]["format"]["type"] == (
        "json_schema"
    )

    assert calls[0]["text"]["format"]["strict"] is True

def test_generate_explanation_uses_default_model():
    calls = []

    class FakeResponses:
        def create(
            self,
            *,
            model,
            input,
            text,
        ):
            calls.append(
                {
                    "model": model,
                    "input": input,
                    "text": text,
                }
            )

            return SimpleNamespace(
                output_text=(
                    '{"summary":"Тестовое объяснение",'
                    '"strengths":[],'
                    '"risks":[],'
                    '"next_checks":[]}'
                )
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

    assert calls[0][
        "text"
    ]["format"]["strict"] is True