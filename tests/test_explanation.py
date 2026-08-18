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
    