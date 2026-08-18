import pandas as pd

from app.scoring import add_competition_score


def test_competition_score_respects_market_depth():
    dataframe = pd.DataFrame(
        {
            "active_seller_count": [
                2,
                5,
                10,
                20,
                25,
            ],
            "strong_seller_count": [
                1,
                2,
                8,
                15,
                20,
            ],
        }
    )

    result = add_competition_score(
        dataframe
    )

    assert result[
        "competition_score"
    ].tolist() == [
        0.0,
        80.0,
        60.0,
        20.0,
        10.0,
    ]
from app.scoring import add_eligibility_status


def test_eligibility_status():
    dataframe = pd.DataFrame(
        {
            "active_seller_count": [
                2,
                10,
                None,
            ]
        }
    )

    result = add_eligibility_status(
        dataframe
    )

    assert result[
        "eligibility_status"
    ].tolist() == [
        "rejected_low_market_depth",
        "eligible",
        "insufficient_competition_data",
    ]

    assert result[
        "is_eligible"
    ].tolist() == [
        False,
        True,
        pd.NA,
    ]
import pytest

from app.scoring import add_opportunity_score


def test_opportunity_score_reweights_available_components():
    dataframe = pd.DataFrame(
        {
            "demand_score": [100.0],
            "growth_score": [50.0],
            "stability_score": [0.0],
            "competition_score": [pd.NA],
            "concentration_score": [pd.NA],
        }
    )

    result = add_opportunity_score(
        dataframe
    )

    assert result.loc[
        0,
        "opportunity_score",
    ] == pytest.approx(
        60.7142857
    )

    assert result.loc[
        0,
        "score_coverage",
    ] == pytest.approx(
        70.0
    )