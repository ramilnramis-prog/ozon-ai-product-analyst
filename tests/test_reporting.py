import pandas as pd

from app.reporting import (
    build_candidate_report,
    build_category_top,
)


def test_candidate_report_keeps_user_facing_fields():
    dataframe = pd.DataFrame(
        {
            "product_key": ["sku:123"],
            "product_name": ["Тестовый товар"],
            "opportunity_rank": [1],
            "opportunity_score": [82.0],
            "score_coverage": [100.0],
            "eligibility_status": ["eligible"],
            "is_eligible": [True],
            "active_seller_count": [10],
            "strong_seller_count": [5],
            "top_3_seller_share": [0.35],
            "top_10_seller_share": [0.70],
            "low_market_depth_warning": [False],
            "high_competition_warning": [False],
        }
    )

    report = build_candidate_report(
        dataframe
    )

    assert "product_name" in report.columns
    assert "opportunity_score" in report.columns

    assert "active_seller_count" in report.columns
    assert "strong_seller_count" in report.columns
    assert "top_3_seller_share" in report.columns
    assert "top_10_seller_share" in report.columns

    assert "low_market_depth_warning" in report.columns
    assert "high_competition_warning" in report.columns

    assert "product_key" not in report.columns

def test_category_top_deduplicates_same_product_name():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Газонокосилка бензиновая LM4600",
                "Газонокосилка бензиновая LM4600",
                "Триммер аккумуляторный",
            ],
            "sku": [
                111,
                222,
                333,
            ],
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Дом и сад",
            ],
            "opportunity_score": [
                70.0,
                55.0,
                65.0,
            ],
        }
    )

    result = build_category_top(
        dataframe,
        top_n=10,
    )

    assert result["product_name"].tolist() == [
        "Газонокосилка бензиновая LM4600",
        "Триммер аккумуляторный",
    ]

    assert result["sku"].tolist() == [
        111,
        333,
    ]

    assert result["category_rank"].tolist() == [
        1,
        2,
    ]
def test_category_top_respects_opportunity_rank():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар eligible",
                "Товар rejected",
            ],
            "root_category": [
                "Дом и сад",
                "Дом и сад",
            ],
            "opportunity_score": [
                70.0,
                95.0,
            ],
            "opportunity_rank": [
                1,
                2,
            ],
        }
    )

    result = build_category_top(
        dataframe,
        top_n=10,
    )

    assert result["product_name"].tolist() == [
        "Товар eligible",
        "Товар rejected",
    ]

    assert result["category_rank"].tolist() == [
        1,
        2,
    ]