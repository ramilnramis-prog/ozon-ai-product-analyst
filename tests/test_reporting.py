import pandas as pd

from app.reporting import build_candidate_report


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