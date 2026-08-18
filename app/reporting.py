import pandas as pd


REPORT_COLUMNS = [
    "opportunity_rank",
    "product_name",
    "opportunity_score",
    "score_coverage",
    "eligibility_status",
    "is_eligible",
    "demand_score",
    "growth_score",
    "stability_score",
    "competition_score",
    "concentration_score",
    "latest_price",
    "latest_sales_per_day",
    "latest_revenue_per_day",
    "growth_rate",
    "trend_direction",
    "active_seller_count",
    "strong_seller_count",
    "top_3_seller_share",
    "top_10_seller_share",
    "low_market_depth_warning",
    "high_competition_warning",
    "niche_key",
    "niche_key_confidence",
]


def build_candidate_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Подготавливает итоговую таблицу кандидатов
    для UI, API и будущего AI-объяснения.

    Внутренние технические поля,
    которые не нужны пользователю,
    в отчет не попадают.
    """

    available_columns = [
        column
        for column in REPORT_COLUMNS
        if column in dataframe.columns
    ]

    report = dataframe[
        available_columns
    ].copy()

    return report