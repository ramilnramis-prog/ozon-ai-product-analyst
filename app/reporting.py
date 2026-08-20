import pandas as pd

from app.niche_grouping import normalize_niche_text


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

def build_category_top(
    dataframe: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Формирует TOP товаров внутри каждой основной категории.

    Одинаковые нормализованные названия товара
    считаются одной товарной концепцией только
    для категорийного отчета.
    """

    required_columns = {
        "root_category",
        "product_name",
        "opportunity_score",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing category top columns: "
            f"{sorted(missing_columns)}"
        )

    result = dataframe.copy()

    result = result[
        result["root_category"].notna()
        & result["product_name"].notna()
    ].copy()

    result["_product_concept_key"] = (
        result["product_name"]
        .map(normalize_niche_text)
    )

    if "opportunity_rank" in result.columns:
        result = result.sort_values(
            "opportunity_rank",
            ascending=True,
            na_position="last",
    )
    else:
        result = result.sort_values(
            "opportunity_score",
            ascending=False,
            na_position="last",
    )

    result = result.drop_duplicates(
        subset=[
            "root_category",
            "_product_concept_key",
        ],
        keep="first",
    )

    result["category_rank"] = (
        result
        .groupby("root_category")
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    result = result[
        result["category_rank"] <= top_n
    ].copy()

    result = result.drop(
        columns=["_product_concept_key"]
    )

    return result.reset_index(drop=True)