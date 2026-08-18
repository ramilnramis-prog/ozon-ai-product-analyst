import pandas as pd


def build_candidate_features(
    dataframe: pd.DataFrame,
    trends: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Собирает одну итоговую строку признаков
    для каждого товара.

    Использует:
    - последний доступный период;
    - динамику;
    - стабильность.
    """

    required_columns = {
        "product_key",
        "period_start",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing candidate columns: "
            f"{sorted(missing_columns)}"
        )

    latest = (
        dataframe
        .dropna(
            subset=[
                "product_key",
                "period_start",
            ]
        )
        .sort_values("period_start")
        .groupby(
            "product_key",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest_columns = [
        "product_key",
        "product_name",
        "sku",
        "category",
        "brand",
        "price",
        "sales_units",
        "sales_per_day",
        "revenue",
        "revenue_per_day",
        "stock",
        "reviews",
        "rating",
        "period_start",
        "period_end",
    ]

    latest_columns = [
        column
        for column in latest_columns
        if column in latest.columns
    ]

    candidates = latest[
        latest_columns
    ].copy()

    rename_latest = {
        "price": "latest_price",
        "sales_units": "latest_sales_units",
        "sales_per_day": "latest_sales_per_day",
        "revenue": "latest_revenue",
        "revenue_per_day": "latest_revenue_per_day",
        "stock": "latest_stock",
        "reviews": "latest_reviews",
        "rating": "latest_rating",
    }

    candidates = candidates.rename(
        columns=rename_latest
    )

    trend_columns = [
        "product_key",
        "history_periods",
        "first_sales_per_day",
        "growth_rate",
        "trend_direction",
    ]

    trend_columns = [
        column
        for column in trend_columns
        if column in trends.columns
    ]

    candidates = candidates.merge(
        trends[trend_columns],
        on="product_key",
        how="left",
        suffixes=("", "_trend"),
    )

    stability_columns = [
        "product_key",
        "volatility_cv",
        "direction_consistency",
        "stability_status",
    ]

    stability_columns = [
        column
        for column in stability_columns
        if column in stability.columns
    ]

    candidates = candidates.merge(
        stability[stability_columns],
        on="product_key",
        how="left",
    )

    return candidates.reset_index(
        drop=True
    )