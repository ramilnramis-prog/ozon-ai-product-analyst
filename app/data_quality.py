import pandas as pd


QUALITY_WEIGHTS = {
    "product_identity": 20,
    "demand": 25,
    "period": 15,
    "seller": 15,
    "inventory": 10,
    "market_context": 10,
    "logistics": 5,
}


def _column_coverage(
    dataframe: pd.DataFrame,
    column: str,
) -> float:
    """
    Возвращает долю заполненных значений от 0 до 1.
    """

    if column not in dataframe.columns:
        return 0.0

    if dataframe.empty:
        return 0.0

    return float(
        dataframe[column].notna().mean()
    )


def _best_coverage(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> float:
    """
    Берет лучшее покрытие среди нескольких
    альтернативных колонок.
    """

    return max(
        (
            _column_coverage(dataframe, column)
            for column in columns
        ),
        default=0.0,
    )


def calculate_data_quality(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """
    Оценивает полноту данных по шкале 0–100.

    Score показывает качество входных данных,
    а не привлекательность товара.
    """

    if dataframe.empty:
        return {
            "data_quality_score": 0.0,
            "data_quality_level": "very_low",
            "components": {},
        }

    components = {
        "product_identity": _best_coverage(
            dataframe,
            ["product_name", "sku"],
        ),
        "demand": _best_coverage(
            dataframe,
            ["sales_units", "revenue"],
        ),
        "period": _best_coverage(
            dataframe,
            [
                "period_days",
                "period_start",
                "period_end",
            ],
        ),
        "seller": _best_coverage(
            dataframe,
            ["seller_id", "seller"],
        ),
        "inventory": _column_coverage(
            dataframe,
            "stock",
        ),
        "market_context": _best_coverage(
            dataframe,
            [
                "reviews",
                "rating",
                "category",
                "brand",
            ],
        ),
        "logistics": _best_coverage(
            dataframe,
            ["weight", "volume"],
        ),
    }

    score = 0.0

    for component, coverage in components.items():
        score += (
            coverage
            * QUALITY_WEIGHTS[component]
        )

    score = round(score, 1)

    if score >= 80:
        level = "high"
    elif score >= 60:
        level = "medium"
    elif score >= 40:
        level = "low"
    else:
        level = "very_low"

    return {
        "data_quality_score": score,
        "data_quality_level": level,
        "components": components,
    }