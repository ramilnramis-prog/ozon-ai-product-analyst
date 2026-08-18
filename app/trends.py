import pandas as pd


def calculate_product_trends(
    dataframe: pd.DataFrame,
    stable_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Рассчитывает динамику продаж каждого товара
    между первым и последним доступным периодом.

    stable_threshold = 0.10 означает:
    изменение в пределах ±10% считаем стабильным.
    """

    required_columns = {
        "product_key",
        "period_start",
        "sales_per_day",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns for trend analysis: "
            f"{sorted(missing_columns)}"
        )

    valid = dataframe.dropna(
        subset=[
            "product_key",
            "period_start",
            "sales_per_day",
        ]
    ).copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "product_key",
                "product_name",
                "history_periods",
                "first_sales_per_day",
                "latest_sales_per_day",
                "growth_rate",
                "trend_direction",
            ]
        )

    rows = []

    for product_key, group in valid.groupby(
        "product_key"
    ):
        group = group.sort_values(
            "period_start"
        )

        first_row = group.iloc[0]
        latest_row = group.iloc[-1]

        first_sales = float(
            first_row["sales_per_day"]
        )

        latest_sales = float(
            latest_row["sales_per_day"]
        )

        history_periods = int(
            group["period_start"].nunique()
        )

        growth_rate = None
        trend_direction = "insufficient_history"

        if history_periods >= 2 and first_sales > 0:
            growth_rate = (
                latest_sales - first_sales
            ) / first_sales

            if growth_rate > stable_threshold:
                trend_direction = "growing"

            elif growth_rate < -stable_threshold:
                trend_direction = "declining"

            else:
                trend_direction = "stable"

        product_name = None

        if "product_name" in group.columns:
            names = (
                group["product_name"]
                .dropna()
                .astype(str)
            )

            if not names.empty:
                product_name = names.iloc[-1]

        rows.append(
            {
                "product_key": product_key,
                "product_name": product_name,
                "history_periods": history_periods,
                "first_sales_per_day": first_sales,
                "latest_sales_per_day": latest_sales,
                "growth_rate": growth_rate,
                "trend_direction": trend_direction,
            }
        )

    return pd.DataFrame(rows)

def calculate_product_stability(
    dataframe: pd.DataFrame,
    min_periods: int = 3,
) -> pd.DataFrame:
    """
    Оценивает стабильность продаж товара во времени.

    volatility_cv:
        коэффициент вариации.
        Чем ближе к 0, тем ровнее продажи.

    direction_consistency:
        насколько последовательно менялись продажи.
        1.0 = все изменения шли в одном направлении.

    Для надежной оценки требуется минимум 3 периода.
    """

    required_columns = {
        "product_key",
        "period_start",
        "sales_per_day",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns for stability analysis: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    for product_key, group in dataframe.groupby(
        "product_key"
    ):
        group = (
            group.dropna(
                subset=[
                    "period_start",
                    "sales_per_day",
                ]
            )
            .sort_values("period_start")
        )

        values = group["sales_per_day"].astype(
            float
        )

        history_periods = int(
            group["period_start"].nunique()
        )

        volatility_cv = None
        direction_consistency = None
        stability_status = "insufficient_history"

        if history_periods >= min_periods:
            mean_sales = values.mean()

            if mean_sales > 0:
                volatility_cv = float(
                    values.std(ddof=0)
                    / mean_sales
                )

            changes = values.diff().dropna()

            if not changes.empty:
                positive = int(
                    (changes > 0).sum()
                )
                negative = int(
                    (changes < 0).sum()
                )
                unchanged = int(
                    (changes == 0).sum()
                )

                direction_consistency = float(
                    max(
                        positive,
                        negative,
                        unchanged,
                    )
                    / len(changes)
                )

            stability_status = "available"

        product_name = None

        if "product_name" in group.columns:
            names = (
                group["product_name"]
                .dropna()
                .astype(str)
            )

            if not names.empty:
                product_name = names.iloc[-1]

        rows.append(
            {
                "product_key": product_key,
                "product_name": product_name,
                "history_periods": history_periods,
                "volatility_cv": volatility_cv,
                "direction_consistency": (
                    direction_consistency
                ),
                "stability_status": (
                    stability_status
                ),
            }
        )

    return pd.DataFrame(rows)
def calculate_product_stability(
    dataframe: pd.DataFrame,
    min_periods: int = 3,
) -> pd.DataFrame:
    """
    Оценивает стабильность продаж товара во времени.

    volatility_cv:
        коэффициент вариации.
        Чем ближе к 0, тем ровнее продажи.

    direction_consistency:
        насколько последовательно менялись продажи.
        1.0 = все изменения шли в одном направлении.

    Для надежной оценки требуется минимум 3 периода.
    """

    required_columns = {
        "product_key",
        "period_start",
        "sales_per_day",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns for stability analysis: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    for product_key, group in dataframe.groupby(
        "product_key"
    ):
        group = (
            group.dropna(
                subset=[
                    "period_start",
                    "sales_per_day",
                ]
            )
            .sort_values("period_start")
        )

        values = group["sales_per_day"].astype(
            float
        )

        history_periods = int(
            group["period_start"].nunique()
        )

        volatility_cv = None
        direction_consistency = None
        stability_status = "insufficient_history"

        if history_periods >= min_periods:
            mean_sales = values.mean()

            if mean_sales > 0:
                volatility_cv = float(
                    values.std(ddof=0)
                    / mean_sales
                )

            changes = values.diff().dropna()

            if not changes.empty:
                positive = int(
                    (changes > 0).sum()
                )
                negative = int(
                    (changes < 0).sum()
                )
                unchanged = int(
                    (changes == 0).sum()
                )

                direction_consistency = float(
                    max(
                        positive,
                        negative,
                        unchanged,
                    )
                    / len(changes)
                )

            stability_status = "available"

        product_name = None

        if "product_name" in group.columns:
            names = (
                group["product_name"]
                .dropna()
                .astype(str)
            )

            if not names.empty:
                product_name = names.iloc[-1]

        rows.append(
            {
                "product_key": product_key,
                "product_name": product_name,
                "history_periods": history_periods,
                "volatility_cv": volatility_cv,
                "direction_consistency": (
                    direction_consistency
                ),
                "stability_status": (
                    stability_status
                ),
            }
        )

    return pd.DataFrame(rows)