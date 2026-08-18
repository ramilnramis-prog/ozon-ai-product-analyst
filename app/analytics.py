import pandas as pd


def add_daily_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет дневные показатели, если их можно рассчитать.

    sales_per_day:
        sales_units / period_days

    revenue_per_day:
        revenue / period_days

    Если готовая дневная метрика уже есть,
    существующее значение не перезаписывается.
    """

    result = dataframe.copy()

    valid_period_days = None

    if "period_days" in result.columns:
        valid_period_days = result["period_days"].where(
            result["period_days"] > 0
        )

    if (
        "sales_units" in result.columns
        and "period_days" in result.columns
    ):
        calculated_sales_per_day = (
            result["sales_units"]
            / valid_period_days
        )

        if "sales_per_day" not in result.columns:
            result["sales_per_day"] = (
                calculated_sales_per_day
            )
        else:
            result["sales_per_day"] = (
                result["sales_per_day"].fillna(
                    calculated_sales_per_day
                )
            )

    if (
        "revenue" in result.columns
        and "period_days" in result.columns
    ):
        calculated_revenue_per_day = (
            result["revenue"]
            / valid_period_days
        )

        if "revenue_per_day" not in result.columns:
            result["revenue_per_day"] = (
                calculated_revenue_per_day
            )
        else:
            result["revenue_per_day"] = (
                result["revenue_per_day"].fillna(
                    calculated_revenue_per_day
                )
            )

    return result

def add_share_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет долю каждого товара в продажах и выручке
    внутри текущего набора данных.

    Значения хранятся от 0 до 1.
    Например: 0.25 = 25%.
    """

    result = dataframe.copy()

    if "sales_units" in result.columns:
        total_sales = result["sales_units"].sum(
            min_count=1
        )

        if pd.notna(total_sales) and total_sales > 0:
            result["sales_share"] = (
                result["sales_units"]
                / total_sales
            )
        else:
            result["sales_share"] = pd.NA

    if "revenue" in result.columns:
        total_revenue = result["revenue"].sum(
            min_count=1
        )

        if pd.notna(total_revenue) and total_revenue > 0:
            result["revenue_share"] = (
                result["revenue"]
                / total_revenue
            )
        else:
            result["revenue_share"] = pd.NA

    return result

def calculate_concentration(
    dataframe: pd.DataFrame,
    value_column: str = "revenue",
) -> dict[str, float | None]:
    """
    Считает долю крупнейших строк в общем объеме метрики.

    Например для revenue:
    top_1_share = доля крупнейшего товара
    top_3_share = совокупная доля трех крупнейших товаров
    top_10_share = совокупная доля десяти крупнейших товаров
    """

    if value_column not in dataframe.columns:
        return {
            "top_1_share": None,
            "top_3_share": None,
            "top_10_share": None,
        }

    values = dataframe[value_column].dropna()

    values = values[
        values > 0
    ].sort_values(
        ascending=False
    )

    total = values.sum()

    if values.empty or total <= 0:
        return {
            "top_1_share": None,
            "top_3_share": None,
            "top_10_share": None,
        }

    return {
        "top_1_share": float(
            values.head(1).sum() / total
        ),
        "top_3_share": float(
            values.head(3).sum() / total
        ),
        "top_10_share": float(
            values.head(10).sum() / total
        ),
    }