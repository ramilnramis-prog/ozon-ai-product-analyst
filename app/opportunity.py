import pandas as pd


def calculate_opportunity_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """
    Рассчитывает основные объективные показатели
    товарного набора / ниши.

    Здесь нет оценки "хорошо" или "плохо".
    Только факты.
    """

    metrics: dict[str, object] = {
        "product_count": int(len(dataframe)),
    }

    if "sales_units" in dataframe.columns:
        sales = dataframe["sales_units"].dropna()

        metrics["total_sales_units"] = (
            float(sales.sum())
            if not sales.empty
            else None
        )

        metrics["median_sales_units"] = (
            float(sales.median())
            if not sales.empty
            else None
        )

    else:
        metrics["total_sales_units"] = None
        metrics["median_sales_units"] = None

    if "revenue" in dataframe.columns:
        revenue = dataframe["revenue"].dropna()

        metrics["total_revenue"] = (
            float(revenue.sum())
            if not revenue.empty
            else None
        )

        metrics["median_revenue"] = (
            float(revenue.median())
            if not revenue.empty
            else None
        )

    else:
        metrics["total_revenue"] = None
        metrics["median_revenue"] = None

    if "price" in dataframe.columns:
        prices = dataframe["price"].dropna()

        prices = prices[
            prices > 0
        ]

        metrics["median_price"] = (
            float(prices.median())
            if not prices.empty
            else None
        )

    else:
        metrics["median_price"] = None

    if "sales_per_day" in dataframe.columns:
        daily_sales = dataframe[
            "sales_per_day"
        ].dropna()

        metrics["total_sales_per_day"] = (
            float(daily_sales.sum())
            if not daily_sales.empty
            else None
        )

    else:
        metrics["total_sales_per_day"] = None

    if "revenue_per_day" in dataframe.columns:
        daily_revenue = dataframe[
            "revenue_per_day"
        ].dropna()

        metrics["total_revenue_per_day"] = (
            float(daily_revenue.sum())
            if not daily_revenue.empty
            else None
        )

    else:
        metrics["total_revenue_per_day"] = None

    if "stock" in dataframe.columns:
        stock = dataframe["stock"]

        valid_stock = stock.dropna()

        metrics["total_stock"] = (
            float(valid_stock.sum())
            if not valid_stock.empty
            else None
        )

        if not valid_stock.empty:
            metrics["out_of_stock_share"] = float(
                (valid_stock <= 0).mean()
            )
        else:
            metrics["out_of_stock_share"] = None

    else:
        metrics["total_stock"] = None
        metrics["out_of_stock_share"] = None

    return metrics