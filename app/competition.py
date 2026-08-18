import pandas as pd

from app.analytics import calculate_concentration


def aggregate_by_seller(
    dataframe: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Объединяет карточки одного продавца.

    Приоритет идентификации:
    1. seller_id
    2. seller

    Если данных о продавцах нет,
    возвращает None.
    """

    if "seller_id" in dataframe.columns:
        seller_column = "seller_id"

    elif "seller" in dataframe.columns:
        seller_column = "seller"

    else:
        return None

    valid = dataframe.copy()

    valid[seller_column] = (
        valid[seller_column]
        .astype("string")
        .str.strip()
    )

    valid = valid[
        valid[seller_column].notna()
        & (valid[seller_column] != "")
    ]

    if valid.empty:
        return None

    aggregation = {}

    if "sales_units" in valid.columns:
        aggregation["sales_units"] = "sum"

    if "revenue" in valid.columns:
        aggregation["revenue"] = "sum"

    if "sales_per_day" in valid.columns:
        aggregation["sales_per_day"] = "sum"

    if "revenue_per_day" in valid.columns:
        aggregation["revenue_per_day"] = "sum"

    if "stock" in valid.columns:
        aggregation["stock"] = "sum"

    if "sku" in valid.columns:
        aggregation["sku"] = "nunique"

    if not aggregation:
        return None

    sellers = (
        valid.groupby(
            seller_column,
            as_index=False,
        )
        .agg(aggregation)
    )

    if "sku" in sellers.columns:
        sellers = sellers.rename(
            columns={
                "sku": "product_count",
            }
        )

    return sellers


def calculate_seller_concentration(
    dataframe: pd.DataFrame,
    value_column: str = "revenue",
) -> dict[str, object]:
    """
    Считает концентрацию рынка по независимым продавцам.
    """

    sellers = aggregate_by_seller(dataframe)

    if sellers is None:
        return {
            "seller_data_available": False,
            "seller_count": None,
            "active_seller_count": None,
            "low_market_depth_warning": None,
            "top_1_seller_share": None,
            "top_3_seller_share": None,
            "top_10_seller_share": None,
        }

    concentration = calculate_concentration(
        sellers,
        value_column=value_column,
    )

    return {
        "seller_data_available": True,
        "seller_count": int(len(sellers)),
        "top_1_seller_share": concentration[
            "top_1_share"
        ],
        "top_3_seller_share": concentration[
            "top_3_share"
        ],
        "top_10_seller_share": concentration[
            "top_10_share"
        ],
     }

def calculate_competition_metrics(
    dataframe: pd.DataFrame,
    strong_seller_monthly_revenue: float = 300_000.0,
    low_market_seller_count: int = 3,
    high_competition_seller_count: int = 15,
) -> dict[str, object]:
    """
    Рассчитывает основные показатели конкуренции по продавцам.

    Сильный продавец:
    эквивалент от 300 000 ₽ выручки в месяц.

    Высокая конкуренция:
    от 15 таких независимых продавцов.
    """

    sellers = aggregate_by_seller(dataframe)

    if sellers is None:
        return {
            "seller_data_available": False,
            "seller_count": None,
            "active_seller_count": None,
            "strong_seller_count": None,
            "strong_seller_share": None,
            "top_3_seller_share": None,
            "top_10_seller_share": None,
            "low_market_depth_warning": None,
            "high_competition_warning": None,
        }

    seller_count = int(len(sellers))

    active_seller_count = None
    low_market_depth_warning = None
    strong_seller_count = None
    strong_seller_share = None
    high_competition_warning = None

    if "revenue_per_day" in sellers.columns:
        sellers = sellers.copy()

        sellers["monthly_revenue_equivalent"] = (
            sellers["revenue_per_day"] * 30
        )

        active_mask = (
            sellers["monthly_revenue_equivalent"] > 0
        )

        active_seller_count = int(
            active_mask.sum()
        )

        low_market_depth_warning = (
            active_seller_count
            <= low_market_seller_count
        )

        strong_mask = (
            sellers["monthly_revenue_equivalent"]
            >= strong_seller_monthly_revenue
        )

        strong_seller_count = int(
            strong_mask.sum()
        )

        total_monthly_revenue = (
            sellers["monthly_revenue_equivalent"].sum()
        )

        if total_monthly_revenue > 0:
            strong_seller_share = float(
                sellers.loc[
                    strong_mask,
                    "monthly_revenue_equivalent",
                ].sum()
                / total_monthly_revenue
            )

        concentration = calculate_concentration(
            sellers,
            value_column="monthly_revenue_equivalent",
        )

        high_competition_warning = (
            strong_seller_count
            >= high_competition_seller_count
        )

    elif "revenue" in sellers.columns:
        concentration = calculate_concentration(
            sellers,
            value_column="revenue",
        )

    else:
        concentration = {
            "top_1_share": None,
            "top_3_share": None,
            "top_10_share": None,
        }

    return {
        "seller_data_available": True,
        "seller_count": seller_count,
        "active_seller_count": active_seller_count,
        "low_market_depth_warning": low_market_depth_warning,
        "strong_seller_count": strong_seller_count,
        "strong_seller_share": strong_seller_share,
        "top_3_seller_share": concentration[
            "top_3_share"
        ],
        "top_10_seller_share": concentration[
            "top_10_share"
        ],
        "high_competition_warning": (
            high_competition_warning
        ),
    }
def calculate_niche_competition(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает конкуренцию отдельно внутри каждой ниши.

    Для каждой niche_key используется только
    последний доступный период этой ниши.
    """

    required_columns = {
        "niche_key",
        "period_start",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns for niche competition: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    valid = dataframe.dropna(
        subset=[
            "niche_key",
            "period_start",
        ]
    ).copy()

    for niche_key, group in valid.groupby(
        "niche_key"
    ):
        latest_period = group[
            "period_start"
        ].max()

        latest = group[
            group["period_start"] == latest_period
        ].copy()

        metrics = calculate_competition_metrics(
            latest
        )

        niche_source = None
        niche_confidence = None

        if "niche_key_source" in latest.columns:
            values = (
                latest["niche_key_source"]
                .dropna()
                .astype(str)
            )

            if not values.empty:
                niche_source = values.iloc[0]

        if "niche_key_confidence" in latest.columns:
            values = (
                latest["niche_key_confidence"]
                .dropna()
                .astype(str)
            )

            if not values.empty:
                niche_confidence = values.iloc[0]

        rows.append(
            {
                "niche_key": niche_key,
                "latest_period": latest_period,
                "niche_key_source": niche_source,
                "niche_key_confidence": (
                    niche_confidence
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)