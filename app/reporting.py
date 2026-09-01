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

    if "category_rank" in result.columns:
        result = result.sort_values(
            by=[
                "root_category",
                "category_rank",
            ],
            ascending=[
                True,
                True,
            ],
            na_position="last",
        )
    elif "opportunity_rank" in result.columns:
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

    if "category_rank" in result.columns:
        result["category_rank"] = (
            result
            .groupby("root_category", dropna=True)["category_rank"]
            .rank(
                method="dense",
                ascending=True,
            )
            .astype("Int64")
        )
    else:
        result["category_rank"] = (
            result
            .groupby("root_category", dropna=True)
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


def build_leaf_category_top(
    dataframe: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    ????????? TOP ??????? ?????? ?????? leaf_category.

    Root category ??????????? ??? ??????????????
    ??????????? ? UI.
    """

    required_columns = {
        "root_category",
        "leaf_category",
        "product_name",
        "opportunity_score",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing leaf category top columns: "
            f"{sorted(missing_columns)}"
        )

    result = dataframe.copy()

    result = result[
        result["root_category"].notna()
        & result["leaf_category"].notna()
        & result["product_name"].notna()
    ].copy()

    result["_product_concept_key"] = (
        result["product_name"]
        .map(normalize_niche_text)
    )

    status_priority = {
        "eligible": 0,
        "insufficient_competition_data": 1,
        "rejected_low_market_depth": 2,
    }

    if "eligibility_status" in result.columns:
        result["_status_priority"] = (
            result["eligibility_status"]
            .map(status_priority)
            .fillna(1)
        )
    else:
        result["_status_priority"] = 1

    result = result.sort_values(
        by=[
            "root_category",
            "leaf_category",
            "_status_priority",
            "opportunity_score",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
        na_position="last",
    )

    result = result.drop_duplicates(
        subset=[
            "root_category",
            "leaf_category",
            "_product_concept_key",
        ],
        keep="first",
    )

    result["leaf_category_rank"] = (
        result
        .groupby(
            ["root_category", "leaf_category"],
            dropna=True,
        )
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    result = result[
        result["leaf_category_rank"] <= top_n
    ].copy()

    return result.drop(
        columns=[
            "_product_concept_key",
            "_status_priority",
        ]
    ).reset_index(drop=True)


def build_unclassified_top(
    dataframe: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Возвращает TOP товаров без root_category.

    Такие товары не участвуют в category ranking,
    но сохраняются в аналитике и сортируются
    по глобальному Opportunity Score.
    """

    required_columns = {
        "product_name",
        "opportunity_score",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = dataframe.copy()

    if "root_category" in result.columns:
        root_category = (
            result["root_category"]
            .astype("string")
            .str.strip()
        )

        result = result[
            root_category.isna()
            | root_category.eq("")
        ].copy()

    result = result[
        result["product_name"].notna()
    ].copy()

    result["_product_concept_key"] = (
        result["product_name"]
        .map(normalize_niche_text)
    )

    if "eligibility_status" in result.columns:
        eligibility = (
            result["eligibility_status"]
            .astype("string")
            .fillna("")
        )
    else:
        eligibility = pd.Series(
            "",
            index=result.index,
            dtype="string",
        )

    result["review_group"] = "needs_review"

    result.loc[
        eligibility.eq("eligible"),
        "review_group",
    ] = "ready_for_evaluation"

    result.loc[
        eligibility.eq(
            "insufficient_competition_data"
        ),
        "review_group",
    ] = "needs_competition_data"

    result.loc[
        eligibility.str.startswith("rejected_"),
        "review_group",
    ] = "rejected"

    group_priority = {
        "ready_for_evaluation": 0,
        "needs_competition_data": 1,
        "rejected": 2,
        "needs_review": 3,
    }

    result["_review_group_priority"] = (
        result["review_group"]
        .map(group_priority)
    )

    sort_columns = [
        "_review_group_priority",
        "opportunity_score",
    ]

    ascending = [
        True,
        False,
    ]

    if "opportunity_rank" in result.columns:
        sort_columns.append("opportunity_rank")
        ascending.append(True)

    result = result.sort_values(
        by=sort_columns,
        ascending=ascending,
        na_position="last",
    )

    result = result.drop_duplicates(
        subset=["_product_concept_key"],
        keep="first",
    )

    result["review_priority"] = (
        result
        .groupby(
            "review_group",
            dropna=False,
        )
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    result = result[
        result["review_priority"] <= top_n
    ].copy()

    return result.drop(
        columns=[
            "_product_concept_key",
            "_review_group_priority",
        ]
    )