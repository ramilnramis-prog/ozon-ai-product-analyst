import pandas as pd


def rank_candidates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ранжирует кандидатов с учетом eligibility.

    Приоритет:
    1. eligible;
    2. недостаточно данных для подтверждения;
    3. rejected.

    Внутри каждой группы:
    Opportunity Score по убыванию.
    """

    result = dataframe.copy()

    if "opportunity_score" not in result.columns:
        raise ValueError(
            "opportunity_score is required for ranking."
        )

    status_priority = {
        "eligible": 0,
        "insufficient_competition_data": 1,
        "rejected_low_market_depth": 2,
    }

    result["_status_priority"] = (
        result["eligibility_status"]
        .map(status_priority)
        .fillna(1)
    )

    result = result.sort_values(
        by=[
            "_status_priority",
            "opportunity_score",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    result["opportunity_rank"] = (
        result.index + 1
    )

    if "root_category" in result.columns:
        category_score_column = (
            "category_opportunity_score"
            if "category_opportunity_score" in result.columns
            else "opportunity_score"
        )

        category_order = result.copy()

        category_order["_category_score_for_rank"] = (
            pd.to_numeric(
                category_order[category_score_column],
                errors="coerce",
            )
            .round(2)
        )

        category_order = category_order.sort_values(
            by=[
                "root_category",
                "_status_priority",
                "_category_score_for_rank",
                "opportunity_rank",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            na_position="last",
        ).copy()

        same_category = category_order["root_category"].eq(
            category_order["root_category"].shift()
        )

        same_status = category_order["_status_priority"].eq(
            category_order["_status_priority"].shift()
        )

        same_score = category_order["_category_score_for_rank"].eq(
            category_order["_category_score_for_rank"].shift()
        )

        new_rank_group = ~(
            same_category
            & same_status
            & same_score
        )

        category_order["_category_rank"] = (
            new_rank_group
            .groupby(
                category_order["root_category"],
                dropna=True,
            )
            .cumsum()
            .astype("Int64")
        )

        result["category_rank"] = (
            category_order["_category_rank"]
            .reindex(result.index)
        )
    else:
        result["category_rank"] = pd.NA

    result = result.drop(
        columns=["_status_priority"]
    )

    return result