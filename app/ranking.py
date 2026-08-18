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

    result = result.drop(
        columns=["_status_priority"]
    )

    return result