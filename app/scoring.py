import pandas as pd


SCORE_WEIGHTS = {
    "demand": 30.0,
    "growth": 25.0,
    "stability": 15.0,
    "competition": 20.0,
    "concentration": 10.0,
}


def clip_score(value: float) -> float:
    """
    Ограничивает любой компонент диапазоном 0–100.
    """

    return float(
        max(
            0.0,
            min(100.0, value),
        )
    )
def percentile_score(
    series: pd.Series,
) -> pd.Series:
    """
    Переводит числовой показатель в относительный
    балл 0–100 внутри текущей выборки.

    Лучшее значение получает высокий балл,
    худшее — низкий.
    """

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="Float64",
    )

    valid = series.notna()
    count = int(valid.sum())

    if count == 0:
        return result

    if count == 1:
        result.loc[valid] = 50.0
        return result

    ranks = series.loc[valid].rank(
        method="average",
        ascending=True,
    )

    result.loc[valid] = (
        (ranks - 1)
        / (count - 1)
        * 100
    )

    return result


def add_demand_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает Demand Score 0–100.

    Использует:
    - продажи в день;
    - выручку в день.

    Если доступен только один показатель,
    используется только он.
    """

    result = dataframe.copy()

    score_columns = []

    if "latest_sales_per_day" in result.columns:
        result["_sales_demand_score"] = (
            percentile_score(
                result["latest_sales_per_day"]
            )
        )
        score_columns.append(
            "_sales_demand_score"
        )

    if "latest_revenue_per_day" in result.columns:
        result["_revenue_demand_score"] = (
            percentile_score(
                result["latest_revenue_per_day"]
            )
        )
        score_columns.append(
            "_revenue_demand_score"
        )

    if not score_columns:
        result["demand_score"] = pd.NA
        return result

    result["demand_score"] = (
        result[score_columns]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    result = result.drop(
        columns=score_columns
    )

    return result
def add_growth_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает Growth Score 0–100.

    0% роста = 50 баллов.
    +50% и выше = 100 баллов.
    -50% и ниже = 0 баллов.
    """

    result = dataframe.copy()

    if "growth_rate" not in result.columns:
        result["growth_score"] = pd.NA
        return result

    growth = result["growth_rate"]

    score = (
        50.0
        + growth * 100.0
    )

    result["growth_score"] = score.clip(
        lower=0.0,
        upper=100.0,
    )

    result.loc[
        growth.isna(),
        "growth_score",
    ] = pd.NA

    return result
def add_stability_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает Stability Score 0–100.

    Использует:
    - volatility_cv: чем ниже волатильность, тем лучше;
    - direction_consistency: чем последовательнее динамика, тем лучше.

    Направление роста здесь не оценивается.
    Оно учитывается отдельно в Growth Score.
    """

    result = dataframe.copy()

    volatility_score = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    consistency_score = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    if "volatility_cv" in result.columns:
        valid_volatility = (
            result["volatility_cv"].notna()
        )

        volatility_score.loc[
            valid_volatility
        ] = (
            100.0
            * (
                1.0
                - result.loc[
                    valid_volatility,
                    "volatility_cv",
                ].clip(
                    lower=0.0,
                    upper=1.0,
                )
            )
        )

    if "direction_consistency" in result.columns:
        valid_consistency = (
            result[
                "direction_consistency"
            ].notna()
        )

        consistency_score.loc[
            valid_consistency
        ] = (
            result.loc[
                valid_consistency,
                "direction_consistency",
            ].clip(
                lower=0.0,
                upper=1.0,
            )
            * 100.0
        )

    components = pd.concat(
        [
            volatility_score.rename(
                "volatility_score"
            ),
            consistency_score.rename(
                "consistency_score"
            ),
        ],
        axis=1,
    )

    result["stability_score"] = (
        components.mean(
            axis=1,
            skipna=True,
        )
    )

    if "stability_status" in result.columns:
        unavailable = (
            result["stability_status"]
            != "available"
        )

        result.loc[
            unavailable,
            "stability_score",
        ] = pd.NA

    return result
def add_competition_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает Competition Score 0–100.

    Учитывает:
    - глубину рынка по active_seller_count;
    - давление сильных продавцов по strong_seller_count.

    <= 3 активных продавцов считается
    недостаточной глубиной рынка.
    """

    result = dataframe.copy()

    result["competition_score"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    required_columns = {
        "active_seller_count",
        "strong_seller_count",
    }

    if not required_columns.issubset(
        result.columns
    ):
        return result

    valid = (
        result["active_seller_count"].notna()
        & result["strong_seller_count"].notna()
    )

    if not valid.any():
        return result

    active = result.loc[
        valid,
        "active_seller_count",
    ]

    strong = result.loc[
        valid,
        "strong_seller_count",
    ]

    depth_multiplier = pd.Series(
        1.0,
        index=active.index,
    )

    depth_multiplier.loc[
        active <= 3
    ] = 0.0

    depth_multiplier.loc[
        active == 4
    ] = 0.60

    depth_multiplier.loc[
        active == 5
    ] = 0.80


    pressure_score = pd.Series(
        100.0,
        index=strong.index,
    )

    pressure_score.loc[
        strong.between(4, 7)
    ] = 80.0

    pressure_score.loc[
        strong.between(8, 11)
    ] = 60.0

    pressure_score.loc[
        strong.between(12, 14)
    ] = 35.0

    pressure_score.loc[
        strong.between(15, 19)
    ] = 20.0

    pressure_score.loc[
        strong.between(20, 24)
    ] = 10.0

    pressure_score.loc[
        strong >= 25
    ] = 0.0


    combined_score = (
        pressure_score
        * depth_multiplier
    )

    result.loc[
        valid,
        "competition_score",
    ] = combined_score

    return result
def add_concentration_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает Concentration Score 0–100.

    Чем меньше доля рынка у лидеров,
    тем выше балл.

    Использует:
    - долю Top-3 продавцов;
    - долю Top-10 продавцов.
    """

    result = dataframe.copy()

    top_3_score = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    top_10_score = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    if "top_3_seller_share" in result.columns:
        top_3 = result[
            "top_3_seller_share"
        ]

        valid_top_3 = top_3.notna()

        top_3_score.loc[
            valid_top_3
        ] = (
            (
                0.75
                - top_3.loc[valid_top_3]
            )
            / 0.50
            * 100.0
        ).clip(
            lower=0.0,
            upper=100.0,
        )

    if "top_10_seller_share" in result.columns:
        top_10 = result[
            "top_10_seller_share"
        ]

        valid_top_10 = top_10.notna()

        top_10_score.loc[
            valid_top_10
        ] = (
            (
                0.95
                - top_10.loc[valid_top_10]
            )
            / 0.35
            * 100.0
        ).clip(
            lower=0.0,
            upper=100.0,
        )

    components = pd.concat(
        [
            top_3_score.rename(
                "top_3_score"
            ),
            top_10_score.rename(
                "top_10_score"
            ),
        ],
        axis=1,
    )

    result["concentration_score"] = (
        top_3_score * 0.70
        + top_10_score * 0.30
    )

    no_data = (
        components.notna().sum(
            axis=1
        )
        == 0
    )

    result.loc[
        no_data,
        "concentration_score",
    ] = pd.NA

    return result
def add_opportunity_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Объединяет компоненты в Opportunity Score 0–100.

    Отсутствующие компоненты не считаются нулём.
    Их вес исключается из расчёта.

    Также показывает, какая доля полной модели
    реально была доступна для расчёта.
    """

    result = dataframe.copy()

    score_mapping = {
        "demand_score": SCORE_WEIGHTS["demand"],
        "growth_score": SCORE_WEIGHTS["growth"],
        "stability_score": SCORE_WEIGHTS["stability"],
        "competition_score": SCORE_WEIGHTS["competition"],
        "concentration_score": SCORE_WEIGHTS[
            "concentration"
        ],
    }

    weighted_sum = pd.Series(
        0.0,
        index=result.index,
    )

    available_weight = pd.Series(
        0.0,
        index=result.index,
    )

    for column, weight in score_mapping.items():
        if column not in result.columns:
            continue

        valid = result[column].notna()

        if not valid.any():
            continue

        weighted_sum.loc[valid] += (
            result.loc[valid, column]
            * weight
        )

        available_weight.loc[valid] += weight

    result["opportunity_score"] = pd.NA

    has_score = available_weight > 0

    result.loc[
        has_score,
        "opportunity_score",
    ] = (
        weighted_sum.loc[has_score]
        / available_weight.loc[has_score]
    )

    result["score_coverage"] = (
        available_weight
        / sum(SCORE_WEIGHTS.values())
        * 100.0
    )

    return result
def add_eligibility_status(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Определяет, можно ли считать кандидата
    подтвержденной рыночной возможностью.

    <= 3 активных продавцов:
    рынок считается слишком узким.

    Если данных о продавцах нет:
    статус остается неизвестным.
    """

    result = dataframe.copy()

    result["eligibility_status"] = (
        "insufficient_competition_data"
    )

    result["is_eligible"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    if "active_seller_count" not in result.columns:
        return result

    active = result["active_seller_count"]

    low_market_depth = (
        active.notna()
        & (active <= 3)
    )

    confirmed_market = (
        active.notna()
        & (active > 3)
    )

    result.loc[
        low_market_depth,
        "eligibility_status",
    ] = "rejected_low_market_depth"

    result.loc[
        low_market_depth,
        "is_eligible",
    ] = False

    result.loc[
        confirmed_market,
        "eligibility_status",
    ] = "eligible"

    result.loc[
        confirmed_market,
        "is_eligible",
    ] = True

    return result
def score_candidates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Запускает полный scoring pipeline
    для таблицы кандидатов.
    """

    result = dataframe.copy()

    result = add_demand_score(
        result
    )

    result = add_growth_score(
        result
    )

    result = add_stability_score(
        result
    )

    result = add_competition_score(
        result
    )

    result = add_concentration_score(
        result
    )

    result = add_opportunity_score(
        result
    )

    result = add_eligibility_status(
        result
    )

    return result