import re

import pandas as pd


def normalize_niche_text(value: object) -> str:
    """
    Нормализует текст для построения ключа ниши.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("ё", "е")

    text = re.sub(
        r"[^a-zа-я0-9]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return " ".join(text.split())


def add_niche_key(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет niche_key.

    Приоритет:
    1. готовый niche_key, если источник его уже содержит;
    2. категория как широкий рыночный контекст;
    3. название товара как безопасный fallback.

    Также сохраняет источник и уверенность группировки.
    """

    result = dataframe.copy()

    if "niche_key" not in result.columns:
        result["niche_key"] = ""

    result["niche_key_source"] = ""
    result["niche_key_confidence"] = ""

    existing_key = (
        result["niche_key"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    has_existing_key = existing_key != ""

    result.loc[
        has_existing_key,
        "niche_key",
    ] = existing_key[has_existing_key]

    result.loc[
        has_existing_key,
        "niche_key_source",
    ] = "provided"

    result.loc[
        has_existing_key,
        "niche_key_confidence",
    ] = "high"

    missing_key = (
        result["niche_key"] == ""
    )

    if "category" in result.columns:
        normalized_category = result[
            "category"
        ].map(normalize_niche_text)

        use_category = (
            missing_key
            & (normalized_category != "")
        )

        result.loc[
            use_category,
            "niche_key",
        ] = (
            "category:"
            + normalized_category[use_category]
        )

        result.loc[
            use_category,
            "niche_key_source",
        ] = "category"

        result.loc[
            use_category,
            "niche_key_confidence",
        ] = "low"

    missing_key = (
        result["niche_key"] == ""
    )

    if "product_name" in result.columns:
        normalized_name = result[
            "product_name"
        ].map(normalize_niche_text)

        use_name = (
            missing_key
            & (normalized_name != "")
        )

        result.loc[
            use_name,
            "niche_key",
        ] = (
            "product:"
            + normalized_name[use_name]
        )

        result.loc[
            use_name,
            "niche_key_source",
        ] = "product_name"

        result.loc[
            use_name,
            "niche_key_confidence",
        ] = "very_low"

    return result