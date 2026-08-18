import re

import pandas as pd


EMPTY_VALUES = {
    "",
    "-",
    "—",
    "–",
    "нет",
    "н/д",
    "nan",
    "none",
}

NUMERIC_FIELDS = {
    "sales_per_day",
    "revenue_per_day",
    "sales_units",
    "revenue",
    "price",
    "old_price",
    "discount",
    "stock",
    "reviews",
    "rating",
    "days_in_stock",
    "lost_revenue",
    "period_days",
    "weight",
    "volume",
    "search_demand",
}


def clean_numeric_value(value: object) -> float | None:
    """
    Преобразует значение в число, если это возможно.

    Примеры:
    "1 290 ₽" -> 1290.0
    "45 шт."  -> 45.0
    "12,5"    -> 12.5
    "—"       -> None
    """

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()

    if text in EMPTY_VALUES:
        return None

    text = text.replace("\u00a0", "")
    text = text.replace(" ", "")

    text = re.sub(
        r"[^0-9,.\-]",
        "",
        text,
    )

    if not text:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    elif "," in text:
        parts = text.split(",")

        if len(parts) == 2 and len(parts[1]) <= 2:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    try:
        return float(text)

    except ValueError:
        return None

def clean_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Очищает известные числовые колонки canonical schema.

    Неизвестные и текстовые колонки остаются без изменений.
    """

    cleaned = dataframe.copy()

    for column in cleaned.columns:
        if column not in NUMERIC_FIELDS:
            continue

        cleaned[column] = cleaned[column].map(
            clean_numeric_value
        )

    return cleaned