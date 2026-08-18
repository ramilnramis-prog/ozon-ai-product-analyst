import pandas as pd

from app.column_inspector import inspect_columns


def normalize_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Приводит известные колонки входной таблицы
    к canonical schema проекта.
    """

    inspection = inspect_columns(
        list(dataframe.columns)
    )

    if not inspection["is_valid"]:
        raise ValueError(
            "Input data does not contain "
            "the minimum required fields."
        )

    duplicate_mappings = inspection[
        "duplicate_mappings"
    ]

    if duplicate_mappings:
        raise ValueError(
            "Multiple source columns map to "
            f"the same canonical field: "
            f"{duplicate_mappings}"
        )

    column_mapping = inspection["mapped"]

    normalized = dataframe.rename(
        columns=column_mapping
    ).copy()

    return normalized