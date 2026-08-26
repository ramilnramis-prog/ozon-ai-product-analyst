import pandas as pd

from app.niche_grouping import normalize_niche_text


def detect_power_source(
    product_name: object,
) -> str | None:
    """
    Определяет тип питания товара по явным признакам
    в названии.

    Возвращает только подтверждённый тип.
    Если данных недостаточно — None.
    """

    name = normalize_niche_text(product_name)

    if not name:
        return None

    if "бенз" in name:
        return "petrol"

    if (
        "аккумулятор" in name
        or "акб" in name
    ):
        return "battery"

    if "электр" in name:
        return "electric"

    return None

def detect_product_role(
    product_name: object,
) -> str | None:
    """
    Определяет явную роль товара по названию.

    Возвращает роль только при достаточно
    явном признаке. Основной товар не угадывает.
    """

    name = normalize_niche_text(product_name)

    if not name:
        return None

    if " для " not in f" {name} ":
        return None

    consumable_markers = (
        "леска",
        "корд",
    )

    spare_part_markers = (
        "нож",
        "лезвие",
    )

    accessory_markers = (
        "шланг",
        "чехол",
        "адаптер",
        "насадка",
    )

    if any(
        marker in name
        for marker in consumable_markers
    ):
        return "consumable"

    if any(
        marker in name
        for marker in spare_part_markers
    ):
        return "spare_part"

    if any(
        marker in name
        for marker in accessory_markers
    ):
        return "accessory"

    return None

def extract_leaf_category(
    category: object,
) -> str | None:
    """
    Возвращает последний уровень иерархии категории.
    """

    if pd.isna(category):
        return None

    text = str(category).strip()

    if not text:
        return None

    parts = [
        part.strip()
        for part in text.split("/")
        if part.strip()
    ]

    if not parts:
        return None

    return parts[-1]

def extract_root_category(
    category: object,
) -> str | None:
    """
    Возвращает верхний уровень иерархии категории.
    """

    if pd.isna(category):
        return None

    text = str(category).strip()

    if not text:
        return None

    parts = [
        part.strip()
        for part in text.split("/")
        if part.strip()
    ]

    if not parts:
        return None

    return parts[0]

def extract_category_depth(
    category: object,
) -> int | None:
    """
    Возвращает глубину иерархии категории.
    """

    if pd.isna(category):
        return None

    text = str(category).strip()

    if not text:
        return None

    parts = [
        part.strip()
        for part in text.split("/")
        if part.strip()
    ]

    if not parts:
        return None

    return len(parts)

def add_product_attributes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет вычисляемые признаки товара.
    """

    result = dataframe.copy()

    if "product_name" in result.columns:
        result["power_source"] = (
            result["product_name"]
            .map(detect_power_source)
            .astype("string")
        )

        result["product_role"] = (
            result["product_name"]
            .map(detect_product_role)
            .astype("string")
        )
    else:
        result["power_source"] = pd.NA
        result["product_role"] = pd.NA

    if "category" in result.columns:
        result["root_category"] = (
            result["category"]
            .map(extract_root_category)
            .astype("string")
        )

        result["leaf_category"] = (
            result["category"]
            .map(extract_leaf_category)
            .astype("string")
    )

        result["category_depth"] = (
            result["category"]
            .map(extract_category_depth)
            .astype("Int64")
        )
    else:
        result["root_category"] = pd.NA
        result["leaf_category"] = pd.NA
        result["category_depth"] = pd.NA

    return result