import re
from typing import Final


FIELD_ALIASES: Final[dict[str, set[str]]] = {
    "product_name": {
        "товар",
        "название товара",
        "наименование",
        "product",
        "product name",
    },
    "sku": {
        "sku",
        "артикул",
        "id товара",
        "product id",
    },
    "category": {
        "категория",
        "категория товара",
        "category",
    },
    "brand": {
        "бренд",
        "brand",
    },
    "seller": {
        "продавец",
        "название продавца",
        "seller",
    },
    "seller_id": {
        "id продавца",
        "seller id",
        "инн продавца",
    },
    "price": {
        "цена",
        "стоимость",
        "price",
    },
    "old_price": {
        "старая цена",
        "old price",
    },
    "discount": {
        "скидка",
        "discount",
    },
    "sales_units": {
        "продажи",
        "продажи шт",
        "продано за 30 дн шт",
        "количество продаж",
        "кол во продаж",
        "заказы",
        "units sold",
        "sales units",
        "orders",
    },
    "sales_per_day": {
    "сред продаж в день шт",
    "средняя продажа в день",
    "средние продажи в день",
    "продажи в день",
    "sales per day",
    "daily sales",
    },
    "revenue": {
        "выручка",
        "выручка за 30 дн",
        "оборот",
        "revenue",
        "sales amount",
    },
    "revenue_per_day": {
    "сред выручка в день",
    "средняя выручка в день",
    "выручка в день",
    "revenue per day",
    "daily revenue",
    },
    "stock": {
        "остаток",
        "остатки",
        "текущий остаток",
        "текущий остаток шт",
        "в наличии",
        "stock",
    },
    "reviews": {
        "отзывы",
        "количество отзывов",
        "кол во отзывов",
        "reviews",
    },
    "rating": {
        "рейтинг",
        "rating",
    },
    "days_in_stock": {
        "дней в наличии",
        "дни в наличии",
        "days in stock",
    },
    "lost_revenue": {
        "упущенная выручка",
        "lost revenue",
    },
    "first_seen": {
        "впервые",
        "first seen",
    },
    "last_seen": {
        "последний раз",
        "last seen",
    },
    "weight": {
        "вес",
        "вес г",
        "weight",
    },
    "volume": {
        "объем",
        "объем л",
        "volume",
    },
}


def normalize_column_name(column_name: str) -> str:
    """Приводит название колонки к удобному виду для сравнения."""
    value = str(column_name).strip().lower()
    value = value.replace("ё", "е")
    value = re.sub(r"[_\W]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def build_alias_index() -> dict[str, str]:
    """Создаёт индекс: нормализованное название -> canonical field."""
    index: dict[str, str] = {}

    for canonical_field, aliases in FIELD_ALIASES.items():
        all_names = aliases | {canonical_field}

        for name in all_names:
            normalized_name = normalize_column_name(name)

            existing_field = index.get(normalized_name)

            if existing_field and existing_field != canonical_field:
                raise ValueError(
                    f"Alias conflict: {name!r} maps to both "
                    f"{existing_field!r} and {canonical_field!r}"
                )

            index[normalized_name] = canonical_field

    return index


ALIAS_TO_FIELD: Final[dict[str, str]] = build_alias_index()


def match_column_name(column_name: str) -> str | None:
    """Возвращает canonical field для известной колонки."""
    normalized_name = normalize_column_name(column_name)
    return ALIAS_TO_FIELD.get(normalized_name)