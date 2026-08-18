from typing import Final


# Минимум, без которого строку товара нельзя нормально идентифицировать.
REQUIRED_FIELDS: Final[set[str]] = {
    "product_name",
}


# Для аналитики должен присутствовать хотя бы один показатель спроса.
REQUIRED_METRIC_GROUPS: Final[tuple[frozenset[str], ...]] = (
    frozenset({"sales_units", "revenue"}),
)


# Полезные поля. Их отсутствие не должно ломать весь анализ.
OPTIONAL_FIELDS: Final[set[str]] = {
    "sku",
    "category",
    "brand",
    "seller",
    "seller_id",
    "price",
    "old_price",
    "discount",
    "stock",
    "reviews",
    "rating",
    "days_in_stock",
    "lost_revenue",
    "first_seen",
    "last_seen",
    "period_start",
    "period_end",
    "period_days",
    "weight",
    "volume",
    "search_demand",
}


# Эти показатели система рассчитывает сама.
DERIVED_FIELDS: Final[set[str]] = {
    "sales_per_day",
    "revenue_per_day",
}


# Технические поля для понимания происхождения данных.
SOURCE_FIELDS: Final[set[str]] = {
    "source_name",
    "source_file",
    "source_sheet",
}