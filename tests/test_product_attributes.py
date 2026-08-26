from app.product_attributes import (
    add_product_attributes,
    detect_power_source,
    extract_category_depth,
    detect_product_role,
    extract_leaf_category,
    extract_root_category,
)


def test_detects_petrol_before_electric_substring():
    name = (
        "Триммер бензиновый NOCORD NTG-52SDE, "
        "52 см3, электростарт"
    )

    assert detect_power_source(name) == "petrol"


def test_detects_battery_before_electric_wording():
    name = (
        "Триммер аккумуляторный садовый, "
        "газонокосилка электрическая, "
        "с двумя аккумуляторами"
    )

    assert detect_power_source(name) == "battery"


def test_detects_electric_product():
    name = "Триммер садовый электрический 1700 Вт"

    assert detect_power_source(name) == "electric"


def test_returns_none_when_power_source_is_unknown():
    name = "Бесщеточный триммер садовый с колесами"

    assert detect_power_source(name) is None


def test_returns_none_for_empty_value():
    assert detect_power_source(None) is None

def test_detect_product_role_consumable():
    assert (
        detect_product_role(
            "Леска для триммера 3 мм 30 м"
        )
        == "consumable"
    )


def test_detect_product_role_spare_part():
    assert (
        detect_product_role(
            "Нож для газонокосилки Champion"
        )
        == "spare_part"
    )


def test_detect_product_role_accessory():
    assert (
        detect_product_role(
            "Шланг для мойки высокого давления 10 метров"
        )
        == "accessory"
    )


def test_detect_product_role_does_not_guess_main_product():
    assert (
        detect_product_role(
            "Триммер аккумуляторный садовый"
        )
        is None
    )


def test_detect_product_role_handles_empty_value():
    assert detect_product_role(None) is None

import pandas as pd

from app.product_attributes import add_product_attributes

def test_add_product_attributes_adds_power_source():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Триммер бензиновый с электростартом",
                "Триммер аккумуляторный электрический",
                "Триммер садовый без явного типа",
            ],
            "category": [
                "Дом и сад / Дача и сад / Садовая техника / Триммеры",
                "Дом и сад / Дача и сад / Садовая техника / Триммеры",
                "Дом и сад / Дача и сад / Садовая техника / Триммеры",
            ],
        }
    )

    result = add_product_attributes(dataframe)

    assert result["root_category"].tolist() == [
    "Дом и сад",
    "Дом и сад",
    "Дом и сад",
    ]
    assert result["power_source"].tolist() == [
        "petrol",
        "battery",
        pd.NA,
    ]

    assert result["leaf_category"].tolist() == [
        "Триммеры",
        "Триммеры",
        "Триммеры",
    ]

    assert result["category_depth"].tolist() == [
    4,
    4,
    4,
    ]

def test_add_product_attributes_handles_missing_product_name():
    dataframe = pd.DataFrame(
        {
            "price": [1000, 1200]
        }
    )

    result = add_product_attributes(dataframe)

    assert result["power_source"].isna().all()
    assert result["leaf_category"].isna().all()
    assert result["category_depth"].isna().all()
    assert result["root_category"].isna().all()

def test_extract_leaf_category_from_hierarchy():
    category = (
        "Дом и сад / Дача и сад / "
        "Садовая техника / Триммеры"
    )

    assert extract_leaf_category(category) == "Триммеры"


def test_extract_leaf_category_handles_single_level():
    assert extract_leaf_category("Садовая техника") == "Садовая техника"


def test_extract_leaf_category_handles_empty_value():
    assert extract_leaf_category(None) is None
    assert extract_leaf_category("") is None

def test_extract_category_depth_from_hierarchy():
    category = (
        "Дом и сад / Дача и сад / "
        "Садовая техника / Триммеры"
    )

    assert extract_category_depth(category) == 4


def test_extract_category_depth_handles_single_level():
    assert extract_category_depth("Дом и сад") == 1


def test_extract_category_depth_handles_empty_value():
    assert extract_category_depth(None) is None
    assert extract_category_depth("") is None

def test_extract_root_category_from_hierarchy():
    category = (
        "Дом и сад / Дача и сад / "
        "Садовая техника / Триммеры"
    )

    assert extract_root_category(category) == "Дом и сад"


def test_extract_root_category_handles_single_level():
    assert extract_root_category("Автотовары") == "Автотовары"


def test_extract_root_category_handles_empty_value():
    assert extract_root_category(None) is None
    assert extract_root_category("") is None

def test_add_product_attributes_adds_product_role():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Леска для триммера 3 мм",
                "Нож для газонокосилки Champion",
                "Шланг для мойки высокого давления",
                "Триммер аккумуляторный садовый",
            ]
        }
    )

    result = add_product_attributes(dataframe)

    assert (
        result["product_role"]
        .astype("string")
        .fillna("unknown")
        .tolist()
    ) == [
        "consumable",
        "spare_part",
        "accessory",
        "unknown",
    ]