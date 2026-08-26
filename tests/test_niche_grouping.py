import pandas as pd

from app.niche_grouping import add_niche_key


def test_niche_key_uses_functional_family_with_category_context():
    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дача и сад",
                "Дача и сад",
            ],
            "leaf_category": [
                "Мотоблоки, культиваторы и электротяпки",
                "Мотоблоки, культиваторы и электротяпки",
            ],
            "functional_family": [
                "cultivator",
                "electric_hoe",
            ],
        }
    )

    result = add_niche_key(dataframe)

    assert result["niche_key"].tolist() == [
        (
            "family:дача и сад:"
            "мотоблоки культиваторы и электротяпки:"
            "cultivator"
        ),
        (
            "family:дача и сад:"
            "мотоблоки культиваторы и электротяпки:"
            "electric hoe"
        ),
    ]

    assert result["niche_key_source"].tolist() == [
        "functional_family",
        "functional_family",
    ]

def test_niche_key_appends_product_role_to_functional_family():
    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
            ],
            "leaf_category": [
                "Садовая техника",
            ],
            "functional_family": [
                "string_trimmer",
            ],
            "product_role": [
                "consumable",
            ],
        }
    )

    result = add_niche_key(dataframe)

    assert result["niche_key"].tolist() == [
        (
            "family:дом и сад:"
            "садовая техника:"
            "string trimmer:"
            "role:consumable"
        ),
    ]

    assert result["niche_key_source"].tolist() == [
        "functional_family",
    ]

    assert result["niche_key_confidence"].tolist() == [
        "high",
    ]

def test_niche_key_uses_category_context_when_family_is_missing():
    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дача и сад",
            ],
            "leaf_category": [
                "Мотоблоки, культиваторы и электротяпки",
            ],
            "functional_family": [
                None,
            ],
        }
    )

    result = add_niche_key(dataframe)

    assert result["niche_key"].tolist() == [
        (
            "category:дача и сад:"
            "мотоблоки культиваторы и электротяпки"
        ),
    ]

    assert result["niche_key_source"].tolist() == [
        "category_context",
    ]

    assert result["niche_key_confidence"].tolist() == [
        "medium",
    ]