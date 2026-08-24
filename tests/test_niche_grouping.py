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