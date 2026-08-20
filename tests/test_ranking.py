import pandas as pd

from app.ranking import rank_candidates


def test_rank_candidates_adds_category_rank():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
                "Товар C",
                "Товар D",
            ],
            "root_category": [
                "Дом и сад",
                "Автотовары",
                "Дом и сад",
                "Дом и сад",
            ],
            "eligibility_status": [
                "eligible",
                "eligible",
                "eligible",
                "eligible",
            ],
            "opportunity_score": [
                95.0,
                90.0,
                85.0,
                80.0,
            ],
        }
    )

    result = rank_candidates(dataframe)

    assert result["opportunity_rank"].tolist() == [
        1,
        2,
        3,
        4,
    ]

    assert result["category_rank"].tolist() == [
        1,
        1,
        2,
        3,
    ]


def test_rank_candidates_leaves_category_rank_empty_without_root_category():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
            ],
            "eligibility_status": [
                "eligible",
                "eligible",
            ],
            "opportunity_score": [
                90.0,
                80.0,
            ],
        }
    )

    result = rank_candidates(dataframe)

    assert result["category_rank"].isna().all()

def test_category_rank_uses_category_opportunity_score():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
                "Товар C",
            ],
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Автотовары",
            ],
            "eligibility_status": [
                "eligible",
                "eligible",
                "eligible",
            ],
            "opportunity_score": [
                95.0,
                90.0,
                80.0,
            ],
            "category_opportunity_score": [
                60.0,
                100.0,
                100.0,
            ],
        }
    )

    result = rank_candidates(dataframe)

    assert result["opportunity_rank"].tolist() == [
        1,
        2,
        3,
    ]

    assert result["category_rank"].tolist() == [
        2,
        1,
        1,
    ]
def test_category_rank_uses_dense_rank_for_tied_scores():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
                "Товар C",
            ],
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Дом и сад",
            ],
            "eligibility_status": [
                "eligible",
                "eligible",
                "eligible",
            ],
            "opportunity_score": [
                95.0,
                90.0,
                85.0,
            ],
            "category_opportunity_score": [
                60.104,
                60.101,
                55.0,
            ],
        }
    )

    result = rank_candidates(dataframe)

    assert result["category_rank"].tolist() == [
        1,
        1,
        2,
    ]