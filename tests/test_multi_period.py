import pandas as pd
import pytest

import app.multi_period as multi_period


def test_combine_period_files_classifies_before_niche_grouping(
    monkeypatch,
):
    source = pd.DataFrame(
        {
            "product_name": [
                "Мотоблок Huter",
                "MATAKLA Электротяпка",
            ],
            "category": [
                (
                    "Дача и сад/"
                    "Мотоблоки, культиваторы и электротяпки"
                ),
                (
                    "Дача и сад/"
                    "Мотоблоки, культиваторы и электротяпки"
                ),
            ],
            "period_start": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-01",
                ]
            ),
        }
    )

    monkeypatch.setattr(
        multi_period,
        "run_analysis_pipeline",
        lambda file_path: {
            "success": True,
            "dataframe": source.copy(),
            "inspection": {},
        },
    )

    classification_calls = []

    def fake_classify_category_dataframe_group(
        dataframe,
        category_name,
        client,
        root_category=None,
        **kwargs,
    ):
        classification_calls.append(
            (
                root_category,
                category_name,
                client,
            )
        )

        result = dataframe.copy()

        result["functional_family"] = [
            "motor_block",
            "electric_hoe",
        ]

        result["functional_family_status"] = [
            "matched",
            "matched",
        ]

        return result

    monkeypatch.setattr(
        multi_period,
        "classify_category_dataframe_group",
        fake_classify_category_dataframe_group,
        raising=False,
    )

    competition_input = {}

    def fake_calculate_niche_competition(
        dataframe,
    ):
        competition_input["dataframe"] = (
            dataframe.copy()
        )

        return pd.DataFrame()

    monkeypatch.setattr(
        multi_period,
        "calculate_niche_competition",
        fake_calculate_niche_competition,
    )

    monkeypatch.setattr(
        multi_period,
        "calculate_product_trends",
        lambda dataframe: pd.DataFrame(),
    )

    monkeypatch.setattr(
        multi_period,
        "calculate_product_stability",
        lambda dataframe: pd.DataFrame(),
    )

    monkeypatch.setattr(
        multi_period,
        "build_candidate_features",
        lambda dataframe, trends, stability: (
            dataframe[
                ["product_key"]
            ]
            .drop_duplicates()
            .copy()
        ),
    )

    monkeypatch.setattr(
        multi_period,
        "score_candidates",
        lambda dataframe: dataframe,
    )

    monkeypatch.setattr(
        multi_period,
        "rank_candidates",
        lambda dataframe: dataframe,
    )

    monkeypatch.setattr(
        multi_period,
        "build_candidate_report",
        lambda dataframe: dataframe,
    )

    fake_client = object()

    result = multi_period.combine_period_files(
        ["fake.xlsx"],
        classification_client=fake_client,
    )

    assert classification_calls == [
        (
            "Дача и сад",
            "Мотоблоки, культиваторы и электротяпки",
            fake_client,
        )
    ]

    assert result["dataframe"][
        "functional_family"
    ].tolist() == [
        "motor_block",
        "electric_hoe",
    ]

    competition_dataframe = (
        competition_input["dataframe"]
    )

    assert competition_dataframe[
        "niche_key"
    ].tolist() == [
        (
            "family:дача и сад:"
            "мотоблоки культиваторы и электротяпки:"
            "motor block"
        ),
        (
            "family:дача и сад:"
            "мотоблоки культиваторы и электротяпки:"
            "electric hoe"
        ),
    ]

def test_combine_period_files_limits_category_classification_calls(
    monkeypatch,
):
    source = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
                "Товар C",
            ],
            "category": [
                "Дом и сад/Категория A",
                "Дом и сад/Категория B",
                "Дом и сад/Категория C",
            ],
            "period_start": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-01",
                    "2026-07-01",
                ]
            ),
        }
    )

    monkeypatch.setattr(
        multi_period,
        "run_analysis_pipeline",
        lambda file_path: {
            "success": True,
            "dataframe": source.copy(),
            "inspection": {},
        },
    )

    calls = []

    def fake_classify_category_dataframe_group(
        dataframe,
        category_name,
        client,
        root_category=None,
        **kwargs,
    ):
        calls.append(
            (
                root_category,
                category_name,
            )
        )

        result = dataframe[
            dataframe["leaf_category"].eq(
                category_name
            )
        ].copy()

        result["functional_family"] = (
            "test_family"
        )
        result["functional_family_status"] = (
            "matched"
        )

        return result

    monkeypatch.setattr(
        multi_period,
        "classify_category_dataframe_group",
        fake_classify_category_dataframe_group,
    )

    monkeypatch.setattr(
        multi_period,
        "calculate_niche_competition",
        lambda dataframe: pd.DataFrame(),
    )
    monkeypatch.setattr(
        multi_period,
        "calculate_product_trends",
        lambda dataframe: pd.DataFrame(),
    )
    monkeypatch.setattr(
        multi_period,
        "calculate_product_stability",
        lambda dataframe: pd.DataFrame(),
    )
    monkeypatch.setattr(
        multi_period,
        "build_candidate_features",
        lambda dataframe, trends, stability: (
            dataframe[
                ["product_key"]
            ]
            .drop_duplicates()
            .copy()
        ),
    )
    monkeypatch.setattr(
        multi_period,
        "score_candidates",
        lambda dataframe: dataframe,
    )
    monkeypatch.setattr(
        multi_period,
        "rank_candidates",
        lambda dataframe: dataframe,
    )
    monkeypatch.setattr(
        multi_period,
        "build_candidate_report",
        lambda dataframe: dataframe,
    )

    multi_period.combine_period_files(
        ["fake.xlsx"],
        classification_client=object(),
        classification_limit=2,
    )

    assert len(calls) == 2


def test_combine_period_files_passes_enrich_cached_to_classifier(
    monkeypatch,
):
    source = pd.DataFrame(
        {
            "product_name": [
                "Неясный товар",
            ],
            "category": [
                "Дом и сад/Категория A",
            ],
            "period_start": pd.to_datetime(
                [
                    "2026-07-01",
                ]
            ),
        }
    )

    monkeypatch.setattr(
        multi_period,
        "run_analysis_pipeline",
        lambda file_path: {
            "success": True,
            "dataframe": source.copy(),
            "inspection": {},
        },
    )

    monkeypatch.setattr(
        multi_period,
        "get_cached_category_classification",
        lambda *args, **kwargs: object(),
    )

    received = []

    def fake_classify_category_dataframe_group(
        dataframe,
        category_name,
        client,
        root_category=None,
        enrich_cached=False,
        **kwargs,
    ):
        received.append(enrich_cached)

        raise RuntimeError(
            "stop after classifier"
        )

    monkeypatch.setattr(
        multi_period,
        "classify_category_dataframe_group",
        fake_classify_category_dataframe_group,
    )

    with pytest.raises(
        RuntimeError,
        match="stop after classifier",
    ):
        multi_period.combine_period_files(
            ["fake.xlsx"],
            classification_client=object(),
            enrich_cached=True,
        )

    assert received == [
        True,
    ]

def test_classification_limit_counts_only_uncached_categories(
    monkeypatch,
):
    source = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
                "Товар C",
            ],
            "category": [
                "Дом и сад/Категория A",
                "Дом и сад/Категория B",
                "Дом и сад/Категория C",
            ],
            "period_start": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-01",
                    "2026-07-01",
                ]
            ),
        }
    )

    monkeypatch.setattr(
        multi_period,
        "run_analysis_pipeline",
        lambda file_path: {
            "success": True,
            "dataframe": source.copy(),
            "inspection": {},
        },
    )

    calls = []

    def fake_get_cached_category_classification(
        category_name,
        *args,
        root_category=None,
        **kwargs,
    ):
        if category_name == "Категория A":
            return object()

        return None

    monkeypatch.setattr(
        multi_period,
        "get_cached_category_classification",
        fake_get_cached_category_classification,
        raising=False,
    )

    def fake_classify_category_dataframe_group(
        dataframe,
        category_name,
        client,
        root_category=None,
        **kwargs,
    ):
        calls.append(category_name)

        mask = (
            dataframe["leaf_category"]
            .eq(category_name)
        )

        result = dataframe.loc[
            mask
        ].copy()

        result["functional_family"] = (
            "test_family"
        )
        result["functional_family_status"] = (
            "matched"
        )

        return result

    monkeypatch.setattr(
        multi_period,
        "classify_category_dataframe_group",
        fake_classify_category_dataframe_group,
    )

    monkeypatch.setattr(
        multi_period,
        "calculate_niche_competition",
        lambda dataframe: pd.DataFrame(),
    )
    monkeypatch.setattr(
        multi_period,
        "calculate_product_trends",
        lambda dataframe: pd.DataFrame(),
    )
    monkeypatch.setattr(
        multi_period,
        "calculate_product_stability",
        lambda dataframe: pd.DataFrame(),
    )
    monkeypatch.setattr(
        multi_period,
        "build_candidate_features",
        lambda dataframe, trends, stability: (
            dataframe[
                ["product_key"]
            ]
            .drop_duplicates()
            .copy()
        ),
    )
    monkeypatch.setattr(
        multi_period,
        "score_candidates",
        lambda dataframe: dataframe,
    )
    monkeypatch.setattr(
        multi_period,
        "rank_candidates",
        lambda dataframe: dataframe,
    )
    monkeypatch.setattr(
        multi_period,
        "build_candidate_report",
        lambda dataframe: dataframe,
    )

    multi_period.combine_period_files(
        ["fake.xlsx"],
        classification_client=object(),
        classification_limit=1,
    )

    assert calls == [
        "Категория A",
        "Категория B",
    ]