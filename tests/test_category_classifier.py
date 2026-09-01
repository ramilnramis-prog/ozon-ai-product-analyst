from pathlib import Path

import pandas as pd

from app.category_classifier import (
    CategoryClassification,
    FunctionalFamilyRule,
    get_family_match_status,
    remove_brand_keywords,
    apply_family_resolutions,
    apply_category_classification,
    match_product_to_category_classification,
    match_functional_families,
    normalize_brand_match_text,
    category_cache_key,
    category_needs_ai,
    get_cached_category_classification,
    load_category_cache,
    save_category_classification,
)


def test_category_cache_key_normalizes_category_name():
    assert category_cache_key(
        " ТРИММЕРЫ "
    ) == "триммеры"

def test_category_cache_key_uses_root_category_context():
    first_key = category_cache_key(
        "Аксессуары",
        root_category="Автотовары",
    )

    second_key = category_cache_key(
        "Аксессуары",
        root_category="Дом и сад",
    )

    assert first_key == (
        "автотовары:аксессуары"
    )
    assert second_key == (
        "дом и сад:аксессуары"
    )
    assert first_key != second_key

def test_category_needs_ai_only_for_unknown_category():
    known_category_keys = {
        "триммеры",
        "газонокосилки",
    }

    assert category_needs_ai(
        "Триммеры",
        known_category_keys,
    ) is False

    assert category_needs_ai(
        "Снегоуборщики и электролопаты",
        known_category_keys,
    ) is True

    assert category_needs_ai(
        None,
        known_category_keys,
    ) is False

def test_load_category_cache_reads_json(tmp_path: Path):
    cache_path = tmp_path / "categories.json"

    cache_path.write_text(
        """
        {
          "триммеры": {
            "category_type": "homogeneous",
            "functional_families": ["trimmer"],
            "confidence": 0.95
          }
        }
        """,
        encoding="utf-8",
    )

    result = load_category_cache(cache_path)

    assert result["триммеры"]["category_type"] == "homogeneous"
    assert result["триммеры"]["functional_families"] == [
        "trimmer",
    ]


def test_load_category_cache_returns_empty_for_missing_file(
    tmp_path: Path,
):
    cache_path = tmp_path / "missing.json"

    result = load_category_cache(cache_path)

    assert result == {}

def test_save_category_classification_round_trip(
    tmp_path: Path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Триммеры",
        category_type="homogeneous",
        functional_families=(
            FunctionalFamilyRule(
                name="trimmer",
                keywords=("триммер",),
            ),
        ),
        confidence=0.95,
    )

    save_category_classification(
        classification,
        cache_path,
    )

    result = load_category_cache(
        cache_path
    )

    assert "триммеры" in result
    assert result["триммеры"] == {
        "category_type": "homogeneous",
        "functional_families": [
            {
                "name": "trimmer",
                "keywords": [
                    "триммер",
                ],
            },
        ],
        "confidence": 0.95,
    }

def test_get_cached_category_classification_hit_and_miss(
    tmp_path: Path,
):
    cache_path = tmp_path / "categories.json"

    save_category_classification(
        CategoryClassification(
            category_name="Триммеры",
            category_type="homogeneous",
            functional_families=(
                FunctionalFamilyRule(
                    name="trimmer",
                    keywords=("триммер",),
                ),
            ),
            confidence=0.95,
        ),
        cache_path,
    )

    cached = get_cached_category_classification(
        " ТРИММЕРЫ ",
        cache_path,
    )

    missing = get_cached_category_classification(
        "Снегоуборщики и электролопаты",
        cache_path,
    )

    assert cached is not None
    assert cached.category_type == "homogeneous"
    assert cached.functional_families == (
        FunctionalFamilyRule(
            name="trimmer",
            keywords=("триммер",),
        ),
)
    assert cached.confidence == 0.95

    assert missing is None

def test_match_functional_families_handles_match_ambiguous_and_unmatched():
    rules = (
        FunctionalFamilyRule(
            name="walk_behind_tractor",
            keywords=("мотоблок",),
        ),
        FunctionalFamilyRule(
            name="cultivator",
            keywords=("культиватор",),
        ),
        FunctionalFamilyRule(
            name="electric_tiller",
            keywords=("электротяпка",),
        ),
    )

    assert match_functional_families(
        "Мотоблок HUTER 7 л.с.",
        rules,
    ) == (
        "walk_behind_tractor",
    )

    assert match_functional_families(
        "Культиватор аккумуляторный, электротяпка",
        rules,
    ) == (
        "cultivator",
        "electric_tiller",
    )

    assert match_functional_families(
        "Садовая техника неизвестного типа",
        rules,
    ) == ()

def test_get_family_match_status():
    assert get_family_match_status(
        ("cultivator",)
    ) == "matched"

    assert get_family_match_status(
        (
            "cultivator",
            "electric_tiller",
        )
    ) == "ambiguous"

    assert get_family_match_status(
        ()
    ) == "unmatched"

def test_match_product_to_category_classification():
    homogeneous = CategoryClassification(
        category_name="Триммеры",
        category_type="homogeneous",
        functional_families=(
            FunctionalFamilyRule(
                name="trimmer",
                keywords=("триммер",),
            ),
        ),
        confidence=0.95,
    )

    matches, status = (
        match_product_to_category_classification(
            "Совершенно любое название товара",
            homogeneous,
        )
    )

    assert matches == ("trimmer",)
    assert status == "matched"

    mixed = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="walk_behind_tractor",
                keywords=("мотоблок",),
            ),
            FunctionalFamilyRule(
                name="cultivator",
                keywords=("культиватор",),
            ),
            FunctionalFamilyRule(
                name="electric_tiller",
                keywords=("электротяпка",),
            ),
        ),
        confidence=0.9,
    )

    matches, status = (
        match_product_to_category_classification(
            "Культиватор аккумуляторный, электротяпка",
            mixed,
        )
    )

    assert matches == ("cultivator",)
    assert status == "matched"

def test_apply_category_classification_to_dataframe():
    classification = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="walk_behind_tractor",
                keywords=("мотоблок",),
            ),
            FunctionalFamilyRule(
                name="cultivator",
                keywords=("культиватор",),
            ),
            FunctionalFamilyRule(
                name="electric_tiller",
                keywords=("электротяпка",),
            ),
        ),
        confidence=0.9,
    )

    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Мотоблок HUTER",
                "Культиватор аккумуляторный, электротяпка",
                "Садовая техника неизвестного типа",
            ]
        }
    )

    result = apply_category_classification(
        dataframe,
        classification,
    )

    assert result[
        "functional_family_status"
    ].tolist() == [
        "matched",
        "matched",
        "unmatched",
    ]

    assert result.loc[
        0,
        "functional_family",
    ] == "walk_behind_tractor"

    assert result.loc[
        1,
        "functional_family",
    ] == "cultivator"

    assert pd.isna(
        result.loc[
            2,
            "functional_family",
        ]
    )

def test_apply_family_resolutions_updates_only_unresolved_rows():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Мотоблок HUTER",
                "Культиватор аккумуляторный, электротяпка",
                "Неясный товар",
            ],
            "functional_family": [
                "walk_behind_tractor",
                pd.NA,
                pd.NA,
            ],
            "functional_family_status": [
                "matched",
                "ambiguous",
                "unmatched",
            ],
        }
    )

    resolutions = [
        {
            "product_name": (
                "Культиватор аккумуляторный, электротяпка"
            ),
            "family_name": "electric_tiller",
            "confidence": 0.92,
        },
        {
            "product_name": "Неясный товар",
            "family_name": "unresolved",
            "confidence": 0.35,
        },
    ]

    result = apply_family_resolutions(
        dataframe,
        resolutions,
    )

    assert result.loc[
        0,
        "functional_family",
    ] == "walk_behind_tractor"

    assert result.loc[
        0,
        "functional_family_status",
    ] == "matched"

    assert result.loc[
        1,
        "functional_family",
    ] == "electric_tiller"

    assert result.loc[
        1,
        "functional_family_status",
    ] == "ai_resolved"

    assert pd.isna(
        result.loc[
            2,
            "functional_family",
        ]
    )

    assert result.loc[
        2,
        "functional_family_status",
    ] == "unresolved"

def test_remove_brand_keywords_keeps_product_terms():
    classification = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="motor_block",
                keywords=(
                    "мотоблок",
                    "motor_block",
                    "BoxBot",
                    "Zitrek",
                ),
            ),
        ),
        confidence=0.8,
    )

    result = remove_brand_keywords(
        classification,
        brand_values=[
            "BoxBot",
            "Huter",
            "DEKO",
        ],
    )

    assert result.functional_families[0].keywords == (
        "мотоблок",
        "motor_block",
        "Zitrek",
    )

def test_normalize_brand_match_text_matches_cyrillic_and_latin():
    assert normalize_brand_match_text(
        "MATAKLA"
    ) == "matakla"

    assert normalize_brand_match_text(
        "матакла"
    ) == "matakla"

def test_remove_brand_keywords_matches_transliterated_brand():
    classification = CategoryClassification(
        category_name="Мотоблоки, культиваторы и электротяпки",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="electric_hoe",
                keywords=(
                    "электротяпка",
                    "матакла",
                ),
            ),
        ),
        confidence=0.8,
    )

    cleaned = remove_brand_keywords(
        classification,
        brand_values=["MATAKLA"],
    )

    assert cleaned.functional_families[
        0
    ].keywords == (
        "электротяпка",
    )

def test_category_cache_round_trip_uses_root_category_context(
    tmp_path: Path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Аксессуары",
        category_type="homogeneous",
        functional_families=(
            FunctionalFamilyRule(
                name="accessory",
                keywords=("аксессуар",),
            ),
        ),
        confidence=0.9,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Автотовары",
    )

    cached = get_cached_category_classification(
        "Аксессуары",
        cache_path,
        root_category="Автотовары",
    )

    wrong_root = get_cached_category_classification(
        "Аксессуары",
        cache_path,
        root_category="Дом и сад",
    )

    assert cached is not None
    assert wrong_root is None