import json
import pandas as pd

from types import SimpleNamespace

from app.category_classifier import (
    CategoryClassification,
    FunctionalFamilyRule,
    save_category_classification,
)

from app.category_classification_ai import (
    build_category_classification_prompt,
    generate_category_repair,
    collect_category_examples,
    classify_category_dataframe_group,
    build_category_repair_prompt,
    build_family_resolution_prompt,
    classify_category,
    generate_category_classification,
    generate_family_resolutions,
)

def reject_candidate_validation_if_requested(kwargs):
    prompt = kwargs.get("input", "")

    if "CANDIDATE FAMILY VALIDATION ONLY" not in prompt:
        return None

    return SimpleNamespace(
        output_text=json.dumps(
            {
                "accept": False,
                "family_name": "",
                "keywords": [],
            },
            ensure_ascii=False,
        )
    )


def empty_discovery_response_if_requested(kwargs):
    prompt = kwargs.get("input", "")

    if "MISSING FAMILY DISCOVERY ONLY" not in prompt:
        return None

    return SimpleNamespace(
        output_text=json.dumps(
            {
                "new_families": [],
            },
            ensure_ascii=False,
        )
    )

def test_build_category_classification_prompt_deduplicates_examples():
    prompt = build_category_classification_prompt(
        " Мотоблоки, культиваторы и электротяпки ",
        [
            "Мотоблок HUTER",
            "Мотоблок HUTER",
            "Культиватор DAEWOO",
            "",
            None,
            "Электротяпка аккумуляторная",
        ],
        max_examples=3,
    )

    assert "мотоблоки культиваторы и электротяпки" in prompt

    assert prompt.count(
        "мотоблок huter"
    ) == 1

    assert "культиватор daewoo" in prompt
    assert "электротяпка аккумуляторная" in prompt


def test_build_category_classification_prompt_limits_examples():
    prompt = build_category_classification_prompt(
        "Триммеры",
        [
            "Товар 1",
            "Товар 2",
            "Товар 3",
        ],
        max_examples=2,
    )

    assert "товар 1" in prompt
    assert "товар 2" in prompt
    assert "товар 3" not in prompt

def test_generate_category_classification_parses_structured_output():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "motoblock",
                                "keywords": [
                                    "мотоблок",
                                ],
                            },
                            {
                                "name": "cultivator",
                                "keywords": [
                                    "культиватор",
                                    "мотокультиватор",
                                ],
                            },
                        ],
                        "confidence": 0.96,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_category_classification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        product_examples=[
            "Мотоблок HUTER",
            "Культиватор DAEWOO",
        ],
        client=fake_client,
    )

    assert result.category_type == "mixed"
    assert result.confidence == 0.96

    assert result.functional_families[0].name == (
        "motoblock"
    )
    assert result.functional_families[0].keywords == (
        "мотоблок",
    )

    assert result.functional_families[1].name == (
        "cultivator"
    )
    assert result.functional_families[1].keywords == (
        "культиватор",
        "мотокультиватор",
    )
def test_classify_category_uses_cache_without_ai(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    cache_path.write_text(
        """
        {
          "триммеры": {
            "category_type": "homogeneous",
            "functional_families": [
              {
                "name": "trimmer",
                "keywords": ["триммер"]
              }
            ],
            "confidence": 0.95
          }
        }
        """,
        encoding="utf-8",
    )

    class FakeResponses:
        def create(self, **kwargs):
            raise AssertionError(
                "AI should not be called for cached category"
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = classify_category(
        category_name="Триммеры",
        product_examples=[
            "Триммер аккумуляторный",
        ],
        client=fake_client,
        cache_path=cache_path,
    )

    assert result.category_type == "homogeneous"
    assert result.functional_families[0].name == "trimmer"
    assert result.confidence == 0.95


def test_classify_category_calls_ai_and_saves_cache(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "motoblock",
                                "keywords": [
                                    "мотоблок",
                                ],
                            },
                            {
                                "name": "cultivator",
                                "keywords": [
                                    "культиватор",
                                ],
                            },
                        ],
                        "confidence": 0.94,
                    },
                    ensure_ascii=False,
                )
            )

    fake_responses = FakeResponses()

    fake_client = SimpleNamespace(
        responses=fake_responses
    )

    result = classify_category(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        product_examples=[
            "Мотоблок HUTER",
            "Культиватор DAEWOO",
        ],
        client=fake_client,
        cache_path=cache_path,
    )

    assert fake_responses.call_count == 1
    assert result.category_type == "mixed"
    assert cache_path.exists()

    saved_text = cache_path.read_text(
        encoding="utf-8",
    )

    assert "мотоблоки культиваторы и электротяпки" in saved_text
    assert "motoblock" in saved_text
    assert "cultivator" in saved_text

def test_build_family_resolution_prompt_uses_allowed_families_and_deduplicates():
    prompt = build_family_resolution_prompt(
        product_names=[
            "Культиватор аккумуляторный, электротяпка",
            "Культиватор аккумуляторный, электротяпка",
            "Мотоблок бензиновый, культиватор",
        ],
        allowed_families=(
            "walk_behind_tractor",
            "cultivator",
            "electric_tiller",
        ),
    )

    assert "walk_behind_tractor" in prompt
    assert "cultivator" in prompt
    assert "electric_tiller" in prompt
    assert "unresolved" in prompt

    assert prompt.count(
        "культиватор аккумуляторный электротяпка"
    ) == 1

    assert "мотоблок бензиновый культиватор" in prompt

def test_family_resolution_prompt_includes_existing_family_rules():
    family_rules = (
        FunctionalFamilyRule(
            name="surface_pump",
            keywords=(
                "насос поверхностный",
                "поверхностный насос",
                "насос для воды",
            ),
        ),
        FunctionalFamilyRule(
            name="pump_station",
            keywords=(
                "насосная станция",
                "станция водоснабжения",
            ),
        ),
    )

    prompt = build_family_resolution_prompt(
        product_names=[
            "Автономная станция водоснабжения ASV 4200P",
        ],
        allowed_families=(
            "surface_pump",
            "pump_station",
        ),
        family_rules=family_rules,
    )

    assert "FAMILY DEFINITIONS" in prompt

    assert '"name": "surface_pump"' in prompt
    assert '"насос поверхностный"' in prompt
    assert '"поверхностный насос"' in prompt

    assert '"name": "pump_station"' in prompt
    assert '"насосная станция"' in prompt
    assert '"станция водоснабжения"' in prompt

def test_generate_family_resolutions_parses_batch_output():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": (
                                    "Культиватор аккумуляторный, "
                                    "электротяпка"
                                ),
                                "family_name": "cultivator",
                                "confidence": 0.88,
                            },
                            {
                                "product_name": (
                                    "Мотоблок бензиновый, "
                                    "культиватор"
                                ),
                                "family_name": "walk_behind_tractor",
                                "confidence": 0.91,
                            },
                            {
                                "product_name": "Неясный товар",
                                "family_name": "unresolved",
                                "confidence": 0.35,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_family_resolutions(
        product_names=[
            "Культиватор аккумуляторный, электротяпка",
            "Мотоблок бензиновый, культиватор",
            "Неясный товар",
        ],
        allowed_families=(
            "walk_behind_tractor",
            "cultivator",
            "electric_tiller",
        ),
        client=fake_client,
    )

    assert result == [
        {
            "product_name": "культиватор аккумуляторный электротяпка",
            "family_name": "cultivator",
            "confidence": 0.88,
        },
        {
            "product_name": "мотоблок бензиновый культиватор",
            "family_name": "walk_behind_tractor",
            "confidence": 0.91,
        },
        {
            "product_name": "неясный товар",
            "family_name": "unresolved",
            "confidence": 0.35,
        },
    ]

import pytest

def test_generate_family_resolutions_rejects_unknown_family():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": "Некий товар",
                                "family_name": "invented_family",
                                "confidence": 0.9,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    with pytest.raises(
        ValueError,
        match="Unexpected functional family",
    ):
        generate_family_resolutions(
            product_names=[
                "Некий товар",
            ],
            allowed_families=(
                "walk_behind_tractor",
                "cultivator",
                "electric_tiller",
            ),
            client=fake_client,
        )


def test_generate_family_resolutions_collapses_duplicate_same_family():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": "Товар A",
                                "family_name": "cultivator",
                                "confidence": 0.9,
                            },
                            {
                                "product_name": "Товар A",
                                "family_name": "cultivator",
                                "confidence": 0.85,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_family_resolutions(
        product_names=[
            "Товар A",
        ],
        allowed_families=(
            "cultivator",
            "walk_behind_tractor",
        ),
        client=fake_client,
    )

    assert result == [
        {
            "product_name": "товар a",
            "family_name": "cultivator",
            "confidence": 0.9,
        }
    ]


def test_generate_family_resolutions_conflicting_duplicate_becomes_unresolved():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": "Товар A",
                                "family_name": "cultivator",
                                "confidence": 0.9,
                            },
                            {
                                "product_name": "Товар A",
                                "family_name": "walk_behind_tractor",
                                "confidence": 0.85,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_family_resolutions(
        product_names=[
            "Товар A",
        ],
        allowed_families=(
            "cultivator",
            "walk_behind_tractor",
        ),
        client=fake_client,
    )

    assert result == [
        {
            "product_name": "товар a",
            "family_name": "unresolved",
            "confidence": 0.0,
        }
    ]

def test_generate_family_resolutions_rejects_unexpected_product():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": "Товар A",
                                "family_name": "cultivator",
                                "confidence": 0.9,
                            },
                            {
                                "product_name": "Выдуманный товар",
                                "family_name": "cultivator",
                                "confidence": 0.9,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    with pytest.raises(
        ValueError,
        match="AI returned unexpected product",
    ):
        generate_family_resolutions(
            product_names=[
                "Товар A",
                "Товар B",
            ],
            allowed_families=(
                "cultivator",
                "walk_behind_tractor",
            ),
            client=fake_client,
        )

def test_collect_category_examples_filters_category_and_limits():
    dataframe = pd.DataFrame(
        {
            "leaf_category": [
                "Триммеры",
                " ТРИММЕРЫ ",
                "Триммеры",
                "Газонокосилки",
            ],
            "product_name": [
                "Триммер A",
                "Триммер B",
                "Триммер C",
                "Газонокосилка A",
            ],
        }
    )

    result = collect_category_examples(
        dataframe,
        leaf_category="триммеры",
        max_examples=2,
    )

    assert result == [
        "Триммер A",
        "Триммер B",
    ]

def test_classify_category_dataframe_group_full_flow(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    dataframe = pd.DataFrame(
        {
            "leaf_category": [
                "Мотоблоки, культиваторы и электротяпки",
                "Мотоблоки, культиваторы и электротяпки",
                "Триммеры",
            ],
            "product_name": [
                "Мотоблок BoxBot",
                "MATAKLA Электротяпка",
                "Триммер аккумуляторный",
            ],
            "brand": [
                "BoxBot",
                "Huter",
                "DEKO",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "category_type": "mixed",
                            "functional_families": [
                                {
                                    "name": "motoblock",
                                    "keywords": [
                                        "мотоблок",
                                        "BoxBot",
                                    ],
                                },
                                {
                                    "name": "cultivator",
                                    "keywords": [
                                        "культиватор",
                                    ],
                                },
                            ],
                            "confidence": 0.82,
                        },
                        ensure_ascii=False,
                    )
                )

            if self.call_count == 2:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": "matakla электротяпка",
                                    "family_name": "cultivator",
                                    "confidence": 0.95,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "motoblock",
                                "keywords": [
                                    "мотоблок",
                                ],
                            },
                            {
                                "name": "cultivator",
                                "keywords": [
                                    "культиватор",
                                ],
                            },
                            {
                                "name": "electric_hoe",
                                "keywords": [
                                    "электротяпка",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": (
                                    "культиватор аккумуляторный "
                                    "электротяпка"
                                ),
                                "family_name": "electric_tiller",
                                "confidence": 0.92,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_responses = FakeResponses()

    fake_client = SimpleNamespace(
        responses=fake_responses
    )

    result = classify_category_dataframe_group(
        dataframe=dataframe,
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        client=fake_client,
        cache_path=cache_path,
    )

    assert len(result) == 2
    assert fake_responses.call_count == 3
    assert cache_path.exists()

    assert result[
        "functional_family"
    ].tolist() == [
        "motoblock",
        "electric_hoe",
    ]

    assert result[
        "functional_family_status"
    ].tolist() == [
        "matched",
        "matched",
    ]

    assert result[
        "category_type"
    ].tolist() == [
        "mixed",
        "mixed",
    ]

    assert result[
        "category_classification_confidence"
    ].tolist() == [
        0.9,
        0.9,
    ]

    saved_text = cache_path.read_text(
        encoding="utf-8",
    )

    assert "BoxBot" not in saved_text
    assert "мотоблок" in saved_text

def test_cached_category_does_not_call_ai_or_mutate_cache(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Аксессуары",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="car_holder",
                keywords=("держатель",),
            ),
        ),
        confidence=0.9,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Автотовары",
    )

    cache_before = cache_path.read_text(
        encoding="utf-8"
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Автотовары",
                "Автотовары",
            ],
            "leaf_category": [
                "Аксессуары",
                "Аксессуары",
            ],
            "product_name": [
                "Автомобильный держатель",
                "Неясный автомобильный товар",
            ],
        }
    )

    class FakeResponses:
        def create(self, **kwargs):
            raise AssertionError(
                "AI must not be called for cached category"
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Аксессуары",
        root_category="Автотовары",
        client=fake_client,
        cache_path=cache_path,
    )

    assert result[
        "functional_family"
    ].tolist() == [
        "car_holder",
        pd.NA,
    ]

    assert result[
        "functional_family_status"
    ].tolist() == [
        "matched",
        "unmatched",
    ]

    assert (
        cache_path.read_text(
            encoding="utf-8"
        )
        == cache_before
    )

def test_dataframe_group_passes_family_rules_to_resolution_prompt(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="family_a",
                keywords=("тип а",),
            ),
            FunctionalFamilyRule(
                name="family_b",
                keywords=("тип б",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
            ],
            "leaf_category": [
                "Тестовая категория",
            ],
            "product_name": [
                "Неясный товар",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            prompt = kwargs["input"]

            if self.call_count == 1:
                assert "FAMILY DEFINITIONS" in prompt
                assert '"name": "family_a"' in prompt
                assert '"тип а"' in prompt
                assert '"name": "family_b"' in prompt
                assert '"тип б"' in prompt

                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": "неясный товар",
                                    "family_name": "unresolved",
                                    "confidence": 0.4,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "family_a",
                                "keywords": ["тип а"],
                            },
                            {
                                "name": "family_b",
                                "keywords": ["тип б"],
                            },
                        ],
                        "confidence": 0.8,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Тестовая категория",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

def test_cached_category_can_be_explicitly_enriched(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="motoblock",
                keywords=("мотоблок",),
            ),
            FunctionalFamilyRule(
                name="cultivator",
                keywords=("культиватор",),
            ),
        ),
        confidence=0.82,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
            ],
            "leaf_category": [
                "Мотоблоки, культиваторы и электротяпки",
            ],
            "product_name": [
                "Электротяпка аккумуляторная",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": (
                                        "электротяпка аккумуляторная"
                                    ),
                                    "family_name": "unresolved",
                                    "confidence": 0.3,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "motoblock",
                                "keywords": [
                                    "мотоблок",
                                ],
                            },
                            {
                                "name": "cultivator",
                                "keywords": [
                                    "культиватор",
                                ],
                            },
                            {
                                "name": "electric_hoe",
                                "keywords": [
                                    "электротяпка",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_responses = FakeResponses()

    fake_client = SimpleNamespace(
        responses=fake_responses
    )

    result = classify_category_dataframe_group(
        dataframe=dataframe,
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    assert fake_responses.call_count == 2

    assert result[
        "functional_family"
    ].tolist() == [
        "electric_hoe",
    ]

    assert result[
        "functional_family_status"
    ].tolist() == [
        "matched",
    ]

    saved = json.loads(
        cache_path.read_text(
            encoding="utf-8"
        )
    )

    families = saved[
        "дом и сад:мотоблоки культиваторы и электротяпки"
    ][
        "functional_families"
    ]

    assert "electric_hoe" in [
        family["name"]
        for family in families
    ]

def test_cached_enrichment_rejects_repair_that_increases_ambiguity(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="family_a",
                keywords=("тип а",),
            ),
            FunctionalFamilyRule(
                name="family_b",
                keywords=("тип б",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    cache_before = cache_path.read_bytes()

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Дом и сад",
            ],
            "leaf_category": [
                "Тестовая категория",
                "Тестовая категория",
                "Тестовая категория",
            ],
            "product_name": [
                "Тип А автоматическая",
                "Система тип Б автоматическая",
                "Автоматическая система",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": (
                                        "автоматическая система"
                                    ),
                                    "family_name": "family_a",
                                    "confidence": 0.95,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "family_a",
                                "keywords": [
                                    "тип а",
                                    "автоматическая",
                                ],
                            },
                            {
                                "name": "family_b",
                                "keywords": [
                                    "тип б",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Тестовая категория",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    cache_after = cache_path.read_bytes()

    assert cache_after == cache_before

    saved = json.loads(
        cache_path.read_text(
            encoding="utf-8"
        )
    )

    families = {
        family["name"]: family["keywords"]
        for family in saved[
            "дом и сад:тестовая категория"
        ]["functional_families"]
    }

    assert "автоматическая" not in families[
        "family_a"
    ]

def test_cached_enrichment_accepts_repair_that_improves_matching(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="family_a",
                keywords=("тип а",),
            ),
            FunctionalFamilyRule(
                name="family_b",
                keywords=("тип б",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    cache_before = cache_path.read_bytes()

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Дом и сад",
            ],
            "leaf_category": [
                "Тестовая категория",
                "Тестовая категория",
                "Тестовая категория",
            ],
            "product_name": [
                "Тип А базовый",
                "Тип Б базовый",
                "Станция водоснабжения",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": (
                                        "станция водоснабжения"
                                    ),
                                    "family_name": "family_a",
                                    "confidence": 0.95,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "family_a",
                                "keywords": [
                                    "тип а",
                                    "станция водоснабжения",
                                ],
                            },
                            {
                                "name": "family_b",
                                "keywords": [
                                    "тип б",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Тестовая категория",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    cache_after = cache_path.read_bytes()

    assert cache_after != cache_before

    saved = json.loads(
        cache_path.read_text(
            encoding="utf-8"
        )
    )

    families = {
        family["name"]: family["keywords"]
        for family in saved[
            "дом и сад:тестовая категория"
        ]["functional_families"]
    }

    assert "станция водоснабжения" in families[
        "family_a"
    ]

def test_cached_enrichment_persists_resolution_evidence(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="pump_station",
                keywords=("насосная станция",),
            ),
            FunctionalFamilyRule(
                name="surface_pump",
                keywords=("поверхностный насос",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
            ],
            "leaf_category": [
                "Поверхностные насосы",
            ],
            "product_name": [
                "Автономная станция водоснабжения ASV 4200P",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": (
                                        "автономная станция "
                                        "водоснабжения asv 4200p"
                                    ),
                                    "family_name": "pump_station",
                                    "confidence": 0.94,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            prompt = kwargs["input"]

            assert "RESOLUTION EVIDENCE" in prompt
            assert (
                "автономная станция водоснабжения asv 4200p"
                in prompt
            )
            assert '"family_name": "pump_station"' in prompt

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "pump_station",
                                "keywords": [
                                    "насосная станция",
                                    "станция водоснабжения",
                                ],
                            },
                            {
                                "name": "surface_pump",
                                "keywords": [
                                    "поверхностный насос",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Поверхностные насосы",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    saved = json.loads(
        cache_path.read_text(
            encoding="utf-8"
        )
    )

    families = {
        family["name"]: family["keywords"]
        for family in saved[
            "дом и сад:поверхностные насосы"
        ]["functional_families"]
    }

    assert "станция водоснабжения" in families[
        "pump_station"
    ]

def test_build_category_repair_prompt_keeps_existing_families():
    classification = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="motoblock",
                keywords=("мотоблок",),
            ),
            FunctionalFamilyRule(
                name="cultivator",
                keywords=("культиватор",),
            ),
        ),
        confidence=0.82,
    )

    prompt = build_category_repair_prompt(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        classification=classification,
        unresolved_products=[
            "MATAKLA Электротяпка",
            "MATAKLA Электротяпка",
        ],
    )

    assert '"name": "motoblock"' in prompt
    assert '"name": "cultivator"' in prompt

    assert prompt.count(
        "matakla электротяпка"
    ) == 1

    assert (
        "не пропущено ли отдельное"
        in prompt.lower()
    )

def test_build_category_repair_prompt_includes_resolution_evidence():
    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="pump_station",
                keywords=("насосная станция",),
            ),
            FunctionalFamilyRule(
                name="surface_pump",
                keywords=("поверхностный насос",),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_category_repair_prompt(
        category_name="Поверхностные насосы",
        classification=classification,
        unresolved_products=[
            "Неясный насос",
        ],
        resolution_evidence=[
            {
                "product_name": (
                    "Автономная станция водоснабжения ASV 4200P"
                ),
                "family_name": "pump_station",
                "confidence": 0.94,
            },
            {
                "product_name": (
                    "Насос поверхностный 1100 Вт"
                ),
                "family_name": "surface_pump",
                "confidence": 0.91,
            },
        ],
    )

    assert "RESOLUTION EVIDENCE" in prompt

    assert (
        "автономная станция водоснабжения asv 4200p"
        in prompt
    )

    assert '"family_name": "pump_station"' in prompt

    assert (
        "насос поверхностный 1100 вт"
        in prompt
    )

    assert '"family_name": "surface_pump"' in prompt

def test_generate_category_repair_adds_missing_family():
    classification = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="motoblock",
                keywords=("мотоблок",),
            ),
            FunctionalFamilyRule(
                name="cultivator",
                keywords=("культиватор",),
            ),
        ),
        confidence=0.82,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "motoblock",
                                "keywords": [
                                    "мотоблок",
                                ],
                            },
                            {
                                "name": "cultivator",
                                "keywords": [
                                    "культиватор",
                                ],
                            },
                            {
                                "name": "electric_hoe",
                                "keywords": [
                                    "электротяпка",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_category_repair(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        classification=classification,
        unresolved_products=[
            "MATAKLA Электротяпка",
        ],
        client=fake_client,
    )

    assert [
        rule.name
        for rule in result.functional_families
    ] == [
        "motoblock",
        "cultivator",
        "electric_hoe",
    ]

    assert result.functional_families[2].keywords == (
        "электротяпка",
    )

    assert result.confidence == 0.9

def test_generate_category_repair_preserves_omitted_existing_family():
    classification = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="motoblock",
                keywords=("мотоблок",),
            ),
            FunctionalFamilyRule(
                name="cultivator",
                keywords=("культиватор",),
            ),
        ),
        confidence=0.82,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "cultivator",
                                "keywords": [
                                    "культиватор",
                                ],
                            },
                            {
                                "name": "electric_hoe",
                                "keywords": [
                                    "электротяпка",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_category_repair(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        classification=classification,
        unresolved_products=[
            "MATAKLA Электротяпка",
        ],
        client=fake_client,
    )

    assert [
        rule.name
        for rule in result.functional_families
    ] == [
        "motoblock",
        "cultivator",
        "electric_hoe",
    ]

    assert result.functional_families[0].keywords == (
        "мотоблок",
    )

    assert result.functional_families[2].keywords == (
        "электротяпка",
    )

    assert result.category_type == "mixed"
    assert result.confidence == 0.9

def test_classify_category_dataframe_group_separates_same_leaf_by_root(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Автотовары",
                "Дом и сад",
            ],
            "leaf_category": [
                "Аксессуары",
                "Аксессуары",
            ],
            "product_name": [
                "Автомобильный держатель",
                "Садовый шланг аксессуар",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1

            prompt = kwargs["input"]

            assert (
                "автомобильный держатель"
                in prompt
            )
            assert (
                "садовый шланг аксессуар"
                not in prompt
            )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "homogeneous",
                        "functional_families": [
                            {
                                "name": "car_holder",
                                "keywords": [
                                    "держатель",
                                ],
                            }
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_responses = FakeResponses()

    fake_client = SimpleNamespace(
        responses=fake_responses
    )

    result = classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Аксессуары",
        root_category="Автотовары",
        client=fake_client,
        cache_path=cache_path,
    )

    assert fake_responses.call_count == 1

    assert result["product_name"].tolist() == [
        "Автомобильный держатель",
    ]

    assert result["functional_family"].tolist() == [
        "car_holder",
    ]

    cache = json.loads(
        cache_path.read_text(
            encoding="utf-8"
        )
    )

    assert "автотовары:аксессуары" in cache
    assert "дом и сад:аксессуары" not in cache

def test_generate_family_resolutions_marks_omitted_product_unresolved():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": "Товар A",
                                "family_name": "cultivator",
                                "confidence": 0.9,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_family_resolutions(
        product_names=[
            "Товар A",
            "Товар B",
        ],
        allowed_families=(
            "cultivator",
            "walk_behind_tractor",
        ),
        client=fake_client,
    )

    assert result == [
        {
            "product_name": "товар a",
            "family_name": "cultivator",
            "confidence": 0.9,
        },
        {
            "product_name": "товар b",
            "family_name": "unresolved",
            "confidence": 0.0,
        },
    ]

def test_generate_family_resolutions_restricts_schema_to_batch_products():
    class FakeResponses:
        def create(self, **kwargs):
            schema = (
                kwargs["text"]
                ["format"]
                ["schema"]
            )

            properties = (
                schema["properties"]
                ["resolutions"]
                ["items"]
                ["properties"]
            )

            assert properties[
                "product_name"
            ]["enum"] == [
                "газонокосилка oasis gbe 3 eco",
                "триммер deko",
            ]

            assert properties[
                "family_name"
            ]["enum"] == [
                "mower",
                "trimmer",
                "unresolved",
            ]

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": (
                                    "газонокосилка oasis gbe 3 eco"
                                ),
                                "family_name": "mower",
                                "confidence": 0.95,
                            },
                            {
                                "product_name": "триммер deko",
                                "family_name": "trimmer",
                                "confidence": 0.9,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_family_resolutions(
        product_names=[
            "Газонокосилка OASIS GBE-3 ECO",
            "Триммер DEKO",
        ],
        allowed_families=(
            "mower",
            "trimmer",
        ),
        client=fake_client,
    )

    assert len(result) == 2

def test_generate_family_resolutions_passes_family_rules_to_prompt():
    family_rules = (
        FunctionalFamilyRule(
            name="surface_pump",
            keywords=(
                "насос поверхностный",
                "поверхностный насос",
            ),
        ),
        FunctionalFamilyRule(
            name="pump_station",
            keywords=(
                "насосная станция",
                "станция водоснабжения",
            ),
        ),
    )

    class FakeResponses:
        def create(self, **kwargs):
            prompt = kwargs["input"]

            assert "FAMILY DEFINITIONS" in prompt
            assert '"name": "surface_pump"' in prompt
            assert '"поверхностный насос"' in prompt
            assert '"name": "pump_station"' in prompt
            assert '"станция водоснабжения"' in prompt

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": (
                                    "автономная станция "
                                    "водоснабжения asv 4200p"
                                ),
                                "family_name": "pump_station",
                                "confidence": 0.94,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_family_resolutions(
        product_names=[
            "Автономная станция водоснабжения ASV 4200P",
        ],
        allowed_families=(
            "surface_pump",
            "pump_station",
        ),
        family_rules=family_rules,
        client=fake_client,
    )

    assert result[0]["family_name"] == "pump_station"

def test_generate_category_repair_restricts_category_type_schema():
    classification = CategoryClassification(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="motoblock",
                keywords=("мотоблок",),
            ),
            FunctionalFamilyRule(
                name="cultivator",
                keywords=("культиватор",),
            ),
        ),
        confidence=0.82,
    )

    class FakeResponses:
        def create(self, **kwargs):
            schema = (
                kwargs["text"]
                ["format"]
                ["schema"]
            )

            assert (
                schema["properties"]
                ["category_type"]
                ["enum"]
            ) == [
                "mixed",
            ]

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "motoblock",
                                "keywords": [
                                    "мотоблок",
                                ],
                            },
                            {
                                "name": "cultivator",
                                "keywords": [
                                    "культиватор",
                                ],
                            },
                            {
                                "name": "electric_hoe",
                                "keywords": [
                                    "электротяпка",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    repaired = generate_category_repair(
        category_name=(
            "Мотоблоки, культиваторы и электротяпки"
        ),
        classification=classification,
        unresolved_products=[
            "MATAKLA Электротяпка",
        ],
        client=fake_client,
    )

    assert repaired.category_type == "mixed"

def test_generate_category_repair_passes_resolution_evidence():
    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="pump_station",
                keywords=("насосная станция",),
            ),
        ),
        confidence=0.8,
    )

    class FakeResponses:
        def create(self, **kwargs):
            prompt = kwargs["input"]

            assert "RESOLUTION EVIDENCE" in prompt
            assert (
                "автономная станция водоснабжения"
                in prompt
            )
            assert '"family_name": "pump_station"' in prompt

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "pump_station",
                                "keywords": [
                                    "насосная станция",
                                    "станция водоснабжения",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_category_repair(
        category_name="Поверхностные насосы",
        classification=classification,
        unresolved_products=[
            "Неясный насос",
        ],
        resolution_evidence=[
            {
                "product_name": (
                    "Автономная станция водоснабжения"
                ),
                "family_name": "pump_station",
                "confidence": 0.94,
            },
        ],
        client=fake_client,
    )

    assert result.functional_families[0].keywords == (
        "насосная станция",
        "станция водоснабжения",
    )

def test_repair_prompt_requires_existing_family_keyword_enrichment():
    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="pump_station",
                keywords=("насосная станция",),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_category_repair_prompt(
        category_name="Поверхностные насосы",
        classification=classification,
        unresolved_products=[
            "Автономная станция водоснабжения ASV 4200P",
        ],
        resolution_evidence=[
            {
                "product_name": (
                    "Автономная станция водоснабжения ASV 4200P"
                ),
                "family_name": "pump_station",
                "confidence": 0.94,
            },
        ],
    )

    assert "ENRICH EXISTING FAMILY KEYWORDS" in prompt

    assert (
        "сохрани текущую структуру без изменений"
        not in prompt
    )

    assert (
        "станция водоснабжения"
        in prompt
    )

def test_category_repair_preserves_existing_family_keywords():
    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="pump_station",
                keywords=(
                    "насосная станция",
                    "pump_station",
                ),
            ),
        ),
        confidence=0.8,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "pump_station",
                                "keywords": [
                                    "автоматическая",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    repaired = generate_category_repair(
        category_name="Поверхностные насосы",
        classification=classification,
        unresolved_products=[],
        resolution_evidence=[],
        client=fake_client,
    )

    keywords = repaired.functional_families[0].keywords

    assert "насосная станция" in keywords
    assert "pump_station" in keywords
    
    # Старые keywords сохраняются,
    # но неподтверждённый новый keyword
    # не должен приниматься.
    assert "автоматическая" not in keywords

def test_cached_enrichment_keeps_only_incrementally_useful_keywords(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="family_a",
                keywords=("тип а",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
                "Дом и сад",
            ],
            "leaf_category": [
                "Тестовая категория",
                "Тестовая категория",
            ],
            "product_name": [
                "Тип А базовый",
                "Автоматическая станция водоснабжения",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": (
                                        "автоматическая "
                                        "станция водоснабжения"
                                    ),
                                    "family_name": "family_a",
                                    "confidence": 0.95,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "family_a",
                                "keywords": [
                                    "тип а",
                                    "автоматическая",
                                    "станция водоснабжения",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Тестовая категория",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    saved = json.loads(
        cache_path.read_text(
            encoding="utf-8"
        )
    )

    keywords = {
        family["name"]: family["keywords"]
        for family in saved[
            "дом и сад:тестовая категория"
        ]["functional_families"]
    }["family_a"]

    assert "станция водоснабжения" in keywords

    # Более общий keyword не должен сохраняться,
    # если после принятия более точного термина
    # он уже не даёт дополнительного улучшения.
    assert "автоматическая" not in keywords

def test_category_repair_rejects_unsupported_new_keywords():
    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="pump_station",
                keywords=(
                    "насосная станция",
                    "pump_station",
                ),
            ),
        ),
        confidence=0.8,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "pump_station",
                                "keywords": [
                                    "насосная станция",
                                    "pump_station",
                                    "автоматическая",
                                    "станция водоснабжения",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    repaired = generate_category_repair(
        category_name="Поверхностные насосы",
        classification=classification,
        unresolved_products=[],
        resolution_evidence=[
            {
                "product_name": (
                    "Автономная станция водоснабжения ASV 4200P"
                ),
                "family_name": "pump_station",
                "confidence": 0.94,
            },
        ],
        client=fake_client,
    )

    keywords = repaired.functional_families[0].keywords

    assert "насосная станция" in keywords
    assert "pump_station" in keywords

    # Есть прямое подтверждение в product_name.
    assert "станция водоснабжения" in keywords

    # В evidence такого термина нет —
    # AI не должен иметь права сохранить его.
    assert "автоматическая" not in keywords

def test_repair_prompt_treats_resolution_evidence_as_provisional():
    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="surface_pump",
                keywords=(
                    "поверхностный насос",
                    "насос поверхностный",
                ),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_category_repair_prompt(
        category_name="Поверхностные насосы",
        classification=classification,
        unresolved_products=[
            "Мотопомпа бензиновая для воды Huter MP-50",
            "Мотопомпа бензиновая для воды Huter MP-70",
        ],
        resolution_evidence=[
            {
                "product_name": (
                    "Мотопомпа бензиновая для воды Huter MP-50"
                ),
                "family_name": "surface_pump",
                "confidence": 0.90,
            },
            {
                "product_name": (
                    "Мотопомпа бензиновая для воды Huter MP-70"
                ),
                "family_name": "surface_pump",
                "confidence": 0.91,
            },
        ],
    )

    assert "PROVISIONAL RESOLUTION EVIDENCE" in prompt

    assert (
        "подтверждённые примеры"
        not in prompt
    )

    assert (
        "может быть ошибочным"
        in prompt
    )

    assert (
        "отдельное functional_family"
        in prompt
    )

def test_category_repair_can_create_new_family_from_provisional_cluster():
    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="surface_pump",
                keywords=(
                    "поверхностный насос",
                    "насос поверхностный",
                ),
            ),
        ),
        confidence=0.8,
    )

    class FakeResponses:
        def create(self, **kwargs):
            prompt = kwargs["input"]

            assert "PROVISIONAL RESOLUTION EVIDENCE" in prompt

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "surface_pump",
                                "keywords": [
                                    "поверхностный насос",
                                    "насос поверхностный",
                                ],
                            },
                            {
                                "name": "motor_pump",
                                "keywords": [
                                    "мотопомпа",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    repaired = generate_category_repair(
        category_name="Поверхностные насосы",
        classification=classification,
        unresolved_products=[
            "Мотопомпа бензиновая для воды Huter MP-50",
            "Мотопомпа бензиновая для воды Huter MP-70",
        ],
        resolution_evidence=[
            {
                "product_name": (
                    "Мотопомпа бензиновая для воды Huter MP-50"
                ),
                "family_name": "surface_pump",
                "confidence": 0.90,
            },
            {
                "product_name": (
                    "Мотопомпа бензиновая для воды Huter MP-70"
                ),
                "family_name": "surface_pump",
                "confidence": 0.91,
            },
        ],
        client=fake_client,
    )

    family_names = {
        rule.name
        for rule in repaired.functional_families
    }

    assert "surface_pump" in family_names
    assert "motor_pump" in family_names

    motor_pump = next(
        rule
        for rule in repaired.functional_families
        if rule.name == "motor_pump"
    )

    assert "мотопомпа" in motor_pump.keywords

def test_cached_enrichment_persists_new_family_when_it_improves_matching(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="surface_pump",
                keywords=(
                    "поверхностный насос",
                ),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Дом и сад",
            ],
            "leaf_category": [
                "Поверхностные насосы",
                "Поверхностные насосы",
                "Поверхностные насосы",
            ],
            "product_name": [
                "Поверхностный насос для воды",
                "Мотопомпа бензиновая Huter MP-50",
                "Мотопомпа бензиновая Huter MP-70",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            candidate_response = (
                reject_candidate_validation_if_requested(
                    kwargs
                )
            )

            if candidate_response is not None:
                return candidate_response

            discovery_response = (
                empty_discovery_response_if_requested(
                    kwargs
                )
            )

            if discovery_response is not None:
                return discovery_response

            self.call_count += 1

            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": (
                                        "мотопомпа бензиновая "
                                        "huter mp 50"
                                    ),
                                    "family_name": "surface_pump",
                                    "confidence": 0.90,
                                },
                                {
                                    "product_name": (
                                        "мотопомпа бензиновая "
                                        "huter mp 70"
                                    ),
                                    "family_name": "surface_pump",
                                    "confidence": 0.91,
                                },
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "surface_pump",
                                "keywords": [
                                    "поверхностный насос",
                                ],
                            },
                            {
                                "name": "motor_pump",
                                "keywords": [
                                    "мотопомпа",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Поверхностные насосы",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    saved = json.loads(
        cache_path.read_text(
            encoding="utf-8"
        )
    )

    families = {
        family["name"]: family["keywords"]
        for family in saved[
            "дом и сад:поверхностные насосы"
        ]["functional_families"]
    }

    assert "motor_pump" in families
    assert "мотопомпа" in families["motor_pump"]

    deterministic = classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Поверхностные насосы",
        root_category="Дом и сад",
        client=object(),
        cache_path=cache_path,
        enrich_cached=False,
    )

    assert (
        deterministic[
            "functional_family_status"
        ].tolist()
        == [
            "matched",
            "matched",
            "matched",
        ]
    )

    assert (
        deterministic[
            "functional_family"
        ].tolist()
        == [
            "surface_pump",
            "motor_pump",
            "motor_pump",
        ]
    )

def test_repair_prompt_prefers_new_family_for_distinct_repeated_cluster():
    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=(
                    "существующий тип",
                ),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_category_repair_prompt(
        category_name="Тестовая категория",
        classification=classification,
        unresolved_products=[
            "Новый тип товара модель один",
            "Новый тип товара модель два",
            "Новый тип товара модель три",
        ],
        resolution_evidence=[
            {
                "product_name": "Новый тип товара модель один",
                "family_name": "existing_family",
                "confidence": 0.90,
            },
            {
                "product_name": "Новый тип товара модель два",
                "family_name": "existing_family",
                "confidence": 0.91,
            },
            {
                "product_name": "Новый тип товара модель три",
                "family_name": "existing_family",
                "confidence": 0.89,
            },
        ],
    )

    assert "MISSING FAMILY DISCOVERY HAS PRIORITY" in prompt

    assert (
        "не добавляй термин как keyword существующей family"
        in prompt
    )

    assert (
        "сначала рассмотри создание новой functional_family"
        in prompt
    )

def test_missing_family_discovery_prompt_has_single_responsibility():
    from app.category_classification_ai import (
        build_missing_family_discovery_prompt,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=(
                    "существующий тип",
                ),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_missing_family_discovery_prompt(
        category_name="Тестовая категория",
        classification=classification,
        problem_products=[
            "Новый тип товара модель один",
            "Новый тип товара модель два",
            "Новый тип товара модель три",
        ],
        provisional_resolutions=[
            {
                "product_name": "Новый тип товара модель один",
                "family_name": "existing_family",
                "confidence": 0.90,
            },
            {
                "product_name": "Новый тип товара модель два",
                "family_name": "existing_family",
                "confidence": 0.91,
            },
            {
                "product_name": "Новый тип товара модель три",
                "family_name": "existing_family",
                "confidence": 0.89,
            },
        ],
    )

    assert "MISSING FAMILY DISCOVERY ONLY" in prompt

    assert (
        "DO NOT ENRICH EXISTING FAMILY KEYWORDS"
        in prompt
    )

    assert '"name": "existing_family"' in prompt
    assert '"существующий тип"' in prompt

    assert "новый тип товара модель один" in prompt

    assert (
        "предварительное назначение"
        in prompt
    )

def test_generate_missing_family_discovery_returns_only_new_families():
    from app.category_classification_ai import (
        generate_missing_family_discovery,
    )

    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="surface_pump",
                keywords=(
                    "поверхностный насос",
                ),
            ),
        ),
        confidence=0.8,
    )

    class FakeResponses:
        def create(self, **kwargs):
            prompt = kwargs["input"]

            assert "MISSING FAMILY DISCOVERY ONLY" in prompt
            assert (
                "DO NOT ENRICH EXISTING FAMILY KEYWORDS"
                in prompt
            )

            schema = kwargs[
                "text"
            ]["format"]["schema"]

            assert set(
                schema["properties"]
            ) == {"new_families"}

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "new_families": [
                            {
                                "name": "motor_pump",
                                "keywords": [
                                    "мотопомпа",
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_missing_family_discovery(
        category_name="Поверхностные насосы",
        classification=classification,
        problem_products=[
            "Мотопомпа бензиновая Huter MP-50",
            "Мотопомпа бензиновая Huter MP-70",
        ],
        provisional_resolutions=[
            {
                "product_name": (
                    "Мотопомпа бензиновая Huter MP-50"
                ),
                "family_name": "surface_pump",
                "confidence": 0.90,
            },
            {
                "product_name": (
                    "Мотопомпа бензиновая Huter MP-70"
                ),
                "family_name": "surface_pump",
                "confidence": 0.91,
            },
        ],
        client=fake_client,
    )

    assert result == (
        FunctionalFamilyRule(
            name="motor_pump",
            keywords=("мотопомпа",),
        ),
    )

def test_cached_enrichment_runs_missing_family_discovery_before_repair(
    tmp_path,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Поверхностные насосы",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="surface_pump",
                keywords=("поверхностный насос",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Дом и сад",
            ],
            "leaf_category": [
                "Поверхностные насосы",
                "Поверхностные насосы",
                "Поверхностные насосы",
            ],
            "product_name": [
                "Поверхностный насос",
                "Мотопомпа Huter MP-50",
                "Мотопомпа Huter MP-70",
            ],
        }
    )

    class FakeResponses:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            candidate_response = (
                reject_candidate_validation_if_requested(
                    kwargs
                )
            )

            if candidate_response is not None:
                return candidate_response

            self.call_count += 1
            prompt = kwargs["input"]

            # 1. provisional resolution
            if self.call_count == 1:
                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "resolutions": [
                                {
                                    "product_name": (
                                        "мотопомпа huter mp 50"
                                    ),
                                    "family_name": "surface_pump",
                                    "confidence": 0.9,
                                },
                                {
                                    "product_name": (
                                        "мотопомпа huter mp 70"
                                    ),
                                    "family_name": "surface_pump",
                                    "confidence": 0.9,
                                },
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            # 2. ОБЯЗАТЕЛЬНО отдельный missing-family pass
            if self.call_count == 2:
                assert "MISSING FAMILY DISCOVERY ONLY" in prompt

                return SimpleNamespace(
                    output_text=json.dumps(
                        {
                            "new_families": [
                                {
                                    "name": "motor_pump",
                                    "keywords": [
                                        "мотопомпа",
                                    ],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )

            # 3. keyword repair — ничего нового
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "category_type": "mixed",
                        "functional_families": [
                            {
                                "name": "surface_pump",
                                "keywords": [
                                    "поверхностный насос",
                                ],
                            },
                            {
                                "name": "motor_pump",
                                "keywords": [
                                    "мотопомпа",
                                ],
                            },
                        ],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Поверхностные насосы",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    assert fake_client.responses.call_count == 3

    saved = json.loads(
        cache_path.read_text(encoding="utf-8")
    )

    families = {
        family["name"]: family["keywords"]
        for family in saved[
            "дом и сад:поверхностные насосы"
        ]["functional_families"]
    }

    assert "motor_pump" in families
    assert "мотопомпа" in families["motor_pump"]
def test_missing_family_discovery_prompt_groups_provisional_products():
    from app.category_classification_ai import (
        build_missing_family_discovery_prompt,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="family_a",
                keywords=("тип а",),
            ),
            FunctionalFamilyRule(
                name="family_b",
                keywords=("тип б",),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_missing_family_discovery_prompt(
        category_name="Тестовая категория",
        classification=classification,
        problem_products=[
            "Отдельный товар один",
            "Отдельный товар два",
            "Другой товар",
        ],
        provisional_resolutions=[
            {
                "product_name": "отдельный товар один",
                "family_name": "family_a",
                "confidence": 0.9,
            },
            {
                "product_name": "отдельный товар два",
                "family_name": "family_a",
                "confidence": 0.9,
            },
            {
                "product_name": "другой товар",
                "family_name": "family_b",
                "confidence": 0.9,
            },
        ],
    )

    assert "PROVISIONAL FAMILY GROUPS" in prompt

    assert '"family_name": "family_a"' in prompt
    assert '"отдельный товар один"' in prompt
    assert '"отдельный товар два"' in prompt

    assert (
        "ищи повторяющийся функциональный подтип"
        in prompt
    )


def test_generate_missing_family_discovery_rejects_singleton_family():
    from app.category_classification_ai import (
        generate_missing_family_discovery,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий товар",),
            ),
        ),
        confidence=0.8,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "new_families": [
                            {
                                "name": "singleton_family",
                                "keywords": [
                                    "редкий отдельный товар",
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = generate_missing_family_discovery(
        category_name="Тестовая категория",
        classification=classification,
        problem_products=[
            "Редкий отдельный товар модель один",
            "Другой проблемный товар",
            "Ещё один проблемный товар",
        ],
        provisional_resolutions=[
            {
                "product_name": "редкий отдельный товар модель один",
                "family_name": "existing_family",
                "confidence": 0.9,
            },
            {
                "product_name": "другой проблемный товар",
                "family_name": "existing_family",
                "confidence": 0.8,
            },
            {
                "product_name": "ещё один проблемный товар",
                "family_name": "existing_family",
                "confidence": 0.8,
            },
        ],
        client=fake_client,
    )

    assert result == ()


def test_collect_repeated_term_candidates_finds_supported_terms():
    from app.category_classification_ai import (
        collect_repeated_term_candidates,
    )

    result = collect_repeated_term_candidates(
        [
            "Мотопомпа бензиновая Huter MP-50",
            "Мотопомпа бензиновая DGM TMP-101",
            "Мотопомпа ECO WP-153C",
            "Насос поверхностный Huter SGP",
            "Редкий отдельный товар",
        ],
        min_support=2,
    )

    by_term = {
        item["term"]: item
        for item in result
    }

    assert "мотопомпа" in by_term
    assert by_term["мотопомпа"]["support"] == 3

    assert len(
        by_term["мотопомпа"]["examples"]
    ) == 3

    assert "редкий" not in by_term


def test_missing_family_discovery_prompt_includes_repeated_term_candidates():
    from app.category_classification_ai import (
        build_missing_family_discovery_prompt,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_missing_family_discovery_prompt(
        category_name="Тестовая категория",
        classification=classification,
        problem_products=[
            "Новый тип товара Alpha",
            "Новый тип товара Beta",
            "Новый тип товара Gamma",
            "Существующий тип товара",
        ],
        provisional_resolutions=[
            {
                "product_name": "новый тип товара alpha",
                "family_name": "existing_family",
                "confidence": 0.9,
            },
            {
                "product_name": "новый тип товара beta",
                "family_name": "existing_family",
                "confidence": 0.9,
            },
            {
                "product_name": "новый тип товара gamma",
                "family_name": "existing_family",
                "confidence": 0.9,
            },
        ],
    )

    assert "REPEATED TERM CANDIDATES" in prompt
    assert '"term": "новый тип товара"' in prompt
    assert '"support": 3' in prompt

    assert (
        "используй повторяемость только как сигнал"
        in prompt
    )


def test_missing_family_candidate_prompt_focuses_on_single_candidate():
    from app.category_classification_ai import (
        build_missing_family_candidate_prompt,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    candidate = {
        "term": "новый тип",
        "support": 3,
        "examples": [
            "новый тип alpha",
            "новый тип beta",
            "новый тип gamma",
        ],
    }

    prompt = build_missing_family_candidate_prompt(
        category_name="Тестовая категория",
        classification=classification,
        candidate=candidate,
    )

    assert "CANDIDATE FAMILY VALIDATION ONLY" in prompt

    assert '"term": "новый тип"' in prompt
    assert '"support": 3' in prompt

    assert '"name": "existing_family"' in prompt
    assert '"существующий тип"' in prompt

    assert (
        "оцени только этот кандидат"
        in prompt
    )

    assert (
        "не ищи другие новые family"
        in prompt
    )


def test_validate_missing_family_candidate_accepts_distinct_family():
    from app.category_classification_ai import (
        validate_missing_family_candidate,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    candidate = {
        "term": "новый тип",
        "support": 3,
        "examples": [
            "новый тип alpha",
            "новый тип beta",
            "новый тип gamma",
        ],
    }

    class FakeResponses:
        def create(self, **kwargs):
            prompt = kwargs["input"]

            assert "CANDIDATE FAMILY VALIDATION ONLY" in prompt

            schema = kwargs["text"]["format"]["schema"]

            assert set(
                schema["properties"]
            ) == {
                "accept",
                "family_name",
                "keywords",
            }

            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "accept": True,
                        "family_name": "new_family",
                        "keywords": [
                            "новый тип",
                        ],
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = validate_missing_family_candidate(
        category_name="Тестовая категория",
        classification=classification,
        candidate=candidate,
        client=fake_client,
    )

    assert result == FunctionalFamilyRule(
        name="new_family",
        keywords=("новый тип",),
    )


def test_validate_missing_family_candidate_rejects_candidate():
    from app.category_classification_ai import (
        validate_missing_family_candidate,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    candidate = {
        "term": "повторяющийся термин",
        "support": 3,
        "examples": [
            "повторяющийся термин alpha",
            "повторяющийся термин beta",
            "повторяющийся термин gamma",
        ],
    }

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "accept": False,
                        "family_name": "",
                        "keywords": [],
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    result = validate_missing_family_candidate(
        category_name="Тестовая категория",
        classification=classification,
        candidate=candidate,
        client=fake_client,
    )

    assert result is None


def test_discover_missing_families_from_repeated_candidates(
    monkeypatch,
):
    from app.category_classification_ai import (
        discover_missing_families_from_repeated_candidates,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    candidates = [
        {
            "term": "новый тип",
            "support": 3,
            "examples": [
                "новый тип alpha",
                "новый тип beta",
                "новый тип gamma",
            ],
        },
        {
            "term": "общая характеристика",
            "support": 4,
            "examples": [
                "общая характеристика один",
                "общая характеристика два",
                "общая характеристика три",
                "общая характеристика четыре",
            ],
        },
    ]

    calls = []

    def fake_validate(
        category_name,
        classification,
        candidate,
        client,
        model="gpt-5-nano",
    ):
        calls.append(candidate["term"])

        if candidate["term"] == "новый тип":
            return FunctionalFamilyRule(
                name="new_family",
                keywords=("новый тип",),
            )

        return None

    monkeypatch.setattr(
        "app.category_classification_ai."
        "validate_missing_family_candidate",
        fake_validate,
    )

    result = discover_missing_families_from_repeated_candidates(
        category_name="Тестовая категория",
        classification=classification,
        candidates=candidates,
        client=object(),
    )

    assert calls == [
        "новый тип",
        "общая характеристика",
    ]

    assert result == (
        FunctionalFamilyRule(
            name="new_family",
            keywords=("новый тип",),
        ),
    )


def test_dataframe_group_checks_repeated_candidates_before_open_discovery(
    tmp_path,
    monkeypatch,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": [
                "Дом и сад",
                "Дом и сад",
                "Дом и сад",
            ],
            "leaf_category": [
                "Тестовая категория",
                "Тестовая категория",
                "Тестовая категория",
            ],
            "product_name": [
                "Существующий тип",
                "Новый тип Alpha",
                "Новый тип Beta",
            ],
        }
    )

    events = []

    def fake_collect(
        product_names,
        min_support=2,
        max_ngram=3,
        max_candidates=50,
    ):
        return [
            {
                "term": "новый тип",
                "support": 2,
                "examples": [
                    "новый тип alpha",
                    "новый тип beta",
                ],
            }
        ]

    def fake_candidate_discovery(
        category_name,
        classification,
        candidates,
        client,
        model="gpt-5-nano",
    ):
        events.append("candidate")

        return (
            FunctionalFamilyRule(
                name="new_family",
                keywords=("новый тип",),
            ),
        )

    def fake_open_discovery(
        category_name,
        classification,
        problem_products,
        provisional_resolutions,
        client,
        model="gpt-5-nano",
    ):
        events.append("open")

        family_names = {
            rule.name
            for rule in classification.functional_families
        }

        assert "new_family" in family_names

        return ()

    def fake_repair(
        category_name,
        classification,
        unresolved_products,
        resolution_evidence,
        client,
        model="gpt-5-nano",
    ):
        events.append("repair")
        return classification

    monkeypatch.setattr(
        "app.category_classification_ai."
        "collect_repeated_term_candidates",
        fake_collect,
    )

    monkeypatch.setattr(
        "app.category_classification_ai."
        "discover_missing_families_from_repeated_candidates",
        fake_candidate_discovery,
    )

    monkeypatch.setattr(
        "app.category_classification_ai."
        "generate_missing_family_discovery",
        fake_open_discovery,
    )

    monkeypatch.setattr(
        "app.category_classification_ai."
        "generate_category_repair",
        fake_repair,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": "новый тип alpha",
                                "family_name": "existing_family",
                                "confidence": 0.9,
                            },
                            {
                                "product_name": "новый тип beta",
                                "family_name": "existing_family",
                                "confidence": 0.9,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Тестовая категория",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    assert events == [
        "candidate",
        "open",
        "repair",
    ]


def test_select_missing_family_candidates_removes_redundant_terms():
    from app.category_classification_ai import (
        select_missing_family_candidates,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("насосная станция",),
            ),
        ),
        confidence=0.8,
    )

    candidates = [
        {
            "term": "мотопомпа",
            "support": 8,
            "examples": ["мотопомпа a", "мотопомпа b"],
        },
        {
            "term": "мотопомпа бензиновая",
            "support": 7,
            "examples": [
                "мотопомпа бензиновая a",
                "мотопомпа бензиновая b",
            ],
        },
        {
            "term": "насосная станция",
            "support": 12,
            "examples": [
                "насосная станция a",
                "насосная станция b",
            ],
        },
    ]

    result = select_missing_family_candidates(
        candidates,
        classification=classification,
        max_candidates=10,
    )

    terms = [
        item["term"]
        for item in result
    ]

    assert "мотопомпа" in terms
    assert "мотопомпа бензиновая" not in terms
    assert "насосная станция" not in terms


def test_select_missing_family_candidates_rejects_fragments_of_existing_keywords():
    from app.category_classification_ai import (
        select_missing_family_candidates,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="family_a",
                keywords=(
                    "насосная станция",
                    "поверхностный насос",
                ),
            ),
        ),
        confidence=0.8,
    )

    candidates = [
        {
            "term": "насос",
            "support": 20,
            "examples": ["насос a", "насос b"],
        },
        {
            "term": "станция",
            "support": 15,
            "examples": ["станция a", "станция b"],
        },
        {
            "term": "насосная",
            "support": 14,
            "examples": ["насосная a", "насосная b"],
        },
        {
            "term": "поверхностный",
            "support": 9,
            "examples": [
                "поверхностный a",
                "поверхностный b",
            ],
        },
        {
            "term": "новый тип",
            "support": 8,
            "examples": [
                "новый тип a",
                "новый тип b",
            ],
        },
    ]

    result = select_missing_family_candidates(
        candidates,
        classification=classification,
        max_candidates=10,
    )

    terms = [
        item["term"]
        for item in result
    ]

    assert terms == ["новый тип"]


def test_candidate_prompt_allows_distinct_named_product_type_despite_power_source():
    from app.category_classification_ai import (
        build_missing_family_candidate_prompt,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    prompt = build_missing_family_candidate_prompt(
        category_name="Тестовая категория",
        classification=classification,
        candidate={
            "term": "отдельный тип товара",
            "support": 5,
            "examples": [
                "отдельный тип товара alpha",
                "отдельный тип товара beta",
            ],
        },
    )

    assert (
        "тип питания сам по себе недостаточен"
        in prompt
    )

    assert (
        "самостоятельным названием типа товара"
        in prompt
    )

    assert (
        "отличается конструкцией или назначением"
        in prompt
    )


def test_dataframe_group_limits_missing_family_candidates_to_five(
    tmp_path,
    monkeypatch,
):
    cache_path = tmp_path / "categories.json"

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category="Дом и сад",
    )

    dataframe = pd.DataFrame(
        {
            "root_category": ["Дом и сад"] * 3,
            "leaf_category": ["Тестовая категория"] * 3,
            "product_name": [
                "Существующий тип",
                "Новый тип Alpha",
                "Новый тип Beta",
            ],
        }
    )

    seen = {}

    def fake_selector(
        candidates,
        classification,
        max_candidates=10,
    ):
        seen["max_candidates"] = max_candidates
        return []

    monkeypatch.setattr(
        "app.category_classification_ai."
        "select_missing_family_candidates",
        fake_selector,
    )

    monkeypatch.setattr(
        "app.category_classification_ai."
        "generate_missing_family_discovery",
        lambda **kwargs: (),
    )

    monkeypatch.setattr(
        "app.category_classification_ai."
        "generate_category_repair",
        lambda category_name,
               classification,
               unresolved_products,
               resolution_evidence,
               client,
               model="gpt-5-nano": classification,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "resolutions": [
                            {
                                "product_name": "новый тип alpha",
                                "family_name": "existing_family",
                                "confidence": 0.9,
                            },
                            {
                                "product_name": "новый тип beta",
                                "family_name": "existing_family",
                                "confidence": 0.9,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    classify_category_dataframe_group(
        dataframe=dataframe,
        category_name="Тестовая категория",
        root_category="Дом и сад",
        client=fake_client,
        cache_path=cache_path,
        enrich_cached=True,
    )

    assert seen["max_candidates"] == 5


def test_collect_repeated_term_candidates_tracks_prefix_support():
    from app.category_classification_ai import (
        collect_repeated_term_candidates,
    )

    result = collect_repeated_term_candidates(
        [
            "Новый тип Alpha",
            "Новый тип Beta",
            "Новый тип Gamma",
            "Товар бренда Alpha",
            "Другой товар Alpha",
        ],
        min_support=2,
    )

    by_term = {
        item["term"]: item
        for item in result
    }

    assert by_term["новый тип"]["support"] == 3
    assert by_term["новый тип"]["prefix_support"] == 3

    assert by_term["alpha"]["support"] >= 2
    assert by_term["alpha"]["prefix_support"] == 0


def test_select_missing_family_candidates_prefers_prefix_supported_term():
    from app.category_classification_ai import (
        select_missing_family_candidates,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    candidates = [
        {
            "term": "частое слово",
            "support": 12,
            "prefix_support": 0,
            "examples": [
                "товар частое слово alpha",
                "другой частое слово beta",
            ],
        },
        {
            "term": "отдельный тип",
            "support": 8,
            "prefix_support": 8,
            "examples": [
                "отдельный тип alpha",
                "отдельный тип beta",
            ],
        },
    ]

    result = select_missing_family_candidates(
        candidates,
        classification=classification,
        max_candidates=1,
    )

    assert result[0]["term"] == "отдельный тип"


def test_repeated_candidate_discovery_updates_working_classification(
    monkeypatch,
):
    from app.category_classification_ai import (
        discover_missing_families_from_repeated_candidates,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="existing_family",
                keywords=("существующий тип",),
            ),
        ),
        confidence=0.8,
    )

    candidates = [
        {
            "term": "новый тип",
            "support": 5,
            "examples": [
                "новый тип alpha",
                "новый тип beta",
            ],
        },
        {
            "term": "характеристика",
            "support": 4,
            "examples": [
                "новый тип характеристика alpha",
                "новый тип характеристика beta",
            ],
        },
    ]

    seen_family_names = []

    def fake_validate(
        category_name,
        classification,
        candidate,
        client,
        model="gpt-5-nano",
    ):
        names = {
            rule.name
            for rule in classification.functional_families
        }

        seen_family_names.append(names)

        if candidate["term"] == "новый тип":
            return FunctionalFamilyRule(
                name="new_family",
                keywords=("новый тип",),
            )

        return None

    monkeypatch.setattr(
        "app.category_classification_ai."
        "validate_missing_family_candidate",
        fake_validate,
    )

    result = discover_missing_families_from_repeated_candidates(
        category_name="Тестовая категория",
        classification=classification,
        candidates=candidates,
        client=object(),
    )

    assert "new_family" not in seen_family_names[0]
    assert "new_family" in seen_family_names[1]

    assert result == (
        FunctionalFamilyRule(
            name="new_family",
            keywords=("новый тип",),
        ),
    )


def test_category_matching_prefers_unique_prefix_family_over_secondary_match():
    from app.category_classifier import (
        apply_category_classification,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="secondary_family",
                keywords=("поверхностный насос",),
            ),
            FunctionalFamilyRule(
                name="explicit_product_type",
                keywords=("мотопомпа",),
            ),
        ),
        confidence=0.9,
    )

    dataframe = pd.DataFrame(
        {
            "product_name": [
                (
                    "Мотопомпа бензиновая / "
                    "Поверхностный насос GP-50"
                ),
            ],
        }
    )

    result = apply_category_classification(
        dataframe,
        classification,
    )

    row = result.iloc[0]

    assert (
        row["functional_family_status"]
        == "matched"
    )

    assert (
        row["functional_family"]
        == "explicit_product_type"
    )


def test_category_matching_keeps_ambiguity_when_multiple_prefix_families_match():
    from app.category_classifier import (
        apply_category_classification,
    )

    classification = CategoryClassification(
        category_name="Тестовая категория",
        category_type="mixed",
        functional_families=(
            FunctionalFamilyRule(
                name="family_a",
                keywords=("новый",),
            ),
            FunctionalFamilyRule(
                name="family_b",
                keywords=("новый тип",),
            ),
        ),
        confidence=0.9,
    )

    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Новый тип товара",
            ],
        }
    )

    result = apply_category_classification(
        dataframe,
        classification,
    )

    row = result.iloc[0]

    assert (
        row["functional_family_status"]
        == "ambiguous"
    )

    assert pd.isna(
        row["functional_family"]
    )
