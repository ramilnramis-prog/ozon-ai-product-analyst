import json
import pandas as pd

from types import SimpleNamespace

from app.category_classifier import (
    CategoryClassification,
    FunctionalFamilyRule,
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

def test_generate_family_resolutions_rejects_missing_product():
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
        match="AI omitted products",
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
def test_generate_family_resolutions_rejects_duplicate_product():
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

    with pytest.raises(
        ValueError,
        match="AI returned duplicate product",
    ):
        generate_family_resolutions(
            product_names=[
                "Товар A",
            ],
            allowed_families=(
                "cultivator",
                "walk_behind_tractor",
            ),
            client=fake_client,
        )

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

def test_generate_category_repair_rejects_removed_existing_family():
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

    with pytest.raises(
        ValueError,
        match="Repair removed existing functional families",
    ):
        generate_category_repair(
            category_name=(
                "Мотоблоки, культиваторы и электротяпки"
            ),
            classification=classification,
            unresolved_products=[
                "MATAKLA Электротяпка",
            ],
            client=fake_client,
        )

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