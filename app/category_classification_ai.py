import json
import pandas as pd

from pathlib import Path
from app.niche_grouping import normalize_niche_text
from app.category_classification_schema import (
    CATEGORY_CLASSIFICATION_JSON_SCHEMA,
    FAMILY_RESOLUTION_JSON_SCHEMA,
)
from app.category_classifier import (
    DEFAULT_CATEGORY_CACHE_PATH,
    CategoryClassification,
    FunctionalFamilyRule,
    remove_brand_keywords,
    apply_category_classification,
    apply_family_resolutions,
    get_cached_category_classification,
    save_category_classification,
)


def build_category_classification_prompt(
    category_name: object,
    product_examples: list[object],
    max_examples: int = 20,
) -> str:
    """
    Формирует prompt для AI-классификации
    неизвестной товарной категории.

    API здесь не вызывается.
    """

    normalized_category = normalize_niche_text(
        category_name
    )

    examples = []

    for value in product_examples:
        normalized = normalize_niche_text(value)

        if not normalized:
            continue

        if normalized in examples:
            continue

        examples.append(normalized)

        if len(examples) >= max_examples:
            break

    examples_json = json.dumps(
        examples,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Ты классифицируешь товарные категории маркетплейса.

CATEGORY:
{normalized_category}

PRODUCT EXAMPLES:
{examples_json}

ЗАДАЧА:

1. Определи тип категории с точки зрения
анализа конкуренции продавца.

homogeneous
— товары являются прямыми или очень близкими
заменителями друг друга и обычно конкурируют
за одну и ту же покупательскую задачу.

mixed
— категория объединяет несколько разных
функциональных типов товаров, которые нельзя
считать одной конкурентной нишей.

unknown
— по названию категории и примерам
нельзя уверенно определить структуру.

2. Не объединяй разные типы товаров
только потому, что они относятся
к одной общей тематике.

Например:
- мотоблок и культиватор — разные functional_family;
- культиватор и электротяпка — разные functional_family;
- мойка высокого давления и аксессуар к мойке —
  разные functional_family;
- основной товар и аксессуар к нему —
  разные functional_family.

3. Если в названии категории или среди примеров
есть несколько устойчивых названий разных
типов товара, считай категорию mixed,
если эти названия не являются очевидными
синонимами одного и того же товара.

Не создавай общее семейство вроде:
- cultivation_machinery;
- garden_equipment;
- tools;
- accessories_and_equipment,

если внутри можно выделить более конкретные
конкурентные товарные семейства.

4. Если категория homogeneous:
верни одно конкретное functional_family.

Если категория mixed:
выдели отдельный functional_family
для каждого реального конкурентного типа товара.

Для каждого functional_family укажи
минимальный набор ключевых слов,
по которым Python сможет определить
семейство по названию товара.

5. Не дели товары только по:
- бренду;
- цвету;
- размеру;
- комплектации;
- мощности;
- количеству аксессуаров.

6. Не создавай отдельные functional_family
для battery / petrol / electric.
Тип питания определяется системой отдельно.

7. Названия functional_family пиши:
- на английском;
- lowercase;
- snake_case;
- коротко и стабильно.

8. confidence должен быть числом от 0 до 1.

Не оценивай:
- спрос;
- продажи;
- конкуренцию;
- прибыльность;
- перспективность товара.

Твоя задача только определить
функциональную структуру категории.
""".strip()

def generate_category_classification(
    category_name: object,
    product_examples: list[object],
    client,
    model: str = "gpt-5-nano",
) -> CategoryClassification:
    """
    Классифицирует неизвестную товарную категорию
    через Structured Output.

    Функция не рассчитывает спрос, конкуренцию
    или Opportunity Score.
    """

    prompt = build_category_classification_prompt(
        category_name=category_name,
        product_examples=product_examples,
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "category_classification",
                "schema": CATEGORY_CLASSIFICATION_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(
        response.output_text
    )

    functional_families = tuple(
        FunctionalFamilyRule(
            name=family["name"],
            keywords=tuple(
                family["keywords"]
            ),
        )
        for family in data[
            "functional_families"
        ]
    )

    return CategoryClassification(
        category_name=str(category_name).strip(),
        category_type=data["category_type"],
        functional_families=functional_families,
        confidence=float(
            data["confidence"]
        ),
    )
def classify_category(
    category_name: object,
    product_examples: list[object],
    client,
    cache_path: Path = DEFAULT_CATEGORY_CACHE_PATH,
    model: str = "gpt-5-nano",
    root_category: object = None,
) -> CategoryClassification:
    """
    Возвращает классификацию категории.

    Сначала проверяет локальный кэш.
    AI вызывается только для новой категории.
    Новый результат сохраняется в кэш.
    """

    cached = get_cached_category_classification(
        category_name,
        cache_path,
        root_category=root_category,
    )

    if cached is not None:
        return cached

    classification = generate_category_classification(
        category_name=category_name,
        product_examples=product_examples,
        client=client,
        model=model,
    )

    save_category_classification(
        classification,
        cache_path,
        root_category=root_category,
    )

    return classification

def build_family_resolution_prompt(
    product_names: list[object],
    allowed_families: tuple[str, ...],
) -> str:
    """
    Формирует prompt для пакетного определения
    functional_family у неоднозначных товаров.

    API здесь не вызывается.
    """

    products = []

    for value in product_names:
        normalized = normalize_niche_text(value)

        if not normalized:
            continue

        if normalized in products:
            continue

        products.append(normalized)

    products_json = json.dumps(
        products,
        ensure_ascii=False,
        indent=2,
    )

    families_json = json.dumps(
        list(allowed_families),
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Ты уточняешь functional_family товаров маркетплейса.

ALLOWED FAMILIES:
{families_json}

PRODUCTS:
{products_json}

ЗАДАЧА:

Для каждого товара выбери ровно одно значение family_name.

Разрешены только:
- одно из значений ALLOWED FAMILIES;
- unresolved, если данных недостаточно.

Не создавай новые functional_family.

Определи реальный тип товара,
а не SEO-слова из названия.

Если название содержит несколько типов товара,
выбери тот, который наиболее вероятно является
основным товаром.

Не используй:
- бренд;
- цвет;
- размер;
- мощность;
- комплектацию;
- тип питания

как основание для создания нового семейства.

Если уверенно выбрать нельзя:
family_name = unresolved.

confidence должен быть числом от 0 до 1.

Верни результат для каждого переданного товара.
""".strip()

def generate_family_resolutions(
    product_names: list[object],
    allowed_families: tuple[str, ...],
    client,
    model: str = "gpt-5-nano",
) -> list[dict[str, object]]:
    """
    Пакетно уточняет functional_family
    для ambiguous/unmatched товаров.
    """

    prompt = build_family_resolution_prompt(
        product_names=product_names,
        allowed_families=allowed_families,
    )
    expected_products = []

    for value in product_names:
        normalized = normalize_niche_text(value)

        if not normalized:
            continue

        if normalized not in expected_products:
            expected_products.append(normalized)

    expected_product_set = set(
        expected_products
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "family_resolution",
                "schema": FAMILY_RESOLUTION_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(
        response.output_text
    )

    allowed = set(
        allowed_families
    ) | {"unresolved"}

    resolutions = []
    returned_products = []

    for item in data["resolutions"]:
        family_name = item["family_name"]

        if family_name not in allowed:
            raise ValueError(
                f"Unexpected functional family: {family_name}"
            )

        confidence = float(
            item["confidence"]
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Resolution confidence must be between 0 and 1"
            )

        normalized_product_name = normalize_niche_text(
            item["product_name"]
        )

        if normalized_product_name not in expected_product_set:
            raise ValueError(
                "AI returned unexpected product: "
                f"{item['product_name']}"
            )

        if normalized_product_name in returned_products:
            raise ValueError(
                "AI returned duplicate product: "
                f"{item['product_name']}"
            )

        returned_products.append(
            normalized_product_name
        )

        resolutions.append(
            {
                "product_name": normalized_product_name,
                "family_name": family_name,
                "confidence": confidence,
            }
        )

    missing_products = (
        expected_product_set
        - set(returned_products)
    )

    if missing_products:
        raise ValueError(
            "AI omitted products: "
            + ", ".join(
                sorted(missing_products)
            )
        )

    return resolutions

def collect_category_examples(
    dataframe: pd.DataFrame,
    leaf_category: object,
    max_examples: int = 20,
    root_category: object = None,
) -> list[str]:
    """
    Собирает уникальные примеры товаров
    одной leaf_category для AI-классификации.
    """

    if (
        "leaf_category" not in dataframe.columns
        or "product_name" not in dataframe.columns
    ):
        return []

    normalized_category = normalize_niche_text(
        leaf_category
    )

    if not normalized_category:
        return []

    category_mask = (
        dataframe["leaf_category"]
        .map(normalize_niche_text)
        .eq(normalized_category)
    )

    normalized_root = normalize_niche_text(
        root_category
    )

    if (
        normalized_root
        and "root_category" in dataframe.columns
    ):
        category_mask = (
            category_mask
            & dataframe["root_category"]
            .map(normalize_niche_text)
            .eq(normalized_root)
        )

    examples = (
        dataframe.loc[
            category_mask,
            "product_name",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(max_examples)
        .tolist()
    )

    return examples

def classify_category_dataframe_group(
    dataframe: pd.DataFrame,
    category_name: object,
    client,
    cache_path: Path = DEFAULT_CATEGORY_CACHE_PATH,
    model: str = "gpt-5-nano",
    root_category: object = None,
) -> pd.DataFrame:
    """
    Полностью классифицирует товары
    одной leaf_category.
    """

    examples = collect_category_examples(
        dataframe,
        leaf_category=category_name,
        root_category=root_category,
    )

    classification = classify_category(
        category_name=category_name,
        product_examples=examples,
        client=client,
        cache_path=cache_path,
        model=model,
        root_category=root_category,
    )

    normalized_category = normalize_niche_text(
        category_name
    )

    category_mask = (
        dataframe["leaf_category"]
        .map(normalize_niche_text)
        .eq(normalized_category)
    )

    normalized_root = normalize_niche_text(
        root_category
    )

    if (
        normalized_root
        and "root_category" in dataframe.columns
    ):
        category_mask = (
            category_mask
            & dataframe["root_category"]
            .map(normalize_niche_text)
            .eq(normalized_root)
        )

    result = dataframe.loc[
        category_mask
    ].copy()

    if "brand" in result.columns:
        classification = remove_brand_keywords(
            classification,
            brand_values=(
                result["brand"]
                .dropna()
                .tolist()
            ),
        )

        save_category_classification(
            classification,
            cache_path,
            root_category=root_category,
        )

    result = apply_category_classification(
        result,
        classification,
    )

    result["category_type"] = (
        classification.category_type
    )

    result[
        "category_classification_confidence"
    ] = classification.confidence

    repair_candidate_index = result.index[
        result["functional_family_status"].isin(
            [
                "ambiguous",
                "unmatched",
            ]
        )
    ]

    needs_resolution = (
        result.loc[
            repair_candidate_index,
            "product_name",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    allowed_families = tuple(
        rule.name
        for rule in classification.functional_families
    )

    if (
        needs_resolution
        and allowed_families
    ):
        resolutions = generate_family_resolutions(
            product_names=needs_resolution,
            allowed_families=allowed_families,
            client=client,
            model=model,
        )

        result = apply_family_resolutions(
            result,
            resolutions,
        )

        repaired_classification = (
            generate_category_repair(
                category_name=category_name,
                classification=classification,
                unresolved_products=needs_resolution,
                client=client,
                model=model,
            )
        )

        if "brand" in result.columns:
            repaired_classification = (
                remove_brand_keywords(
                    repaired_classification,
                    brand_values=(
                        result["brand"]
                        .dropna()
                        .tolist()
                    ),
                )
            )

        save_category_classification(
            repaired_classification,
            cache_path,
            root_category=root_category,
        )

        repaired_rows = (
            apply_category_classification(
                result.loc[
                    repair_candidate_index
                ].copy(),
                repaired_classification,
            )
        )

        repaired_rows = repaired_rows.loc[
            repaired_rows[
                "functional_family_status"
            ].eq("matched")
        ]

        result.loc[
            repaired_rows.index,
            "functional_family",
        ] = repaired_rows[
            "functional_family"
        ]

        result.loc[
            repaired_rows.index,
            "functional_family_status",
        ] = repaired_rows[
            "functional_family_status"
        ]

        result[
            "category_classification_confidence"
        ] = repaired_classification.confidence

    return result
def build_category_repair_prompt(
    category_name: object,
    classification: CategoryClassification,
    unresolved_products: list[object],
) -> str:
    """
    Формирует prompt для проверки,
    не пропустила ли классификация
    отдельное functional_family.
    """

    existing_families = [
        {
            "name": rule.name,
            "keywords": list(rule.keywords),
        }
        for rule in classification.functional_families
    ]

    unresolved = []

    for value in unresolved_products:
        normalized = normalize_niche_text(value)

        if not normalized:
            continue

        if normalized in unresolved:
            continue

        unresolved.append(normalized)

    return f"""
Ты проверяешь уже существующую классификацию
товарной категории маркетплейса.

CATEGORY:
{normalize_niche_text(category_name)}

CURRENT FUNCTIONAL FAMILIES:
{json.dumps(
    existing_families,
    ensure_ascii=False,
    indent=2,
)}

UNRESOLVED PRODUCTS:
{json.dumps(
    unresolved,
    ensure_ascii=False,
    indent=2,
)}

ЗАДАЧА:

Проверь, не пропущено ли отдельное
functional_family.

Важно:

1. Не удаляй существующие functional_family.

2. Не переименовывай существующие
functional_family.

3. Добавляй новое семейство только если
unresolved товар действительно представляет
отдельный конкурентный тип товара.

4. Не создавай новое семейство только из-за:
- бренда;
- цвета;
- размера;
- мощности;
- комплектации;
- battery / petrol / electric.

5. Основной товар и аксессуар могут быть
разными functional_family.

6. Если нового семейства не требуется,
сохрани текущую структуру без изменений.

7. Верни ПОЛНУЮ обновленную классификацию,
включая все существующие семейства.

8. Названия новых functional_family:
- английский;
- lowercase;
- snake_case;
- короткие и стабильные.

9. confidence — число от 0 до 1.

Не анализируй спрос, продажи,
конкуренцию или прибыльность.
""".strip()

def generate_category_repair(
    category_name: object,
    classification: CategoryClassification,
    unresolved_products: list[object],
    client,
    model: str = "gpt-5-nano",
) -> CategoryClassification:
    """
    Проверяет существующую классификацию
    и при необходимости добавляет
    пропущенные functional_family.
    """

    prompt = build_category_repair_prompt(
        category_name=category_name,
        classification=classification,
        unresolved_products=unresolved_products,
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "category_repair",
                "schema": CATEGORY_CLASSIFICATION_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(
        response.output_text
    )

    repaired_rules = tuple(
        FunctionalFamilyRule(
            name=family["name"],
            keywords=tuple(
                family["keywords"]
            ),
        )
        for family in data[
            "functional_families"
        ]
    )

    existing_names = {
        rule.name
        for rule in classification.functional_families
    }

    repaired_names = {
        rule.name
        for rule in repaired_rules
    }

    missing_existing = (
        existing_names
        - repaired_names
    )

    if missing_existing:
        raise ValueError(
            "Repair removed existing functional families: "
            + ", ".join(
                sorted(missing_existing)
            )
        )

    if (
        data["category_type"]
        != classification.category_type
    ):
        raise ValueError(
            "Repair changed category_type"
        )

    confidence = float(
        data["confidence"]
    )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Repair confidence must be between 0 and 1"
        )

    return CategoryClassification(
        category_name=str(category_name).strip(),
        category_type=data["category_type"],
        functional_families=repaired_rules,
        confidence=confidence,
    )