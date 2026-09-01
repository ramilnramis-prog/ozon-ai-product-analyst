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
    family_rules: tuple[FunctionalFamilyRule, ...] | None = None,
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

    allowed_family_set = set(
        allowed_families
    )

    family_definitions = []

    for rule in family_rules or ():
        if rule.name not in allowed_family_set:
            continue

        family_definitions.append(
            {
                "name": rule.name,
                "keywords": list(rule.keywords),
            }
        )

    family_definitions_json = json.dumps(
        family_definitions,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Ты уточняешь functional_family товаров маркетплейса.

ALLOWED FAMILIES:
{families_json}

FAMILY DEFINITIONS:
{family_definitions_json}

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
Используй FAMILY DEFINITIONS как описание
границ уже существующих functional_family.

Keywords являются подсказками и примерами,
а не обязательным условием совпадения.

Не относись к family только потому, что она
семантически "ближайшая".

Если товар по своему реальному типу не подходит
ни к одной существующей family, верни unresolved.
family_name = unresolved.

confidence должен быть числом от 0 до 1.

Верни результат для каждого переданного товара.
""".strip()

def generate_family_resolutions(
    product_names: list[object],
    allowed_families: tuple[str, ...],
    client,
    model: str = "gpt-5-nano",
    family_rules: tuple[FunctionalFamilyRule, ...] | None = None,
) -> list[dict[str, object]]:
    """
    Пакетно уточняет functional_family
    для ambiguous/unmatched товаров.
    """

    prompt = build_family_resolution_prompt(
        product_names=product_names,
        allowed_families=allowed_families,
        family_rules=family_rules,
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

    resolution_schema = json.loads(
        json.dumps(
            FAMILY_RESOLUTION_JSON_SCHEMA
        )
    )

    resolution_properties = (
        resolution_schema[
            "properties"
        ][
            "resolutions"
        ][
            "items"
        ][
            "properties"
        ]
    )

    resolution_properties[
        "product_name"
    ]["enum"] = expected_products

    resolution_properties[
        "family_name"
    ]["enum"] = list(
        dict.fromkeys(
            (
                *allowed_families,
                "unresolved",
            )
        )
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "family_resolution",
                "schema": resolution_schema,
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
            existing_resolution = next(
                resolution
                for resolution in resolutions
                if (
                    resolution["product_name"]
                    == normalized_product_name
                )
            )

            if (
                existing_resolution["family_name"]
                == family_name
            ):
                existing_resolution["confidence"] = max(
                    float(existing_resolution["confidence"]),
                    confidence,
                )
            else:
                existing_resolution[
                    "family_name"
                ] = "unresolved"
                existing_resolution[
                    "confidence"
                ] = 0.0

            continue

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

    returned_product_set = set(
        returned_products
    )

    for product_name in expected_products:
        if product_name in returned_product_set:
            continue

        resolutions.append(
            {
                "product_name": product_name,
                "family_name": "unresolved",
                "confidence": 0.0,
            }
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
    enrich_cached: bool = False,
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

    cached_classification = (
        get_cached_category_classification(
            category_name,
            cache_path,
            root_category=root_category,
        )
    )

    was_cached = (
        cached_classification is not None
    )

    allow_enrichment = (
       not was_cached
       or enrich_cached
    )

    if was_cached:
        classification = cached_classification
    else:
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
    if allow_enrichment:
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

    deterministic_before = result.copy()

    if (
        allow_enrichment
        and needs_resolution
        and allowed_families
    ):
        resolutions = generate_family_resolutions(
            product_names=needs_resolution,
            allowed_families=allowed_families,
            family_rules=classification.functional_families,
            client=client,
            model=model,
        )

        result = apply_family_resolutions(
            result,
            resolutions,
        )

        repeated_candidates = (
            collect_repeated_term_candidates(
                needs_resolution,
                min_support=2,
            )
        )

        selected_candidates = (
            select_missing_family_candidates(
                repeated_candidates,
                classification=classification,
                max_candidates=5,
            )
        )

        candidate_discovered_families = (
            discover_missing_families_from_repeated_candidates(
                category_name=category_name,
                classification=classification,
                candidates=selected_candidates,
                client=client,
                model=model,
            )
        )

        classification_for_open_discovery = (
            CategoryClassification(
                category_name=classification.category_name,
                category_type=classification.category_type,
                functional_families=tuple(
                    (
                        *classification.functional_families,
                        *candidate_discovered_families,
                    )
                ),
                confidence=classification.confidence,
            )
        )

        discovered_families = (
            generate_missing_family_discovery(
                category_name=category_name,
                classification=classification_for_open_discovery,
                problem_products=needs_resolution,
                provisional_resolutions=resolutions,
                client=client,
                model=model,
            )
        )

        classification_for_repair = (
            CategoryClassification(
                category_name=classification.category_name,
                category_type=classification.category_type,
                functional_families=tuple(
                    (
                        *classification_for_open_discovery.functional_families,
                        *discovered_families,
                    )
                ),
                confidence=classification.confidence,
            )
        )

        repaired_classification = (
            generate_category_repair(
                category_name=category_name,
                classification=classification_for_repair,
                resolution_evidence=resolutions,
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

        original_rules_by_name = {
            rule.name: rule
            for rule in classification.functional_families
        }

        repaired_rules_by_name = {
            rule.name: rule
            for rule in repaired_classification.functional_families
        }

        rule_order = [
            rule.name
            for rule in classification.functional_families
        ]

        # Новые functional_family, найденные repair,
        # сохраняем как кандидатов. Общий quality gate
        # ниже всё равно проверит их влияние на категорию.
        for rule in repaired_classification.functional_families:
            if rule.name not in rule_order:
                rule_order.append(rule.name)

        accepted_rules_by_name = {
            rule.name: rule
            for rule in classification.functional_families
        }

        for rule in repaired_classification.functional_families:
            if rule.name not in accepted_rules_by_name:
                accepted_rules_by_name[rule.name] = rule

        def build_incremental_classification():
            return CategoryClassification(
                category_name=(
                    repaired_classification.category_name
                ),
                category_type=(
                    repaired_classification.category_type
                ),
                functional_families=tuple(
                    accepted_rules_by_name[name]
                    for name in rule_order
                ),
                confidence=(
                    repaired_classification.confidence
                ),
            )

        def get_status_metrics(frame):
            counts = frame[
                "functional_family_status"
            ].value_counts()

            return (
                int(counts.get("matched", 0)),
                int(counts.get("ambiguous", 0)),
                int(counts.get("unmatched", 0)),
            )

        current_classification = (
            build_incremental_classification()
        )

        current_deterministic = (
            apply_category_classification(
                deterministic_before.copy(),
                current_classification,
            )
        )

        current_metrics = get_status_metrics(
            current_deterministic
        )

        keyword_candidates = []

        for repaired_rule in (
            repaired_classification.functional_families
        ):
            original_rule = original_rules_by_name.get(
                repaired_rule.name
            )

            # Keywords нового family пока не режем
            # по одному. Вся новая family проверяется
            # общим quality gate.
            if original_rule is None:
                continue

            original_keywords = {
                normalize_niche_text(keyword)
                for keyword in original_rule.keywords
            }

            for position, keyword in enumerate(
                repaired_rule.keywords
            ):
                normalized_keyword = (
                    normalize_niche_text(keyword)
                )

                if not normalized_keyword:
                    continue

                if normalized_keyword in original_keywords:
                    continue

                keyword_candidates.append(
                    (
                        repaired_rule.name,
                        keyword,
                        normalized_keyword,
                        position,
                    )
                )

        # Более конкретные фразы проверяем первыми.
        # Например "станция водоснабжения"
        # должна получить шанс раньше,
        # чем одиночное "автоматическая".
        keyword_candidates.sort(
            key=lambda item: (
                -len(item[2].split()),
                -len(item[2]),
                item[3],
            )
        )

        for (
            family_name,
            keyword,
            normalized_keyword,
            _,
        ) in keyword_candidates:
            current_rule = accepted_rules_by_name[
                family_name
            ]

            current_normalized_keywords = {
                normalize_niche_text(value)
                for value in current_rule.keywords
            }

            if (
                normalized_keyword
                in current_normalized_keywords
            ):
                continue

            trial_rule = FunctionalFamilyRule(
                name=family_name,
                keywords=tuple(
                    (
                        *current_rule.keywords,
                        keyword,
                    )
                ),
            )

            previous_rule = accepted_rules_by_name[
                family_name
            ]

            accepted_rules_by_name[
                family_name
            ] = trial_rule

            trial_classification = (
                build_incremental_classification()
            )

            trial_deterministic = (
                apply_category_classification(
                    deterministic_before.copy(),
                    trial_classification,
                )
            )

            trial_metrics = get_status_metrics(
                trial_deterministic
            )

            (
                current_matched,
                current_ambiguous,
                current_unmatched,
            ) = current_metrics

            (
                trial_matched,
                trial_ambiguous,
                trial_unmatched,
            ) = trial_metrics

            incrementally_useful = (
                trial_matched >= current_matched
                and trial_ambiguous
                <= current_ambiguous
                and trial_unmatched
                <= current_unmatched
                and (
                    trial_matched > current_matched
                    or trial_ambiguous
                    < current_ambiguous
                    or trial_unmatched
                    < current_unmatched
                )
            )

            if incrementally_useful:
                current_classification = (
                    trial_classification
                )
                current_deterministic = (
                    trial_deterministic
                )
                current_metrics = trial_metrics
            else:
                accepted_rules_by_name[
                    family_name
                ] = previous_rule

        repaired_classification = (
            build_incremental_classification()
        )

        candidate_deterministic = (
            apply_category_classification(
                deterministic_before.copy(),
                repaired_classification,
            )
        )

        before_counts = (
            deterministic_before[
                "functional_family_status"
            ].value_counts()
        )

        candidate_counts = (
            candidate_deterministic[
                "functional_family_status"
            ].value_counts()
        )

        before_matched = int(
            before_counts.get("matched", 0)
        )
        before_ambiguous = int(
            before_counts.get("ambiguous", 0)
        )
        before_unmatched = int(
            before_counts.get("unmatched", 0)
        )

        candidate_matched = int(
            candidate_counts.get("matched", 0)
        )
        candidate_ambiguous = int(
            candidate_counts.get("ambiguous", 0)
        )
        candidate_unmatched = int(
            candidate_counts.get("unmatched", 0)
        )

        quality_gate_passed = (
            candidate_matched >= before_matched
            and candidate_ambiguous <= before_ambiguous
            and candidate_unmatched <= before_unmatched
            and (
                candidate_matched > before_matched
                or candidate_ambiguous < before_ambiguous
                or candidate_unmatched < before_unmatched
            )
        )

        if quality_gate_passed:
            save_category_classification(
                repaired_classification,
                cache_path,
                root_category=root_category,
            )

            repaired_rows = (
                candidate_deterministic.loc[
                    repair_candidate_index
                ].copy()
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


def collect_repeated_term_candidates(
    product_names: list[object],
    min_support: int = 2,
    max_ngram: int = 3,
    max_candidates: int = 50,
) -> list[dict[str, object]]:
    """
    Собирает повторяющиеся термины и короткие фразы
    из разных problem products.

    Функция только находит статистически повторяющиеся
    текстовые кандидаты. Она не решает, является ли
    кандидат отдельной functional_family.
    """

    if min_support < 1:
        raise ValueError(
            "min_support must be at least 1"
        )

    if max_ngram < 1:
        raise ValueError(
            "max_ngram must be at least 1"
        )

    stopwords = {
        "а",
        "без",
        "в",
        "во",
        "для",
        "до",
        "и",
        "из",
        "или",
        "на",
        "над",
        "от",
        "по",
        "под",
        "при",
        "с",
        "со",
        "the",
        "for",
        "with",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
    }

    normalized_products = []

    for value in product_names:
        product = normalize_niche_text(value)

        if not product:
            continue

        if product in normalized_products:
            continue

        normalized_products.append(product)

    candidate_products = {}

    for product in normalized_products:
        tokens = [
            token
            for token in product.split()
            if token.isalpha()
        ]

        product_candidates = set()

        for size in range(
            1,
            min(max_ngram, len(tokens)) + 1,
        ):
            for start in range(
                0,
                len(tokens) - size + 1,
            ):
                window = tokens[
                    start:start + size
                ]

                if size == 1:
                    token = window[0]

                    if token in stopwords:
                        continue

                    if len(token) < 4:
                        continue

                else:
                    content_tokens = [
                        token
                        for token in window
                        if token not in stopwords
                        and len(token) >= 3
                    ]

                    if not content_tokens:
                        continue

                    if window[0] in stopwords:
                        continue

                    if window[-1] in stopwords:
                        continue

                term = " ".join(window)

                product_candidates.add(term)

        for term in product_candidates:
            candidate_products.setdefault(
                term,
                [],
            ).append(product)

    result = []

    for term, examples in candidate_products.items():
        support = len(examples)

        if support < min_support:
            continue

        prefix_support = sum(
            1
            for example in examples
            if (
                example == term
                or example.startswith(term + " ")
            )
        )

        result.append(
            {
                "term": term,
                "support": support,
                "prefix_support": prefix_support,
                "examples": examples,
            }
        )

    result.sort(
        key=lambda item: (
            -int(item["support"]),
            -len(str(item["term"]).split()),
            -len(str(item["term"])),
            str(item["term"]),
        )
    )

    return result[:max_candidates]



def build_missing_family_candidate_prompt(
    category_name: object,
    classification: CategoryClassification,
    candidate: dict[str, object],
) -> str:
    """
    Формирует prompt для проверки одного
    конкретного кандидата на новую functional_family.
    """

    existing_families = [
        {
            "name": rule.name,
            "keywords": list(rule.keywords),
        }
        for rule in classification.functional_families
    ]

    normalized_candidate = {
        "term": normalize_niche_text(
            candidate.get("term")
        ),
        "support": candidate.get(
            "support",
            0,
        ),
        "examples": [
            normalized
            for value in candidate.get(
                "examples",
                [],
            )
            if (
                normalized := normalize_niche_text(
                    value
                )
            )
        ],
    }

    return f"""
CANDIDATE FAMILY VALIDATION ONLY

Ты проверяешь один конкретный повторяющийся
товарный кандидат внутри категории маркетплейса.

CATEGORY:
{normalize_niche_text(category_name)}

CURRENT FUNCTIONAL FAMILIES:
{json.dumps(
    existing_families,
    ensure_ascii=False,
    indent=2,
)}

CANDIDATE:
{json.dumps(
    normalized_candidate,
    ensure_ascii=False,
    indent=2,
)}

ЗАДАЧА:

оцени только этот кандидат.

Определи, представляет ли он отдельный
устойчивый функциональный тип товара,
который действительно отличается от
CURRENT FUNCTIONAL FAMILIES.

не ищи другие новые family.

Повторяемость термина является только сигналом,
а не доказательством новой family.

Новая family оправдана только если:

- examples представляют один реальный тип товара;
- у этого типа есть общее функциональное назначение;
- существующие family описывают другой тип товара;
- для включения кандидата в существующую family
  пришлось бы слишком широко трактовать её смысл.

Не создавай новую family только из-за:

- бренда;
- модели;
- цвета;
- размера;
- мощности;
- комплектации.

тип питания сам по себе недостаточен
для создания отдельной functional_family.

Но не отклоняй кандидат автоматически только
потому, что его тип товара часто связан
с определённым источником питания.

Если повторяющийся термин является
самостоятельным названием типа товара
и этот тип функционально отличается от
существующих family, кандидат может быть
отдельной functional_family.

Считай различие функциональным, если тип
отличается конструкцией или назначением,
способом эксплуатации либо представляет
отдельный устойчивый класс товара,
а не просто вариант мощности или питания.

Если кандидат является лишь синонимом,
вариантом названия или характеристикой
существующей family, отклони его.

Не анализируй продажи, спрос,
конкуренцию или прибыльность.
""".strip()



def validate_missing_family_candidate(
    category_name: object,
    classification: CategoryClassification,
    candidate: dict[str, object],
    client,
    model: str = "gpt-5-nano",
) -> FunctionalFamilyRule | None:
    """
    Проверяет один повторяющийся текстовый кандидат
    на отдельную functional_family.

    AI оценивает семантику.
    Python проверяет минимальную поддержку
    и не допускает перезапись существующей family.
    """

    try:
        support = int(
            candidate.get("support", 0)
        )
    except (TypeError, ValueError):
        return None

    if support < 2:
        return None

    candidate_term = normalize_niche_text(
        candidate.get("term")
    )

    if not candidate_term:
        return None

    examples = []

    for value in candidate.get(
        "examples",
        [],
    ):
        normalized = normalize_niche_text(
            value
        )

        if not normalized:
            continue

        if normalized in examples:
            continue

        examples.append(normalized)

    if len(examples) < 2:
        return None

    prompt = build_missing_family_candidate_prompt(
        category_name=category_name,
        classification=classification,
        candidate={
            "term": candidate_term,
            "support": support,
            "examples": examples,
        },
    )

    validation_schema = {
        "type": "object",
        "properties": {
            "accept": {
                "type": "boolean",
            },
            "family_name": {
                "type": "string",
            },
            "keywords": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
        "required": [
            "accept",
            "family_name",
            "keywords",
        ],
        "additionalProperties": False,
    }

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": (
                        "missing_family_candidate_validation"
                    ),
                    "schema": validation_schema,
                    "strict": True,
                }
            },
        )
    except Exception as exc:
        if exc.__class__.__name__ == "APITimeoutError":
            return None
        raise

    data = json.loads(
        response.output_text
    )

    if not data["accept"]:
        return None

    family_name = str(
        data["family_name"]
    ).strip().lower()

    family_name = (
        family_name
        .replace("-", "_")
        .replace(" ", "_")
    )

    if not family_name:
        return None

    existing_names = {
        rule.name
        for rule in classification.functional_families
    }

    if family_name in existing_names:
        return None

    proposed_keywords = [
        candidate_term
    ]

    for value in data["keywords"]:
        keyword = normalize_niche_text(
            value
        )

        if not keyword:
            continue

        # Дополнительный однословный keyword от AI слишком легко
        # оказывается характеристикой, а не типом товара.
        # Сам candidate_term уже добавлен отдельно и не теряется.
        if keyword != candidate_term and len(keyword.split()) < 2:
            continue

        if keyword in proposed_keywords:
            continue

        proposed_keywords.append(keyword)

    accepted_keywords = []

    for keyword in proposed_keywords:
        keyword_support = sum(
            1
            for example in examples
            if keyword in example
        )

        if keyword_support < 2:
            continue

        accepted_keywords.append(keyword)

    if not accepted_keywords:
        return None

    return FunctionalFamilyRule(
        name=family_name,
        keywords=tuple(
            accepted_keywords
        ),
    )




def select_missing_family_candidates(
    candidates: list[dict[str, object]],
    classification: CategoryClassification,
    max_candidates: int = 10,
) -> list[dict[str, object]]:
    """
    Отбирает ограниченный набор неперекрывающихся
    повторяющихся терминов для AI-проверки.

    Python здесь не решает, является ли термин
    новой functional_family. Он только сокращает
    дубли и уже известные правила.
    """

    if max_candidates < 1:
        return []

    existing_keywords = {
        normalized
        for rule in classification.functional_families
        for keyword in rule.keywords
        if (
            normalized := normalize_niche_text(
                keyword
            )
        )
    }

    prepared = []

    for candidate in candidates:
        term = normalize_niche_text(
            candidate.get("term")
        )

        if not term:
            continue

        try:
            support = int(
                candidate.get("support", 0)
            )
        except (TypeError, ValueError):
            continue

        if support < 2:
            continue

        # Уже известный deterministic keyword
        # или его самостоятельный словесный фрагмент
        # не должен снова отправляться AI.
        #
        # Например:
        # "станция" входит в "насосная станция";
        # "насос" входит в "поверхностный насос".
        #
        # Используем границы слов, чтобы не считать
        # произвольные подстроки совпадением.
        covered_by_existing_keyword = any(
            (
                f" {term} "
                in f" {existing_keyword} "
            )
            for existing_keyword
            in existing_keywords
        )

        if covered_by_existing_keyword:
            continue

        examples = []

        for value in candidate.get(
            "examples",
            [],
        ):
            normalized = normalize_niche_text(
                value
            )

            if not normalized:
                continue

            if normalized in examples:
                continue

            examples.append(normalized)

        if len(examples) < 2:
            continue

        try:
            prefix_support = int(
                candidate.get("prefix_support", 0)
            )
        except (TypeError, ValueError):
            prefix_support = 0

        prefix_support = max(
            0,
            min(prefix_support, support),
        )

        prepared.append(
            {
                "term": term,
                "support": support,
                "prefix_support": prefix_support,
                "examples": examples,
            }
        )

    # Сильный prefix_support повышает вероятность,
    # что термин является названием самого типа товара,
    # а не брендом, характеристикой или единицей измерения.
    #
    # Это только ранжирование: кандидаты с низким
    # prefix_support не отбрасываются автоматически.
    prepared.sort(
        key=lambda item: (
            -(
                int(item["prefix_support"])
                / int(item["support"])
            ),
            -int(item["prefix_support"]),
            -int(item["support"]),
            len(str(item["term"]).split()),
            len(str(item["term"])),
            str(item["term"]),
        )
    )

    selected = []

    for candidate in prepared:
        term = str(candidate["term"])
        support = int(candidate["support"])

        redundant = False

        for accepted in selected:
            accepted_term = str(
                accepted["term"]
            )
            accepted_support = int(
                accepted["support"]
            )

            # Например:
            # "мотопомпа"
            # и "мотопомпа бензиновая".
            #
            # Если короткий термин уже имеет
            # не меньшую поддержку, длинный
            # кандидат не даёт нового кластера.
            if (
                accepted_support >= support
                and (
                    term.startswith(
                        accepted_term + " "
                    )
                    or term.endswith(
                        " " + accepted_term
                    )
                )
            ):
                redundant = True
                break

        if redundant:
            continue

        selected.append(candidate)

        if len(selected) >= max_candidates:
            break

    return selected


def discover_missing_families_from_repeated_candidates(
    category_name: object,
    classification: CategoryClassification,
    candidates: list[dict[str, object]],
    client,
    model: str = "gpt-5-nano",
) -> tuple[FunctionalFamilyRule, ...]:
    """
    Проверяет повторяющиеся текстовые кандидаты
    по одному и собирает только подтверждённые
    новые functional_family.

    Каждая подтверждённая family сразу добавляется
    во временную working_classification, чтобы
    следующие кандидаты уже знали о ней.
    """

    discovered_by_name = {}
    discovered_order = []

    existing_names = {
        rule.name
        for rule in classification.functional_families
    }

    working_classification = classification

    for candidate in candidates:
        rule = validate_missing_family_candidate(
            category_name=category_name,
            classification=working_classification,
            candidate=candidate,
            client=client,
            model=model,
        )

        if rule is None:
            continue

        if rule.name in existing_names:
            continue

        if rule.name not in discovered_by_name:
            discovered_order.append(
                rule.name
            )
            discovered_by_name[
                rule.name
            ] = list(rule.keywords)
        else:
            for keyword in rule.keywords:
                if (
                    keyword
                    not in discovered_by_name[
                        rule.name
                    ]
                ):
                    discovered_by_name[
                        rule.name
                    ].append(keyword)

        discovered_rules = tuple(
            FunctionalFamilyRule(
                name=name,
                keywords=tuple(
                    discovered_by_name[name]
                ),
            )
            for name in discovered_order
        )

        working_classification = (
            CategoryClassification(
                category_name=classification.category_name,
                category_type=classification.category_type,
                functional_families=tuple(
                    (
                        *classification.functional_families,
                        *discovered_rules,
                    )
                ),
                confidence=classification.confidence,
            )
        )

    return tuple(
        FunctionalFamilyRule(
            name=name,
            keywords=tuple(
                discovered_by_name[name]
            ),
        )
        for name in discovered_order
    )


def build_missing_family_discovery_prompt(
    category_name: object,
    classification: CategoryClassification,
    problem_products: list[object],
    provisional_resolutions: list[dict[str, object]] | None = None,
) -> str:
    """
    Формирует prompt только для поиска
    пропущенных functional_family.

    Эта стадия не должна изменять keywords
    уже существующих family.
    """

    existing_families = [
        {
            "name": rule.name,
            "keywords": list(rule.keywords),
        }
        for rule in classification.functional_families
    ]

    products = []

    for value in problem_products:
        normalized = normalize_niche_text(value)

        if not normalized:
            continue

        if normalized in products:
            continue

        products.append(normalized)

    provisional = []

    for item in provisional_resolutions or []:
        product_name = normalize_niche_text(
            item.get("product_name")
        )

        family_name = str(
            item.get("family_name", "")
        ).strip()

        confidence = item.get("confidence")

        if not product_name:
            continue

        if not family_name:
            continue

        provisional.append(
            {
                "product_name": product_name,
                "family_name": family_name,
                "confidence": confidence,
            }
        )

    provisional_groups_by_name = {}

    for item in provisional:
        family_name = item["family_name"]

        if family_name not in provisional_groups_by_name:
            provisional_groups_by_name[family_name] = []

        provisional_groups_by_name[family_name].append(
            item["product_name"]
        )

    provisional_groups = [
        {
            "family_name": family_name,
            "products": product_names,
        }
        for family_name, product_names
        in provisional_groups_by_name.items()
    ]

    repeated_term_candidates = (
        collect_repeated_term_candidates(
            products,
            min_support=2,
        )
    )

    return f"""
MISSING FAMILY DISCOVERY ONLY

Ты анализируешь товарную категорию маркетплейса.

CATEGORY:
{normalize_niche_text(category_name)}

EXISTING FUNCTIONAL FAMILIES:
{json.dumps(
    existing_families,
    ensure_ascii=False,
    indent=2,
)}

PROBLEM PRODUCTS:
{json.dumps(
    products,
    ensure_ascii=False,
    indent=2,
)}

PROVISIONAL FAMILY GROUPS:
{json.dumps(
    provisional_groups,
    ensure_ascii=False,
    indent=2,
)}

REPEATED TERM CANDIDATES:
{json.dumps(
    repeated_term_candidates,
    ensure_ascii=False,
    indent=2,
)}

используй повторяемость только как сигнал.
Повторяющийся термин сам по себе не является
доказательством существования новой family.

Проверь функциональный смысл каждого кандидата
и его соответствие CURRENT FUNCTIONAL FAMILIES.

PROVISIONAL RESOLUTIONS:
{json.dumps(
    provisional,
    ensure_ascii=False,
    indent=2,
)}

ЗАДАЧА:

Для каждой PROVISIONAL FAMILY GROUP
ищи повторяющийся функциональный подтип,
который по своему реальному назначению
не соответствует существующей family.

Не считай provisional family правильной
только потому, что несколько товаров были
предварительно назначены в неё.

Найди только пропущенные functional_family.

DO NOT ENRICH EXISTING FAMILY KEYWORDS

Не изменяй keywords существующих family.
Не удаляй существующие family.
Не переименовывай существующие family.

PROVISIONAL RESOLUTIONS содержат предварительное назначение
товара к существующей family.

Предварительное назначение может быть ошибочным,
даже если confidence высокий.

Проверь, образуют ли несколько problem products
устойчивый отдельный функциональный тип товара,
который плохо соответствует существующим
functional_family.

Новая functional_family оправдана, если:

- несколько товаров представляют один реальный тип;
- у них есть общее функциональное назначение;
- этот тип отличается от существующих family;
- отнесение к существующей family требует
  слишком широкого толкования её назначения.

Не создавай новую family только из-за:

- бренда;
- модели;
- цвета;
- размера;
- мощности;
- комплектации;
- типа питания.

Название новой functional_family должно быть:

- английским;
- lowercase;
- snake_case;
- коротким;
- функционально осмысленным.

Если отдельного нового типа нет,
не создавай новую family.

Не анализируй продажи, спрос,
конкуренцию или прибыльность.
""".strip()

def generate_missing_family_discovery(
    category_name: object,
    classification: CategoryClassification,
    problem_products: list[object],
    provisional_resolutions: list[dict[str, object]] | None,
    client,
    model: str = "gpt-5-nano",
) -> tuple[FunctionalFamilyRule, ...]:
    """
    Ищет только пропущенные functional_family.

    Не изменяет существующие family
    и их keywords.
    """

    prompt = build_missing_family_discovery_prompt(
        category_name=category_name,
        classification=classification,
        problem_products=problem_products,
        provisional_resolutions=provisional_resolutions,
    )

    discovery_schema = {
        "type": "object",
        "properties": {
            "new_families": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": [
                        "name",
                        "keywords",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "new_families",
        ],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "missing_family_discovery",
                "schema": discovery_schema,
                "strict": True,
            }
        },
    )

    data = json.loads(
        response.output_text
    )

    existing_names = {
        rule.name
        for rule in classification.functional_families
    }

    discovered_by_name = {}
    discovered_order = []

    for family in data["new_families"]:
        name = str(
            family["name"]
        ).strip()

        if not name:
            continue

        # Discovery имеет право вернуть
        # только действительно новые family.
        if name in existing_names:
            continue

        keywords = []

        for value in family["keywords"]:
            keyword = str(value).strip()

            if not keyword:
                continue

            if keyword in keywords:
                continue

            keywords.append(keyword)

        if not keywords:
            continue

        normalized_keywords = [
            normalize_niche_text(keyword)
            for keyword in keywords
            if normalize_niche_text(keyword)
        ]

        supporting_products = set()

        for product in problem_products:
            normalized_product = normalize_niche_text(
                product
            )

            if not normalized_product:
                continue

            if any(
                keyword in normalized_product
                for keyword in normalized_keywords
            ):
                supporting_products.add(
                    normalized_product
                )

        # Новая functional_family должна быть
        # подтверждена минимум двумя разными
        # проблемными товарами.
        #
        # Это отсекает случайные singleton-family,
        # предложенные AI по одному SKU.
        if len(supporting_products) < 2:
            continue

        if name not in discovered_by_name:
            discovered_order.append(name)
            discovered_by_name[name] = keywords
            continue

        # Если AI продублировал одну новую family,
        # объединяем keywords детерминированно.
        for keyword in keywords:
            if (
                keyword
                not in discovered_by_name[name]
            ):
                discovered_by_name[
                    name
                ].append(keyword)

    return tuple(
        FunctionalFamilyRule(
            name=name,
            keywords=tuple(
                discovered_by_name[name]
            ),
        )
        for name in discovered_order
    )

def build_category_repair_prompt(
    category_name: object,
    classification: CategoryClassification,
    unresolved_products: list[object],
    resolution_evidence: list[dict[str, object]] | None = None,
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

    evidence = []

    for item in resolution_evidence or []:
        product_name = normalize_niche_text(
            item.get("product_name")
        )

        family_name = str(
            item.get("family_name", "")
        ).strip()

        confidence = item.get(
            "confidence"
        )

        if (
            not product_name
            or not family_name
        ):
            continue

        evidence.append(
            {
                "product_name": product_name,
                "family_name": family_name,
                "confidence": confidence,
            }
        )

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

PROVISIONAL RESOLUTION EVIDENCE:
{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2,
)}

PROVISIONAL RESOLUTION EVIDENCE — это предварительные
гипотезы связи товар → functional_family.

Предыдущее решение family_name может быть ошибочным,
даже если confidence высокий.

Не считай эти назначения окончательно подтверждёнными.

Используй их как дополнительный сигнал, но заново
проверь реальный функциональный тип каждого товара
относительно CURRENT FUNCTIONAL FAMILIES.

Если несколько товаров образуют устойчивый
повторяющийся функциональный кластер, который
плохо соответствует существующим family,
рассмотри отдельное functional_family даже тогда,
когда PROVISIONAL RESOLUTION EVIDENCE ранее
отнесла эти товары к существующей family.

Если несколько корректно отнесённых примеров
показывают устойчивый отличительный термин для
уже существующей family, можно добавить этот
термин в keywords этой family.

Не добавляй:
- бренды;
- модели;
- размеры;
- мощность;
- случайные общие слова.

ЗАДАЧА:

У тебя две независимые задачи.

MISSING FAMILY DISCOVERY HAS PRIORITY

Перед расширением keywords существующих family
сначала проверь, не образуют ли проблемные товары
отдельный повторяющийся функциональный тип.

Если несколько товаров:

- имеют общий устойчивый термин или назначение;
- представляют один реальный тип товара;
- этот тип функционально отличается от уже
  существующих CURRENT FUNCTIONAL FAMILIES;
- для отнесения к существующей family требуется
  слишком широкое толкование её назначения;

сначала рассмотри создание новой functional_family.

Даже если PROVISIONAL RESOLUTION EVIDENCE
отнесла такие товары к существующей family,
это не является доказательством, что отдельной
family не существует.

Если устойчивый термин обозначает отдельный
тип товара, не добавляй термин как keyword существующей family.

Сначала рассмотри создание новой functional_family.

Только если новый термин является настоящим
синонимом или альтернативным названием уже
существующего функционального типа, используй
его для расширения keywords существующей family.

Не создавай новую family из-за различий только
в бренде, модели, цвете, размере, мощности,
комплектации или типе питания.

A. ENRICH EXISTING FAMILY KEYWORDS

Используй PROVISIONAL RESOLUTION EVIDENCE как
дополнительный источник примеров для улучшения
keywords уже существующих functional_family,
но только после проверки, что назначенная family
действительно соответствует типу товара.

Если товар был уверенно отнесён к существующей
family, проверь, содержит ли его название
устойчивый функциональный термин, которого ещё
нет в keywords этой family.

Такой термин можно добавить, если он:

- описывает тип или назначение товара;
- помогает отличать эту family от других;
- применим не только к одной конкретной карточке;
- не является брендом или моделью;
- не является цветом, размером или мощностью;
- не является случайным общим словом.

Не добавляй весь product_name как keyword.

Не добавляй слишком общие слова вроде:
"насос", "товар", "комплект", "оборудование",
если они не различают functional_family.

B. FIND MISSING FUNCTIONAL FAMILIES

Проверь UNRESOLVED PRODUCTS и определи,
не пропущено ли отдельное functional_family.

Важно:

1. Не удаляй существующие functional_family.

2. Не переименовывай существующие
functional_family.

3. Для существующих family разрешено добавлять
новые полезные keywords на основании
RESOLUTION EVIDENCE.

4. Добавляй новое семейство только если
unresolved товар действительно представляет
отдельный конкурентный тип товара.

5. Не создавай новое семейство только из-за:
- бренда;
- цвета;
- размера;
- мощности;
- комплектации;
- battery / petrol / electric.

6. Основной товар и аксессуар могут быть
разными functional_family.

7. Даже если предварительный resolution уже
назначил товар существующей family, проверь,
не образует ли группа похожих товаров отдельное
functional_family.

8. Если нового семейства не требуется,
проверь, можно ли безопасно улучшить keywords
существующих family по
PROVISIONAL RESOLUTION EVIDENCE.

9. Верни ПОЛНУЮ обновлённую классификацию,
включая все существующие семейства.

10. Названия новых functional_family:
- английский;
- lowercase;
- snake_case;
- короткие и стабильные.

11. confidence — число от 0 до 1.

Не анализируй спрос, продажи,
конкуренцию или прибыльность.
""".strip()

def generate_category_repair(
    category_name: object,
    classification: CategoryClassification,
    unresolved_products: list[object],
    client,
    model: str = "gpt-5-nano",
    resolution_evidence: list[dict[str, object]] | None = None,
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
        resolution_evidence=resolution_evidence,
    )

    repair_schema = json.loads(
        json.dumps(
            CATEGORY_CLASSIFICATION_JSON_SCHEMA
        )
    )

    repair_schema[
        "properties"
    ][
        "category_type"
    ]["enum"] = [
        classification.category_type
    ]

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "category_repair",
                "schema": repair_schema,
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

    repaired_by_name = {
        rule.name: rule
        for rule in repaired_rules
    }

    evidence_products_by_family = {}

    for item in resolution_evidence or []:
        if not isinstance(item, dict):
            continue

        family_name = str(
            item.get("family_name", "")
        ).strip()

        product_name = normalize_niche_text(
            item.get("product_name")
        )

        if (
            not family_name
            or not product_name
        ):
            continue

        evidence_products_by_family.setdefault(
            family_name,
            [],
        ).append(product_name)

    merged_rules = []

    for rule in classification.functional_families:
        repaired_rule = repaired_by_name.get(
            rule.name
        )

        if repaired_rule is None:
            merged_rules.append(rule)
            continue

        existing_keywords = list(
            rule.keywords
        )

        existing_normalized = {
            normalize_niche_text(keyword)
            for keyword in rule.keywords
        }

        evidence_products = (
            evidence_products_by_family.get(
                rule.name,
                [],
            )
        )

        accepted_new_keywords = []

        for keyword in repaired_rule.keywords:
            normalized_keyword = normalize_niche_text(
                keyword
            )

            if not normalized_keyword:
                continue

            # Старые keywords всегда сохраняем.
            if normalized_keyword in existing_normalized:
                continue

            # Новый keyword для существующей family
            # разрешён только при прямом подтверждении
            # хотя бы одним product_name этой family.
            if not any(
                normalized_keyword in product_name
                for product_name in evidence_products
            ):
                continue

            accepted_new_keywords.append(
                keyword
            )

        merged_keywords = tuple(
            dict.fromkeys(
                (
                    *existing_keywords,
                    *accepted_new_keywords,
                )
            )
        )

        merged_rules.append(
            FunctionalFamilyRule(
                name=rule.name,
                keywords=merged_keywords,
            )
        )
        
    existing_names = {
        rule.name
        for rule in classification.functional_families
    }

    merged_names = set(existing_names)

    for rule in repaired_rules:
        if rule.name in merged_names:
            continue

        merged_rules.append(rule)
        merged_names.add(rule.name)

    merged_rules = tuple(merged_rules)

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
        functional_families=merged_rules,
        confidence=confidence,
    )
