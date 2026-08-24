import json

from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from app.niche_grouping import normalize_niche_text


CategoryType = Literal[
    "homogeneous",
    "mixed",
    "unknown",
]

@dataclass(frozen=True)
class FunctionalFamilyRule:
    """
    Одно функциональное семейство товара
    и признаки для его определения.
    """

    name: str
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryClassification:
    """
    Результат анализа одной товарной категории.
    """

    category_name: str
    category_type: CategoryType
    functional_families: tuple[
        FunctionalFamilyRule,
        ...
    ] = ()
    confidence: float = 0.0

CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


def normalize_brand_match_text(
    value: object,
) -> str:
    """
    Нормализует текст для сравнения брендов,
    включая кириллица → латиница.
    """

    normalized = normalize_niche_text(value)

    return normalized.translate(
        CYRILLIC_TO_LATIN
    )

def category_cache_key(
    category_name: object,
    root_category: object = None,
) -> str:
    """
    Возвращает стабильный ключ категории
    для поиска в сохранённых классификациях.
    """

    normalized_category = normalize_niche_text(
        category_name
    )

    normalized_root = normalize_niche_text(
        root_category
    )

    if normalized_root and normalized_category:
        return (
            normalized_root
            + ":"
            + normalized_category
        )

    return normalized_category


def category_needs_ai(
    category_name: object,
    known_category_keys: set[str],
) -> bool:
    """
    Определяет, нужно ли отправлять категорию
    на AI-классификацию.

    Пустые категории здесь не отправляются в AI:
    они обрабатываются отдельным fallback-механизмом.
    """

    key = category_cache_key(category_name)

    if not key:
        return False

    return key not in known_category_keys

DEFAULT_CATEGORY_CACHE_PATH = Path(
    "config/category_classifications.json"
)


def load_category_cache(
    path: Path = DEFAULT_CATEGORY_CACHE_PATH,
) -> dict[str, dict]:
    """
    Загружает сохранённые классификации категорий.

    Если файл отсутствует или пустой,
    возвращает пустой словарь.
    """

    if not path.exists():
        return {}

    text = path.read_text(
        encoding="utf-8",
    ).strip()

    if not text:
        return {}

    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(
            "Category cache must contain a JSON object"
        )

    return data

def save_category_classification(
    classification: CategoryClassification,
    path: Path = DEFAULT_CATEGORY_CACHE_PATH,
    root_category: object = None,
) -> None:
    """
    Сохраняет одну классификацию категории в JSON-кэш.
    """

    cache = load_category_cache(path)

    key = category_cache_key(
        classification.category_name,
        root_category=root_category,
    )

    if not key:
        raise ValueError(
            "Category name cannot be empty"
        )

    cache[key] = {
    "category_type": classification.category_type,
    "functional_families": [
        {
            "name": rule.name,
            "keywords": list(rule.keywords),
        }
        for rule in classification.functional_families
    ],
    "confidence": classification.confidence,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

def get_cached_category_classification(
    category_name: object,
    path: Path = DEFAULT_CATEGORY_CACHE_PATH,
    root_category: object = None,
) -> CategoryClassification | None:
    """
    Возвращает сохранённую классификацию категории.

    Если категория ещё неизвестна системе,
    возвращает None.
    """

    key = category_cache_key(
        category_name,
        root_category=root_category,
)

    if not key:
        return None

    cache = load_category_cache(path)

    data = cache.get(key)

    if data is None:
        return None

    return CategoryClassification(
        category_name=str(category_name).strip(),
        category_type=data["category_type"],
        functional_families=tuple(
            FunctionalFamilyRule(
                name=rule["name"],
                keywords=tuple(
                   rule.get(
                       "keywords",
                      [],
                )
            ),
        )
    for rule in data.get(
        "functional_families",
        []
    )
        ),
        confidence=float(
            data.get(
                "confidence",
                0.0,
            )
        ),
    )
def match_functional_families(
    product_name: object,
    rules: tuple[FunctionalFamilyRule, ...],
) -> tuple[str, ...]:
    """
    Возвращает все functional_family,
    ключевые слова которых найдены
    в названии товара.

    Пустой tuple означает, что семейство
    определить не удалось.

    Несколько значений означают
    неоднозначную классификацию.
    """

    normalized_name = normalize_niche_text(
        product_name
    )

    if not normalized_name:
        return ()

    matches = []

    for rule in rules:
        for keyword in rule.keywords:
            normalized_keyword = normalize_niche_text(
                keyword
            )

            if (
                normalized_keyword
                and normalized_keyword
                in normalized_name
            ):
                matches.append(rule.name)
                break

    return tuple(matches)

def get_family_match_status(
    matches: tuple[str, ...],
) -> str:
    """
    Возвращает статус определения
    functional_family.
    """

    if len(matches) == 1:
        return "matched"

    if len(matches) > 1:
        return "ambiguous"

    return "unmatched"

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

def match_product_to_category_classification(
    product_name: object,
    classification: CategoryClassification,
) -> tuple[tuple[str, ...], str]:
    """
    Применяет классификацию категории
    к одному товару.

    Возвращает:
    - найденные functional_family;
    - статус matched / ambiguous / unmatched.
    """

    rules = classification.functional_families

    if classification.category_type == "unknown":
        return (), "unmatched"

    if classification.category_type == "homogeneous":
        if len(rules) == 1:
            return (
                (rules[0].name,),
                "matched",
            )

        return (), "unmatched"

    matches = match_functional_families(
        product_name,
        rules,
    )

    status = get_family_match_status(
        matches
    )

    return matches, status

def apply_category_classification(
    dataframe: pd.DataFrame,
    classification: CategoryClassification,
) -> pd.DataFrame:
    """
    Применяет готовую классификацию категории
    ко всем товарам DataFrame.

    AI здесь не вызывается.
    """

    result = dataframe.copy()

    result["functional_family"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result["functional_family_status"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    if "product_name" not in result.columns:
        return result

    for index, product_name in result[
        "product_name"
    ].items():
        matches, status = (
            match_product_to_category_classification(
                product_name,
                classification,
            )
        )

        result.at[
            index,
            "functional_family_status",
        ] = status

        if status == "matched":
            result.at[
                index,
                "functional_family",
            ] = matches[0]

    return result

def apply_family_resolutions(
    dataframe: pd.DataFrame,
    resolutions: list[dict[str, object]],
) -> pd.DataFrame:
    """
    Применяет готовые AI-resolutions
    к ambiguous/unmatched товарам.

    AI здесь не вызывается.
    """

    result = dataframe.copy()

    resolution_map = {
        normalize_niche_text(
            item["product_name"]
        ): item
        for item in resolutions
    }

    for index, row in result.iterrows():
        if (
            row.get("functional_family_status")
            not in {"ambiguous", "unmatched"}
        ):
            continue

        normalized_name = normalize_niche_text(
            row.get("product_name")
        )

        resolution = resolution_map.get(
            normalized_name
        )

        if resolution is None:
            continue

        family_name = resolution["family_name"]

        if family_name == "unresolved":
            result.at[
                index,
                "functional_family_status",
            ] = "unresolved"
            continue

        result.at[
            index,
            "functional_family",
        ] = family_name

        result.at[
            index,
            "functional_family_status",
        ] = "ai_resolved"

    return result

def remove_brand_keywords(
    classification: CategoryClassification,
    brand_values: list[object],
) -> CategoryClassification:
    """
    Удаляет из AI-generated keywords значения,
    которые совпадают с известными брендами.

    Названия functional_family не изменяет.
    """

    normalized_brands = {
        normalize_brand_match_text(value)
        for value in brand_values
        if normalize_brand_match_text(value)
    }

    cleaned_rules = []

    for rule in classification.functional_families:
        keywords = tuple(
            keyword
            for keyword in rule.keywords
            if normalize_brand_match_text(keyword)
            not in normalized_brands
        )

        cleaned_rules.append(
            FunctionalFamilyRule(
                name=rule.name,
                keywords=keywords,
            )
        )

    return CategoryClassification(
        category_name=classification.category_name,
        category_type=classification.category_type,
        functional_families=tuple(cleaned_rules),
        confidence=classification.confidence,
    )