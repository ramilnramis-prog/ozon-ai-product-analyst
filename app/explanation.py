import json
import pandas as pd

from app.explanation_schema import EXPLANATION_JSON_SCHEMA


EXPLANATION_FIELDS = [
    "opportunity_rank",
    "product_name",
    "opportunity_score",
    "score_coverage",
    "eligibility_status",
    "demand_score",
    "growth_score",
    "stability_score",
    "competition_score",
    "concentration_score",
    "latest_price",
    "latest_sales_per_day",
    "latest_revenue_per_day",
    "growth_rate",
    "trend_direction",
    "active_seller_count",
    "strong_seller_count",
    "top_3_seller_share",
    "top_10_seller_share",
    "low_market_depth_warning",
    "high_competition_warning",
    "niche_key_confidence",
]


def build_explanation_context(
    row: pd.Series,
) -> dict[str, object]:
    """
    Формирует только подтвержденные аналитикой факты,
    которые разрешено передавать AI для объяснения.

    AI не получает исходные сырые данные
    и не рассчитывает Opportunity Score самостоятельно.
    """

    context = {}

    for field in EXPLANATION_FIELDS:
        if field not in row.index:
            continue

        value = row[field]

        if pd.isna(value):
            context[field] = None
            continue

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float):
            value = round(
                value,
                2,
            )

        context[field] = value

    return context
def build_explanation_prompt(
    context: dict[str, object],
) -> str:
    """
    Создает строгий prompt для AI-объяснения.

    Функция не вызывает API.
    Она только подготавливает текст,
    который позже можно передать модели.
    """

    facts_json = json.dumps(
        context,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Ты — аналитик товаров маркетплейса.

Объясни результат анализа простым деловым языком.

СТРОГИЕ ПРАВИЛА:
1. Используй только факты из блока ANALYTICS FACTS.
2. Не пересчитывай Opportunity Score.
3. Не изменяй рейтинг товара.
4. Не придумывай отсутствующие показатели.
5. Значение null означает, что данных нет.
6. Если score_coverage меньше 100%, явно укажи,
   что итог рассчитан по неполному набору данных.
7. Если competition_score отсутствует,
   не делай вывод о низкой или высокой конкуренции.
8. Если eligibility_status =
   insufficient_competition_data,
   укажи, что товар пока нельзя считать
   полностью подтвержденной рыночной возможностью.
9. Если niche_key_confidence = very_low,
   укажи, что границы товарной ниши
   требуют дополнительного подтверждения.
10. Не предлагай новые цифры от себя.
11. Все технические значения и статусы из ANALYTICS FACTS
    воспроизводи точно, без переименования и исправления.
    Например eligibility_status нельзя переводить,
    сокращать или заменять другим значением.
12. Не расширяй смысл метрик.
    stability_score означает стабильность динамики продаж
    товара, а не стабильность всего рынка.
13. В разделе "Что проверить дальше" упоминай только
    отсутствующие или недостаточно подтвержденные поля,
    которые присутствуют в ANALYTICS FACTS.
    Не добавляй новые направления анализа,
    которых нет среди переданных фактов.

Структура ответа:

Итог
Краткий вывод в 1–2 предложениях.

Сильные стороны
Только подтвержденные положительные факторы.

Риски и ограничения
Только подтвержденные риски,
предупреждения и отсутствующие данные.

Что проверить дальше
Каких данных не хватает для более уверенного решения.

ANALYTICS FACTS:
{facts_json}
""".strip()

def generate_explanation(
    context: dict[str, object],
    client,
    model: str = "gpt-5-nano",
) -> dict[str, object]:
    """
    Генерирует структурированное AI-объяснение
    на основе уже рассчитанных аналитических фактов.

    Score и другие метрики рассчитываются
    до вызова модели.
    """

    prompt = build_explanation_prompt(
        context
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "marketplace_explanation",
                "schema": EXPLANATION_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    return json.loads(
        response.output_text
    )