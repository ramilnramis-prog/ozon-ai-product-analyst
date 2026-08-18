from pathlib import Path

import pandas as pd

from app.period_parser import extract_period_from_filename
from app.opportunity import calculate_opportunity_metrics
from app.analytics import add_daily_metrics, add_share_metrics
from app.column_inspector import inspect_columns
from app.competition import calculate_competition_metrics
from app.data_cleaner import clean_numeric_columns
from app.data_loader import load_table
from app.data_quality import calculate_data_quality
from app.normalizer import normalize_dataframe


def run_analysis_pipeline(
    file_path: str | Path,
) -> dict[str, object]:
    """
    Полный базовый pipeline анализа файла.

    Этапы:
    1. загрузка;
    2. проверка колонок;
    3. нормализация;
    4. очистка;
    5. расчет дневных метрик;
    6. расчет долей;
    7. анализ конкуренции;
    8. оценка качества данных.
    """

    raw_dataframe = load_table(file_path)

    inspection = inspect_columns(
        list(raw_dataframe.columns)
    )

    if not inspection["is_valid"]:
        return {
            "success": False,
            "inspection": inspection,
            "dataframe": None,
            "competition": None,
            "data_quality": None,
        }

    normalized = normalize_dataframe(
        raw_dataframe
    )

    cleaned = clean_numeric_columns(
        normalized
    )

    period = extract_period_from_filename(
        file_path
    )

    if (
        "period_days" not in cleaned.columns
        and period["period_days"] is not None
    ):
        cleaned["period_days"] = period[
            "period_days"
        ]

    if (
        "period_start" not in cleaned.columns
        and period["period_start"] is not None
    ):
        cleaned["period_start"] = period[
            "period_start"
        ]

    if (
        "period_end" not in cleaned.columns
        and period["period_end"] is not None
    ):
        cleaned["period_end"] = period[
            "period_end"
        ]

    if period["period_source"] is not None:
        cleaned["period_source"] = period[
            "period_source"
        ]

    analyzed = add_daily_metrics(
        cleaned
    )

    analyzed = add_share_metrics(
        analyzed
    )

    competition = calculate_competition_metrics(
        analyzed
    )

    opportunity = calculate_opportunity_metrics(
        analyzed
)
    data_quality = calculate_data_quality(
        analyzed
    )

    return {
        "success": True,
        "inspection": inspection,
        "dataframe": analyzed,
        "competition": competition,
        "data_quality": data_quality,
        "opportunity": opportunity,
    }