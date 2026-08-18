import re
from datetime import date
from pathlib import Path


DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})\b"
)


def extract_period_from_filename(
    file_path: str | Path,
) -> dict[str, object]:
    """
    Пытается определить период отчета по имени файла.

    Пример:
    report 2026-06-01 2026-06-30.xlsx

    ->
    period_start = 2026-06-01
    period_end = 2026-06-30
    period_days = 30
    """

    filename = Path(file_path).name

    matches = DATE_PATTERN.findall(filename)

    if len(matches) < 2:
        return {
            "period_start": None,
            "period_end": None,
            "period_days": None,
            "period_source": None,
        }

    try:
        start = date.fromisoformat(matches[0])
        end = date.fromisoformat(matches[1])

    except ValueError:
        return {
            "period_start": None,
            "period_end": None,
            "period_days": None,
            "period_source": None,
        }

    if end < start:
        return {
            "period_start": None,
            "period_end": None,
            "period_days": None,
            "period_source": None,
        }

    period_days = (end - start).days + 1

    return {
        "period_start": start,
        "period_end": end,
        "period_days": period_days,
        "period_source": "filename",
    }