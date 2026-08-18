from app.column_mapping import match_column_name
from app.schema import REQUIRED_FIELDS, REQUIRED_METRIC_GROUPS


def inspect_columns(columns: list[str]) -> dict[str, object]:
    """
    Проверяет набор колонок входного файла.

    Возвращает:
    - какие колонки распознаны;
    - какие не распознаны;
    - есть ли дубли по смыслу;
    - каких обязательных данных не хватает.
    """

    mapped: dict[str, str] = {}
    unmapped: list[str] = []
    canonical_sources: dict[str, list[str]] = {}

    for column in columns:
        source_column = str(column)
        canonical_field = match_column_name(source_column)

        if canonical_field is None:
            unmapped.append(source_column)
            continue

        mapped[source_column] = canonical_field

        canonical_sources.setdefault(
            canonical_field,
            [],
        ).append(source_column)

    detected_fields = set(mapped.values())

    missing_required = sorted(
        REQUIRED_FIELDS - detected_fields
    )

    missing_metric_groups = []

    for group in REQUIRED_METRIC_GROUPS:
        if not detected_fields.intersection(group):
            missing_metric_groups.append(sorted(group))

    duplicate_mappings = {
        field: sources
        for field, sources in canonical_sources.items()
        if len(sources) > 1
    }

    is_valid = (
        not missing_required
        and not missing_metric_groups
    )

    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "duplicate_mappings": duplicate_mappings,
        "missing_required": missing_required,
        "missing_metric_groups": missing_metric_groups,
        "is_valid": is_valid,
    }