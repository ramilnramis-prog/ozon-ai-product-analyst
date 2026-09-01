from pathlib import Path

import pandas as pd

from app.category_classifier import (
    get_cached_category_classification,
)

from app.trends import (
    calculate_product_stability,
    calculate_product_trends,
)

from app.category_classification_ai import (
    classify_category_dataframe_group,
)

from app.pipeline import run_analysis_pipeline
from app.candidate_features import build_candidate_features
from app.niche_grouping import add_niche_key
from app.product_attributes import add_product_attributes
from app.competition import calculate_niche_competition
from app.scoring import score_candidates
from app.ranking import rank_candidates
from app.reporting import build_candidate_report



def _normalize_product_name(value: object) -> str:
    """
    Нормализует название товара для запасного сопоставления.
    """

    if pd.isna(value):
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def _add_product_key(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Создаёт стабильный ключ товара между периодами.

    Приоритет:
    1. SKU
    2. название товара
    """

    result = dataframe.copy()

    result["product_key"] = ""
    result["product_key_source"] = ""

    if "sku" in result.columns:
        sku = (
            result["sku"]
            .astype("string")
            .str.strip()
        )

        valid_sku = (
            sku.notna()
            & (sku != "")
        )

        result.loc[
            valid_sku,
            "product_key",
        ] = "sku:" + sku[valid_sku]

        result.loc[
            valid_sku,
            "product_key_source",
        ] = "sku"

    missing_key = (
        result["product_key"] == ""
    )

    if "product_name" in result.columns:
        normalized_names = (
            result["product_name"]
            .map(_normalize_product_name)
        )

        valid_name = (
            missing_key
            & (normalized_names != "")
        )

        result.loc[
            valid_name,
            "product_key",
        ] = (
            "name:"
            + normalized_names[valid_name]
        )

        result.loc[
            valid_name,
            "product_key_source",
        ] = "product_name"

    return result


def combine_period_files(
    file_paths: list[str | Path],
    classification_client=None,
    classification_limit: int | None = None,
    enrich_cached: bool = False,
) -> dict[str, object]:
    """
    Запускает основной pipeline для каждого файла
    и объединяет успешные периоды в одну таблицу.
    """

    frames: list[pd.DataFrame] = []
    failed_files: list[dict[str, object]] = []

    for file_path in file_paths:
        result = run_analysis_pipeline(
            file_path
        )

        if not result["success"]:
            failed_files.append(
                {
                    "file": str(file_path),
                    "inspection": result[
                        "inspection"
                    ],
                }
            )
            continue

        dataframe = result["dataframe"].copy()

        dataframe["source_file"] = (
            Path(file_path).name
        )

        dataframe = _add_product_key(
            dataframe
        )

        frames.append(dataframe)

    if not frames:
        return {
            "success": False,
            "dataframe": None,
            "files_processed": 0,
            "files_failed": failed_files,
        }

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = add_product_attributes(
        combined
    )

    if classification_client is not None:
        category_pairs = (
            combined[
                [
                    "root_category",
                    "leaf_category",
                ]
            ]
            .dropna()
            .drop_duplicates()
        )

        new_categories_used = 0

        for _, category_row in (
            category_pairs.iterrows()
        ):
            root_category = category_row[
                "root_category"
            ]
            leaf_category = category_row[
                "leaf_category"
            ]

            if (
                not str(root_category).strip()
                or not str(leaf_category).strip()
            ):
                continue

            cached_classification = (
                get_cached_category_classification(
                    leaf_category,
                    root_category=root_category,
                )
            )

            if cached_classification is None:
                if (
                    classification_limit is not None
                    and new_categories_used
                    >= classification_limit
                ):
                    continue

                new_categories_used += 1

            try:
                classified = (
                    classify_category_dataframe_group(
                        dataframe=combined,
                        category_name=leaf_category,
                        root_category=root_category,
                        client=classification_client,
                        enrich_cached=enrich_cached,
                    )
                )
            except Exception as exc:
                if exc.__class__.__name__ == "APITimeoutError":
                    continue
                raise

            for column in (
                "functional_family",
                "functional_family_status",
                "category_type",
                "category_classification_confidence",
            ):
                if column not in classified.columns:
                    continue

                if column not in combined.columns:
                    combined[column] = pd.NA

                combined.loc[
                    classified.index,
                    column,
                ] = classified[column]

    combined = add_niche_key(
       combined
    )
    niche_competition = calculate_niche_competition(
       combined
    )

    trends = calculate_product_trends(
       combined
    )
    stability = calculate_product_stability(
       combined
    )
    candidates = build_candidate_features(
       combined,
       trends,
       stability,
    )
    if "niche_key" in combined.columns:
      latest_niche_keys = (
        combined
        .dropna(
            subset=[
                "product_key",
                "period_start",
            ]
        )
        .sort_values("period_start")
        .groupby(
            "product_key",
            as_index=False,
        )
        .tail(1)[
            [
                "product_key",
                "niche_key",
                "niche_key_source",
                "niche_key_confidence",
            ]
        ]
    )

      candidates = candidates.merge(
        latest_niche_keys,
        on="product_key",
        how="left",
    )
    if (
        not niche_competition.empty
        and "niche_key" in candidates.columns
        and "niche_key" in niche_competition.columns
    ):
        competition_columns = [
            column
            for column in [
                "niche_key",
                "seller_data_available",
                "seller_count",
                "active_seller_count",
                "strong_seller_count",
                "strong_seller_share",
                "top_3_seller_share",
                "top_10_seller_share",
                "low_market_depth_warning",
                "high_competition_warning",
            ]
            if column in niche_competition.columns
        ]

        candidates = candidates.merge(
            niche_competition[
                competition_columns
            ],
            on="niche_key",
            how="left",
        )

    candidates = score_candidates(
          candidates
)
    candidates = rank_candidates(
          candidates
)
    report = build_candidate_report(
          candidates
)
    return {
        "success": True,
        "dataframe": combined,
        "trends": trends,
        "stability": stability,
        "candidates": candidates,
        "report": report,
        "niche_competition": niche_competition,
        "files_processed": len(frames),
        "files_failed": failed_files,
    }