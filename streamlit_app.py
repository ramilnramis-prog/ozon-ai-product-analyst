from pathlib import Path
import tempfile

import streamlit as st

from app.ai_client import get_openai_client
from app.multi_period import combine_period_files
from app.reporting import build_leaf_category_top


st.set_page_config(
    page_title="Ozon AI Product Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("Ozon AI Product Analyst")
st.caption(
    "Загрузите XLSX или CSV — система автоматически "
    "проанализирует товары, ниши и категории."
)

uploaded_files = st.file_uploader(
    "Загрузите один или несколько файлов",
    type=["xlsx", "csv"],
    accept_multiple_files=True,
)

if uploaded_files and st.button(
    "Запустить анализ",
    type="primary",
):
    try:
        with st.spinner("Анализируем данные..."):
            client = get_openai_client()

            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []

                for uploaded_file in uploaded_files:
                    file_path = (
                        Path(temp_dir)
                        / uploaded_file.name
                    )

                    file_path.write_bytes(
                        uploaded_file.getbuffer()
                    )

                    file_paths.append(file_path)

                result = combine_period_files(
                    file_paths=file_paths,
                    classification_client=client,
                    classification_limit=3,
                    enrich_cached=False,
                )

        if not result["success"]:
            st.error(
                "Не удалось обработать загруженные файлы."
            )

            if result.get("files_failed"):
                st.write(result["files_failed"])

            st.stop()

        candidates = result["candidates"]
        report = result["report"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Файлов обработано",
            result["files_processed"],
        )

        col2.metric(
            "Кандидатов",
            len(candidates),
        )

        eligible_count = (
            int(candidates["is_eligible"].fillna(False).sum())
            if "is_eligible" in candidates.columns
            else 0
        )

        col3.metric(
            "Eligible",
            eligible_count,
        )

        st.subheader("Общий TOP")

        global_top = report.head(20)

        st.dataframe(
            global_top,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "root_category" in candidates.columns
            and "leaf_category" in candidates.columns
            and candidates["leaf_category"].notna().any()
        ):
            st.subheader("TOP по категориям")

            category_top = build_leaf_category_top(
                candidates,
                top_n=10,
            )

            root_categories = (
                category_top["root_category"]
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            for root_category in root_categories:
                st.markdown(
                    f"### {root_category}"
                )

                root_data = category_top[
                    category_top["root_category"]
                    == root_category
                ]

                leaf_categories = (
                    root_data["leaf_category"]
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )

                for leaf_category in leaf_categories:
                    with st.expander(
                        str(leaf_category),
                        expanded=False,
                    ):
                        leaf_data = root_data[
                            root_data["leaf_category"]
                            == leaf_category
                        ].copy()

                        if "is_eligible" in leaf_data.columns:
                            top_data = leaf_data[
                                leaf_data["is_eligible"]
                                .fillna(False)
                                .astype(bool)
                            ].copy()
                        elif "eligibility_status" in leaf_data.columns:
                            top_data = leaf_data[
                                leaf_data["eligibility_status"]
                                == "eligible"
                            ].copy()
                        else:
                            top_data = leaf_data.copy()

                        if top_data.empty:
                            st.info(
                                "Подходящих "
                                "кандидатов "
                                "для TOP нет."
                            )
                            continue

                        visible_columns = [
                            column
                            for column in [
                                "leaf_category_rank",
                                "product_name",
                                "opportunity_score",
                                "eligibility_status",
                                "latest_price",
                                "latest_sales_per_day",
                                "active_seller_count",
                                "strong_seller_count",
                                "high_competition_warning",
                                "functional_family",
                                "niche_key",
                            ]
                            if column in top_data.columns
                        ]

                        st.dataframe(
                            top_data[
                                visible_columns
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

        if result.get("files_failed"):
            with st.expander(
                "Файлы с ошибками"
            ):
                st.write(
                    result["files_failed"]
                )

    except Exception as exc:
        st.exception(exc)
