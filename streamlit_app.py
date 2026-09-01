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

        global_top = report.head(20).copy()

        global_top.insert(
            0,
            "\u041c\u0435\u0441\u0442\u043e",
            range(1, len(global_top) + 1),
        )

        global_columns = {
            "\u041c\u0435\u0441\u0442\u043e": "\u041c\u0435\u0441\u0442\u043e",
            "product_name": "\u0422\u043e\u0432\u0430\u0440",
            "root_category": "\u041a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f",
            "leaf_category": "\u041f\u043e\u0434\u043a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f",
            "opportunity_score": "\u041e\u0446\u0435\u043d\u043a\u0430",
            "latest_price": "\u0426\u0435\u043d\u0430, \u20bd",
            "latest_sales_per_day": "\u041f\u0440\u043e\u0434\u0430\u0436\u0438/\u0434\u0435\u043d\u044c",
            "active_seller_count": "\u0410\u043a\u0442\u0438\u0432\u043d\u044b\u0445 \u043f\u0440\u043e\u0434\u0430\u0432\u0446\u043e\u0432",
            "strong_seller_count": "\u0421\u0438\u043b\u044c\u043d\u044b\u0445 \u043f\u0440\u043e\u0434\u0430\u0432\u0446\u043e\u0432",
            "high_competition_warning": "\u0412\u044b\u0441\u043e\u043a\u0430\u044f \u043a\u043e\u043d\u043a\u0443\u0440\u0435\u043d\u0446\u0438\u044f",
        }

        visible_global = [
            column
            for column in global_columns
            if column in global_top.columns
        ]

        global_display = (
            global_top[visible_global]
            .rename(columns=global_columns)
        )

        competition_column = (
            "\u0412\u044b\u0441\u043e\u043a\u0430\u044f "
            "\u043a\u043e\u043d\u043a\u0443\u0440\u0435\u043d\u0446\u0438\u044f"
        )

        if competition_column in global_display.columns:
            global_display[competition_column] = (
                global_display[competition_column]
                .map({
                    True: "\u0414\u0430",
                    False: "\u041d\u0435\u0442",
                })
                .fillna("\u2014")
            )

        st.dataframe(
            global_display,
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

                        column_names = {
                            "leaf_category_rank": "\u041c\u0435\u0441\u0442\u043e",
                            "product_name": "\u0422\u043e\u0432\u0430\u0440",
                            "opportunity_score": "\u041e\u0446\u0435\u043d\u043a\u0430",
                            "latest_price": "\u0426\u0435\u043d\u0430, \u20bd",
                            "latest_sales_per_day": "\u041f\u0440\u043e\u0434\u0430\u0436\u0438/\u0434\u0435\u043d\u044c",
                            "active_seller_count": "\u0410\u043a\u0442\u0438\u0432\u043d\u044b\u0445 \u043f\u0440\u043e\u0434\u0430\u0432\u0446\u043e\u0432",
                            "strong_seller_count": "\u0421\u0438\u043b\u044c\u043d\u044b\u0445 \u043f\u0440\u043e\u0434\u0430\u0432\u0446\u043e\u0432",
                            "high_competition_warning": "\u0412\u044b\u0441\u043e\u043a\u0430\u044f \u043a\u043e\u043d\u043a\u0443\u0440\u0435\u043d\u0446\u0438\u044f",
                        }

                        visible_columns = [
                            column
                            for column in column_names
                            if column in top_data.columns
                        ]

                        display_data = (
                            top_data[visible_columns]
                            .rename(columns=column_names)
                            .copy()
                        )

                        competition_column = (
                            "\u0412\u044b\u0441\u043e\u043a\u0430\u044f "
                            "\u043a\u043e\u043d\u043a\u0443\u0440\u0435\u043d\u0446\u0438\u044f"
                        )

                        if competition_column in display_data.columns:
                            display_data[competition_column] = (
                                display_data[competition_column]
                                .map({
                                    True: "\u0414\u0430",
                                    False: "\u041d\u0435\u0442",
                                })
                                .fillna("\u2014")
                            )

                        st.dataframe(
                            display_data,
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
