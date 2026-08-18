from app.column_inspector import inspect_columns


def test_valid_columns():
    result = inspect_columns(
        ["Товар", "Продажи", "Неизвестная колонка"]
    )

    assert result["mapped"] == {
        "Товар": "product_name",
        "Продажи": "sales_units",
    }

    assert result["unmapped"] == [
        "Неизвестная колонка"
    ]

    assert result["missing_required"] == []
    assert result["missing_metric_groups"] == []
    assert result["is_valid"] is True