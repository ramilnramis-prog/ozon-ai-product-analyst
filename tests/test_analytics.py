import pandas as pd

from app.analytics import add_daily_metrics, add_share_metrics


def test_add_daily_metrics():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
            ],
            "sales_units": [
                120.0,
                60.0,
            ],
            "revenue": [
                120000.0,
                60000.0,
            ],
            "period_days": [
                30.0,
                30.0,
            ],
        }
    )

    result = add_daily_metrics(dataframe)

    assert result["sales_per_day"].tolist() == [
        4.0,
        2.0,
    ]

    assert result["revenue_per_day"].tolist() == [
        4000.0,
        2000.0,
    ]
def test_invalid_period_days_do_not_create_infinity():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
            ],
            "sales_units": [
                100.0,
                100.0,
            ],
            "revenue": [
                50000.0,
                50000.0,
            ],
            "period_days": [
                0.0,
                -5.0,
            ],
        }
    )

    result = add_daily_metrics(dataframe)

    assert result["sales_per_day"].isna().all()
    assert result["revenue_per_day"].isna().all()

def test_add_share_metrics():
    dataframe = pd.DataFrame(
        {
            "product_name": [
                "Товар A",
                "Товар B",
                "Товар C",
            ],
            "sales_units": [
                50.0,
                25.0,
                25.0,
            ],
            "revenue": [
                50000.0,
                25000.0,
                25000.0,
            ],
        }
    )

    result = add_share_metrics(dataframe)

    assert result["sales_share"].tolist() == [
        0.5,
        0.25,
        0.25,
    ]

    assert result["revenue_share"].tolist() == [
        0.5,
        0.25,
        0.25,
    ]

    assert result["sales_share"].sum() == 1.0
    assert result["revenue_share"].sum() == 1.0