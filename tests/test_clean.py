import pandas as pd

from src.pharmacy.clean import clean_dataframe, load_params, normalize_shop_type


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "R1",
                "date": "2026-01-01",
                "flu_cases_reported": 100,
                "temp_c": 24.0,
                "nearby_clinics": 5,
                "discount_pct": 10.0,
                "stock_lines": -1,
                "distance_to_hospital_km": 3.5,
                "shop_type": "Chain ",
                "locality": "Old Town",
                "units_billed_in_gst_return": 350,
                "units_sold": 352,
            },
            {
                "record_id": "R1",
                "date": "2026-01-01",
                "flu_cases_reported": 100,
                "temp_c": 24.0,
                "nearby_clinics": 5,
                "discount_pct": 10.0,
                "stock_lines": -1,
                "distance_to_hospital_km": 3.5,
                "shop_type": "Chain ",
                "locality": "Old Town",
                "units_billed_in_gst_return": 350,
                "units_sold": 352,
            },
            {
                "record_id": "R2",
                "date": "2026-01-02",
                "flu_cases_reported": 120,
                "temp_c": 22.5,
                "nearby_clinics": 7,
                "discount_pct": 15.0,
                "stock_lines": 400,
                "distance_to_hospital_km": 2.0,
                "shop_type": "hos.",
                "locality": "New Town",
                "units_billed_in_gst_return": 410,
                "units_sold": 412,
            },
            {
                "record_id": "R3",
                "date": "2026-01-03",
                "flu_cases_reported": 80,
                "temp_c": 26.0,
                "nearby_clinics": 3,
                "discount_pct": 5.0,
                "stock_lines": 600,
                "distance_to_hospital_km": 5.0,
                "shop_type": " ind.",
                "locality": "Cantonment",
                "units_billed_in_gst_return": 290,
                "units_sold": 295,
            },
        ]
    )


def test_load_params() -> None:
    params = load_params("params.yaml")
    assert "data" in params
    assert "cleaning" in params
    assert "mlflow" in params
    assert "training" in params


def test_clean_drops_duplicates_and_leaking_column() -> None:
    cleaned = clean_dataframe(_sample_df())
    assert len(cleaned) == 3
    assert "units_billed_in_gst_return" not in cleaned.columns


def test_clean_replaces_sentinel_stock_lines() -> None:
    cleaned = clean_dataframe(_sample_df())
    assert (cleaned["stock_lines"] == -1).sum() == 0
    assert float(cleaned.loc[0, "stock_lines"]) == 500.0


def test_normalize_shop_type_variants() -> None:
    raw = pd.Series(
        ["Chain", "cha.", " chain", "ind.", "INDEPENDENT", "hospital store", "hos.", "HOSPITAL_STORE"]
    )
    cleaned = normalize_shop_type(raw)
    assert set(cleaned.unique()) == {"chain", "independent", "hospital_store"}
