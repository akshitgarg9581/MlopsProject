from pathlib import Path
import numpy as np
import pandas as pd
import yaml


SHOP_TYPE_MAP = {
    "hospital_store": "hospital_store",
    "hospital store": "hospital_store",
    "hos.": "hospital_store",
    "independent": "independent",
    "ind.": "independent",
    "chain": "chain",
    "cha.": "chain",
}


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_shop_type(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map(lambda v: SHOP_TYPE_MAP.get(v, v.replace(" ", "_")))


def clean_dataframe(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    if params is None:
        params = load_params()

    clean_cfg = params["cleaning"]
    out = df.copy()

    out = out.drop_duplicates()

    leaking_cols = clean_cfg.get("leaking_columns", [])
    cols_to_drop = [c for c in leaking_cols if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    sentinel_col = clean_cfg.get("sentinel_col")
    sentinel_val = clean_cfg.get("sentinel_value")
    if sentinel_col in out.columns:
        out[sentinel_col] = out[sentinel_col].replace(sentinel_val, np.nan)
        median_val = out[sentinel_col].median()
        if pd.notna(median_val):
            out[sentinel_col] = out[sentinel_col].fillna(median_val)

    shop_col = clean_cfg.get("shop_type_col")
    if shop_col in out.columns:
        out[shop_col] = normalize_shop_type(out[shop_col])

    return out.reset_index(drop=True)


def load_and_clean_train(params_path: str = "params.yaml") -> pd.DataFrame:
    params = load_params(params_path)
    train_path = Path(params["data"]["train_path"])
    raw = pd.read_csv(train_path)
    return clean_dataframe(raw, params)


if __name__ == "__main__":
    cleaned = load_and_clean_train()
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Columns: {list(cleaned.columns)}")
    print(f"stock_lines min: {cleaned['stock_lines'].min()}")
    print(f"shop_type counts:\n{cleaned['shop_type'].value_counts()}")
