import pandas as pd


def check_leaking_columns(train: pd.DataFrame, holdout: pd.DataFrame, target: str = "units_sold") -> dict:
    extra_cols = [c for c in train.columns if c not in holdout.columns and c != target]
    corr = {}
    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            corr[col] = round(float(train[col].corr(train[target])), 6)
    return {"extra_columns": extra_cols, "correlations": corr}


def check_unit_mixup(train: pd.DataFrame, later: pd.DataFrame, group_col: str = "locality") -> dict:
    numeric_cols = [
        c for c in train.select_dtypes(include="number").columns
        if c != "units_sold" and c in later.columns
    ]
    summary = {}
    for col in numeric_cols:
        train_stats = train.groupby(group_col)[col].agg(["count", "min", "mean", "max"]).round(3).to_dict("index")
        later_stats = later.groupby(group_col)[col].agg(["min", "mean", "max"]).round(3).to_dict("index")
        summary[col] = {"train": train_stats, "later": later_stats}
    return summary


def check_sentinel_values(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    candidates = (-9999, -999, -99, -1, 999, 9999)
    numeric_cols = train.select_dtypes(include="number").columns
    hits = {}
    for col in numeric_cols:
        vc = train[col].value_counts()
        col_hits = {val: int(vc[val]) for val in candidates if val in vc.index}
        top5 = {float(k): int(v) for k, v in vc.head(5).items()}
        later_range = (
            (float(later[col].min()), float(later[col].max()))
            if col in later.columns
            else None
        )
        hits[col] = {
            "candidate_hits": col_hits,
            "top5": top5,
            "train_range": (float(train[col].min()), float(train[col].max())),
            "later_range": later_range,
        }
    return hits


def check_duplicates(train: pd.DataFrame) -> dict:
    exact_dups = int(train.duplicated().sum())
    without_id_dups = int(train.drop(columns=["record_id"]).duplicated().sum())
    return {"exact_duplicates": exact_dups, "without_record_id": without_id_dups}


def check_stuck_sensor(train: pd.DataFrame, min_run: int = 5) -> dict:
    df = train.sort_values(["date", "record_id"]).reset_index(drop=True)
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != "units_sold"
    ]
    runs = {}
    for col in numeric_cols:
        series = df[col]
        group_ids = (series != series.shift()).cumsum()
        grouped = df.groupby(group_ids).agg(
            value=(col, "first"),
            row_count=(col, "size"),
            start_date=("date", "first"),
            end_date=("date", "last"),
            n_days=("date", "nunique"),
        )
        long_runs = grouped[(grouped["row_count"] >= min_run) & (grouped["n_days"] >= 3)]
        if not long_runs.empty:
            runs[col] = long_runs.to_dict("records")
    return runs


def check_target_typos(train: pd.DataFrame, later: pd.DataFrame, target: str = "units_sold") -> dict:
    sorted_vals = train[target].sort_values(ascending=False).reset_index(drop=True)
    top_25 = [float(x) for x in sorted_vals.head(25).tolist()]
    later_max = float(later[target].max())
    high_rows = train[train[target] > later_max * 3][["record_id", "date", target]].sort_values(target, ascending=False)
    return {
        "train_median": float(train[target].median()),
        "later_max": later_max,
        "top_25_desc": top_25,
        "high_outlier_count": int(len(high_rows)),
        "high_outlier_rows": high_rows.to_dict("records"),
    }


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    text_cols = ["shop_type", "locality"]
    result = {}
    for col in text_cols:
        result[col] = {
            "train_counts": {repr(k): int(v) for k, v in train[col].value_counts(dropna=False).items()},
            "later_counts": {repr(k): int(v) for k, v in later[col].value_counts(dropna=False).items()},
            "later_true_levels": int(later[col].nunique()),
        }
    return result


def main() -> None:
    train = pd.read_csv("data/raw/train.csv")
    later = pd.read_csv("data/raw/later.csv")
    holdout = pd.read_csv("data/raw/holdout_features.csv")

    print(f"Rows: train={len(train)}, later={len(later)}, holdout={len(holdout)}")

    print("\n1. LEAKING COLUMN:")
    print(check_leaking_columns(train, holdout))

    print("\n2. UNIT MIXUP (by locality):")
    for col, stats in check_unit_mixup(train, later, "locality").items():
        print(f"  {col}:")
        print(f"    train: {stats['train']}")
        print(f"    later: {stats['later']}")

    print("\n3. SENTINEL VALUES:")
    for col, info in check_sentinel_values(train, later).items():
        print(f"  {col}: {info}")

    print("\n4. DUPLICATE ROWS:")
    print(check_duplicates(train))

    print("\n5. STUCK SENSOR:")
    print(check_stuck_sensor(train))

    print("\n6. TARGET TYPOS:")
    print(check_target_typos(train, later))

    print("\n7. SPELLING VARIANTS:")
    for col, counts in check_spelling_variants(train, later).items():
        print(f"  {col}: {counts}")


if __name__ == "__main__":
    main()
