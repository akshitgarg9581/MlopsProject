from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.pharmacy.clean import clean_dataframe, load_params
from src.pharmacy.train import (
    build_estimator,
    build_preprocessor,
    file_sha256,
    log_sklearn_model,
)


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected.dropna().to_numpy(), quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected.dropna().to_numpy(), bins=edges)
    act_counts, _ = np.histogram(actual.dropna().to_numpy(), bins=edges)

    exp_pct = np.clip(exp_counts / max(len(expected.dropna()), 1), eps, None)
    act_pct = np.clip(act_counts / max(len(actual.dropna()), 1), eps, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def run_monitoring_and_retraining(params_path: str = "params.yaml") -> dict:
    params = load_params(params_path)
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]
    train_cfg = params["training"]
    mon_cfg = params["monitoring"]

    num_cols = data_cfg["numeric_features"]
    cat_cols = data_cfg["categorical_features"]
    feature_cols = num_cols + cat_cols
    target_col = data_cfg["target_col"]
    date_col = data_cfg["date_col"]
    group_col = data_cfg["group_col"]
    id_col = data_cfg["id_col"]

    raw_train_path = data_cfg["train_path"]
    data_hash = file_sha256(raw_train_path)

    train_clean = clean_dataframe(pd.read_csv(raw_train_path), params)
    later_df = clean_dataframe(pd.read_csv(data_cfg["later_path"]), params)
    holdout_df = clean_dataframe(pd.read_csv(data_cfg["holdout_path"]), params)

    v1_model = joblib.load(train_cfg["model_output_path"])

    psi_scores = {}
    for col in num_cols:
        psi_scores[col] = round(compute_psi(train_clean[col], later_df[col], bins=mon_cfg["psi_bins"]), 4)
    most_drifted = max(psi_scores, key=psi_scores.get)

    later_preds_v1 = v1_model.predict(later_df[feature_cols])
    mae_by_group = {}
    for group_val in ["Old Town", "New Town", "Cantonment"]:
        mask = later_df[group_col] == group_val
        mae_by_group[group_val] = round(
            float(mean_absolute_error(later_df.loc[mask, target_col], later_preds_v1[mask])),
            4,
        )
    worst_served = max(mae_by_group, key=mae_by_group.get)

    later_sorted = later_df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    split_idx = int(len(later_sorted) * mon_cfg["retrain_split_ratio"])
    later_train = later_sorted.iloc[:split_idx]
    later_check = later_sorted.iloc[split_idx:]

    retrain_df = pd.concat([train_clean, later_train], ignore_index=True)
    X_retrain = retrain_df[feature_cols]
    y_retrain = retrain_df[target_col]
    X_check = later_check[feature_cols]
    y_check = later_check[target_col]

    mae_v1_check = float(mean_absolute_error(y_check, v1_model.predict(X_check)))

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    best_v2_mae = float("inf")
    best_v2_run_id = None
    best_v2_model_name = None
    best_v2_pipeline = None

    for model_name, model_params in train_cfg["models"].items():
        with mlflow.start_run(run_name=f"v2_{model_name}") as run:
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("model_version_tag", "v2")
            for k, v in model_params.items():
                mlflow.log_param(k, v)

            pipe = Pipeline(
                steps=[
                    ("prep", build_preprocessor(num_cols, cat_cols)),
                    ("model", build_estimator(model_name, model_params)),
                ]
            )
            pipe.fit(X_retrain, y_retrain)
            check_mae = float(mean_absolute_error(y_check, pipe.predict(X_check)))
            mlflow.log_metric("mae_later_check", check_mae)
            log_sklearn_model(pipe, artifact_path="model")

            print(f"v2_{model_name}: mae_later_check={check_mae:.4f} (run_id={run.info.run_id})")

            if check_mae < best_v2_mae:
                best_v2_mae = check_mae
                best_v2_run_id = run.info.run_id
                best_v2_model_name = model_name
                best_v2_pipeline = pipe

    reg_name = mlflow_cfg["registered_model_name"]
    mv2 = mlflow.register_model(f"runs:/{best_v2_run_id}/model", reg_name)
    client = MlflowClient(tracking_uri=mlflow_cfg["tracking_uri"])

    improvement = (mae_v1_check - best_v2_mae) / mae_v1_check
    if improvement >= mon_cfg["promotion_min_improvement"]:
        decision = "promote"
        champion_version = int(mv2.version)
        client.set_registered_model_alias(reg_name, "champion", str(mv2.version))
        champion_model = best_v2_pipeline
        joblib.dump(champion_model, train_cfg["model_output_path"])
    else:
        decision = "keep"
        champion_version = 1
        champion_model = v1_model

    holdout_preds = champion_model.predict(holdout_df[feature_cols])
    pred_out = Path(mon_cfg["predictions_output_path"])
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "record_id": holdout_df[id_col],
            "prediction": np.round(holdout_preds, 4),
        }
    ).to_csv(pred_out, index=False)

    summary = {
        "psi": psi_scores,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
        "mae_v1_on_check": round(mae_v1_check, 4),
        "mae_v2_on_check": round(best_v2_mae, 4),
        "v2_best_model": best_v2_model_name,
        "improvement_pct": round(improvement * 100, 2),
        "decision": decision,
        "champion_version": champion_version,
        "holdout_rows": len(holdout_df),
    }
    print("\nSTAGE 4 SUMMARY:", summary)
    return summary


if __name__ == "__main__":
    run_monitoring_and_retraining()
