import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pharmacy.clean import clean_dataframe, load_params


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def build_estimator(name: str, cfg: dict):
    if name == "ridge":
        return Ridge(**cfg)
    if name == "random_forest":
        return RandomForestRegressor(**cfg)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(**cfg)
    raise ValueError(f"Unknown model: {name}")


def log_sklearn_model(pipeline: Pipeline, artifact_path: str = "model") -> None:
    try:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=artifact_path,
            serialization_format="cloudpickle",
        )
    except TypeError:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
        )


def train_v1(params_path: str = "params.yaml") -> dict:
    params = load_params(params_path)
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]
    train_cfg = params["training"]

    raw_train_path = data_cfg["train_path"]
    data_hash = file_sha256(raw_train_path)

    raw_df = pd.read_csv(raw_train_path)
    df = clean_dataframe(raw_df, params)

    num_cols = data_cfg["numeric_features"]
    cat_cols = data_cfg["categorical_features"]
    feature_cols = num_cols + cat_cols
    target_col = data_cfg["target_col"]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    best_mae = float("inf")
    best_run_id = None
    best_model_name = None
    best_pipeline = None

    for model_name, model_params in train_cfg["models"].items():
        with mlflow.start_run(run_name=f"v1_{model_name}") as run:
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("model_version_tag", "v1")
            for k, v in model_params.items():
                mlflow.log_param(k, v)

            eval_pipe = Pipeline(
                steps=[
                    ("prep", build_preprocessor(num_cols, cat_cols)),
                    ("model", build_estimator(model_name, model_params)),
                ]
            )
            eval_pipe.fit(X_train, y_train)
            preds = eval_pipe.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, preds))
            mlflow.log_metric("val_mae", val_mae)

            full_pipe = Pipeline(
                steps=[
                    ("prep", build_preprocessor(num_cols, cat_cols)),
                    ("model", build_estimator(model_name, model_params)),
                ]
            )
            full_pipe.fit(X, y)
            log_sklearn_model(full_pipe, artifact_path="model")

            print(f"{model_name}: val_mae={val_mae:.4f} (run_id={run.info.run_id})")

            if val_mae < best_mae:
                best_mae = val_mae
                best_run_id = run.info.run_id
                best_model_name = model_name
                best_pipeline = full_pipe

    reg_name = mlflow_cfg["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best_run_id}/model", reg_name)
    client = MlflowClient(tracking_uri=mlflow_cfg["tracking_uri"])
    client.set_registered_model_alias(reg_name, "champion", mv.version)

    model_out = Path(train_cfg["model_output_path"])
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_out)

    result = {
        "experiment_name": mlflow_cfg["experiment_name"],
        "best_model": best_model_name,
        "best_val_mae": best_mae,
        "best_run_id": best_run_id,
        "registered_model_name": reg_name,
        "champion_version": int(mv.version),
        "data_sha256": data_hash,
    }
    print("\nBEST V1 MODEL:", result)
    return result


if __name__ == "__main__":
    train_v1()
