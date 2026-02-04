import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Schema alignment
# -----------------------------
def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime
    for col in ["ETD", "ATD", "ETA", "ATA", "estimated_delivery", "actual_delivery"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Compute transit_time_days if missing
    if "transit_time_days" not in df.columns:
        if {"ATD", "ATA"}.issubset(df.columns):
            df["transit_time_days"] = (df["ATA"] - df["ATD"]).dt.days

    # Compute delay_days if missing
    if "delay_days" not in df.columns:
        if {"ETD", "ATD"}.issubset(df.columns):
            df["delay_days"] = (df["ATD"] - df["ETD"]).dt.days

    return df


# -----------------------------
# Derived features
# -----------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # lane feature
    if {"POL", "POD"}.issubset(df.columns):
        df["lane"] = df["POL"].astype(str) + "->" + df["POD"].astype(str)

    # temporal features: preferred ETD
    base = df["ETD"].where(df["ETD"].notna(), df["ATD"])
    df["ETD_month"] = base.dt.month
    df["ETD_weekday"] = base.dt.weekday

    # vessel presence indicator
    df["has_vessel"] = df["Vessel"].notna().astype(int)

    return df


# -----------------------------
# Feature selection
# -----------------------------
NUMERIC_FEATURES = [
    "ContainerCount",
    "TotalWeight",
    "TotalVolume",
    "LegOrder",
    "transit_time_days",
    "ETD_month",
    "ETD_weekday",
    "has_vessel",
]

CATEGORICAL_FEATURES = [
    "MasterCarrier",
    "CarrierCategory",
    "POL",
    "POD",
    "lane",
]


# -----------------------------
# Prepare training data
# -----------------------------
def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = align_schema(df)
    df = add_derived_features(df)

    # drop missing target rows
    df = df.dropna(subset=["delay_days"])

    # keep only columns that exist
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X = df[num_cols + cat_cols]
    y = df["delay_days"].astype(float)

    return X, y, num_cols, cat_cols


# -----------------------------
# Pipeline
# -----------------------------
def build_pipeline(num_cols, cat_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])


# -----------------------------
# Train + save
# -----------------------------
def train_and_save_model(
    data_path="data/processed_shipments_data.csv",
    model_path="artifacts/delay_model.pkl"
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df = pd.read_csv(data_path)

    X, y, num_cols, cat_cols = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"MAE: {mae:.2f} days")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_and_save_model()