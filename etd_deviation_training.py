"""
ETD Deviation Model Training
Separate module for training ETD deviation prediction model.
"""
import os
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Feature selection
# -----------------------------
NUMERIC_FEATURES = [
    "doc_completeness",           # Document completeness (0-7)
    "commodity_incomplete",       # HS code status: binary (1=incomplete, 0=complete)
    "port_congestion",            # Port congestion (0-20)
    "carrier_confirmation_conf",  # Carrier confirmation (0-1 continuous probability)
]

CATEGORICAL_FEATURES = [
    # No categorical features - using only numeric data quality metrics
]


# -----------------------------
# Schema alignment
# -----------------------------
def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime columns and compute ETD_deviation_days."""
    df = df.copy()

    # Ensure datetime
    for col in ["ETD", "ATD", "ETA", "ATA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Compute ETD_deviation_days if missing
    if "ETD_deviation_days" not in df.columns:
        if {"ETD", "ATD"}.issubset(df.columns):
            df["ETD_deviation_days"] = (df["ATD"] - df["ETD"]).dt.days

    return df


# -----------------------------
# Outlier removal
# -----------------------------
def remove_outliers(df: pd.DataFrame, target_col: str, max_abs_value: float = 100.0) -> pd.DataFrame:
    """
    Remove outliers based on absolute value threshold.
    
    Args:
        df: DataFrame with data
        target_col: Target column to check for outliers
        max_abs_value: Maximum absolute value allowed (default 100 days)
    
    Returns:
        DataFrame without outliers
    """
    df = df.copy()
    
    initial_count = len(df)
    df = df[df[target_col].abs() <= max_abs_value]
    removed_count = initial_count - len(df)
    
    print(f"Outlier removal for '{target_col}':")
    print(f"  Threshold: |value| <= {max_abs_value} days")
    print(f"  Removed: {removed_count} rows ({removed_count/initial_count*100:.1f}%)")
    print(f"  Remaining: {len(df)} rows")
    
    return df


# -----------------------------
# Prepare training data
# -----------------------------
def prepare_deviation_training_data(
    df: pd.DataFrame, 
    remove_outliers_flag: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Prepare data for ETD deviation model training."""
    df = align_schema(df)

    # Drop missing target rows
    df = df.dropna(subset=["ETD_deviation_days"])
    
    # Remove outliers if requested
    if remove_outliers_flag:
        df = remove_outliers(df, "ETD_deviation_days", max_abs_value=100.0)

    # Keep only columns that exist
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X = df[num_cols + cat_cols]
    y = df["ETD_deviation_days"].astype(float)

    return X, y, num_cols, cat_cols


# -----------------------------
# Build pipeline
# -----------------------------
def build_pipeline(num_cols, cat_cols):
    """Build ML pipeline for ETD deviation prediction."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
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
# Train + save deviation model
# -----------------------------
def train_and_save_deviation_model(
    data_path="data/shipments_to_predict_etd.csv",
    model_path="artifacts/etd_deviation_model.pkl"
):
    """Train and save ETD deviation prediction model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df = pd.read_csv(data_path)

    X, y, num_cols, cat_cols = prepare_deviation_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"ETD Deviation Model MAE: {mae:.2f} days")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Saved deviation model to {model_path}")


if __name__ == "__main__":
    print("Training ETD deviation model...")
    train_and_save_deviation_model()
