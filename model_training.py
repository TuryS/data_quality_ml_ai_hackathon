import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, median_absolute_error


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
            # PATCH: use fractional days instead of .dt.days
            dt_days = (df["ATA"] - df["ATD"]).dt.total_seconds() / 86400.0
            df["transit_time_days"] = dt_days

    # Compute delay_days if missing
    if "delay_days" not in df.columns:
        if {"ETD", "ATD"}.issubset(df.columns):
            # PATCH: use fractional days instead of .dt.days
            dt_days = (df["ATD"] - df["ETD"]).dt.total_seconds() / 86400.0
            df["delay_days"] = dt_days

    return df


# -----------------------------
# Derived features
# -----------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # lane feature
    if {"POL", "POD"}.issubset(df.columns):
        df["lane"] = df["POL"].astype(str) + "->" + df["POD"].astype(str)

    # PATCH: guard for missing ETD/ATD and all-null bases
    etd = df["ETD"] if "ETD" in df.columns else pd.Series(pd.NaT, index=df.index)
    atd = df["ATD"] if "ATD" in df.columns else pd.Series(pd.NaT, index=df.index)
    base = etd.where(etd.notna(), atd)

    # If both ETD and ATD are missing for a row, month/weekday become NaN (fine)
    df["ETD_month"] = base.dt.month
    df["ETD_weekday"] = base.dt.weekday

    # vessel presence indicator
    # PATCH: guard when 'Vessel' is missing
    if "Vessel" in df.columns:
        df["has_vessel"] = df["Vessel"].notna().astype(int)
    else:
        df["has_vessel"] = 0

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


def _oof_predictions(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                     n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Compute out-of-fold predictions for the training set using the given pipeline.
    Returns an array aligned with y.index.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)

    for train_idx, valid_idx in kf.split(X):
        X_tr = X.iloc[train_idx]
        y_tr = y.values[train_idx]
        X_va = X.iloc[valid_idx]

        m = pickle.loads(pickle.dumps(pipeline))  # fresh clone
        m.fit(X_tr, y_tr)
        oof[valid_idx] = m.predict(X_va)

    return oof


def _train_residual_model(X: pd.DataFrame, y: pd.Series, mu_oof: np.ndarray) -> Pipeline:
    """
    Train a residual model to predict absolute error magnitude sigma(x) ~= E|y - mu(x)|.
    Reuses the same preprocessing as the base pipeline to keep feature parity.
    """
    abs_err = np.abs(y.values - mu_oof)

    # Rebuild the same preprocessor based on the columns present
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            # PATCH: include only if list non-empty (cosmetic, but explicit)
            *([("num", numeric_transformer, num_cols)] if len(num_cols) else []),
            *([("cat", categorical_transformer, cat_cols)] if len(cat_cols) else []),
        ]
    )

    resid_base = GradientBoostingRegressor(random_state=42)
    resid_pipe = Pipeline([("pre", pre), ("model", resid_base)])
    resid_pipe.fit(X, abs_err)
    return resid_pipe


def _fit_isotonic_per_threshold(mu_oof: np.ndarray, sigma_oof: np.ndarray,
                                y: pd.Series, thresholds_days: List[float]) -> Dict[float, IsotonicRegression]:
    """
    Fit isotonic regression calibrators mapping score -> probability for each threshold.
    score = (mu_oof - T) / sigma_oof
    """
    sigma_oof = np.clip(sigma_oof, 0.25, None)  # floor at 0.25 days ~ 6 hours

    calibrators = {}
    for T in thresholds_days:
        score = (mu_oof - T) / sigma_oof
        label = (y.values > T).astype(int)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(score, label)
        calibrators[T] = iso

    return calibrators


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = align_schema(df)
    df = add_derived_features(df)

    # drop missing target rows
    df = df.dropna(subset=["delay_days"])

    # keep only columns that exist
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # PATCH: explicit guard to avoid empty design matrix silently
    if not num_cols and not cat_cols:
        raise ValueError(
            "No model features available. "
            "Check your input data and feature lists."
        )

    X = df[num_cols + cat_cols]
    y = df["delay_days"].astype(float)

    return X, y, num_cols, cat_cols


def build_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
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
            # PATCH: add blocks only if there are columns
            *([("num", numeric_transformer, num_cols)] if len(num_cols) else []),
            *([("cat", categorical_transformer, cat_cols)] if len(cat_cols) else []),
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


def train_and_save_model(
    data_path="data/processed_shipments_data.csv",
    model_path="artifacts/delay_model.pkl",
    resid_model_path="artifacts/resid_model.pkl",
    calibrators_dir="artifacts/calibrators",
    thresholds_days=(5.0,),  # default primary threshold 1 day
    n_splits_oof=5
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(resid_model_path), exist_ok=True)
    Path(calibrators_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    X, y, num_cols, cat_cols = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1) Train base regressor (same as before)
    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.2f} days")

    # --- Additional accuracy indicators ---
    rmse = np.sqrt(np.mean((preds - y_test.values)**2))
    r2 = r2_score(y_test, preds)
    medae = median_absolute_error(y_test, preds)

    abs_errors = np.abs(preds - y_test.values)

    pct_within_0_5d = np.mean(abs_errors <= 0.5) * 100
    pct_within_1d   = np.mean(abs_errors <= 1.0) * 100
    pct_within_2d   = np.mean(abs_errors <= 2.0) * 100

    print(f"RMSE: {rmse:.2f} days")
    print(f"R²: {r2:.3f}")
    print(f"Median AE: {medae:.2f} days")

    print(f"Within 0.5 days: {pct_within_0_5d:.1f}%")
    print(f"Within 1.0 day:  {pct_within_1d:.1f}%")
    print(f"Within 2.0 days: {pct_within_2d:.1f}%")

    print(f"Error mean: {abs_errors.mean():.2f}, std: {abs_errors.std():.2f}")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved base model to {model_path}")

    # 2) Out-of-fold mu for the full training set (use all data for residuals/calibration)
    print("Computing out-of-fold predictions for residuals and calibration...")
    base_for_oof = build_pipeline(num_cols, cat_cols)  # same structure
    mu_oof = _oof_predictions(base_for_oof, X, y, n_splits=n_splits_oof, random_state=42)

    # 3) Train residual model sigma(x)
    print("Training residual model...")
    resid_model = _train_residual_model(X, y, mu_oof)
    with open(resid_model_path, "wb") as f:
        pickle.dump(resid_model, f)
    print(f"Saved residual model to {resid_model_path}")

    # 4) Compute sigma_oof using the trained residual model
    sigma_oof = resid_model.predict(X)
    sigma_oof = np.clip(sigma_oof, 0.25, None)

    # 5) Fit isotonic calibrators per threshold
    print(f"Fitting isotonic calibrators for thresholds: {thresholds_days}")
    calibrators = _fit_isotonic_per_threshold(mu_oof, sigma_oof, y, list(thresholds_days))

    # 6) Save calibrators
    for T, iso in calibrators.items():
        cal_path = Path(calibrators_dir) / f"iso_T{T}.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(iso, f)
        print(f"Saved calibrator: {cal_path}")

if __name__ == "__main__":
    train_and_save_model()