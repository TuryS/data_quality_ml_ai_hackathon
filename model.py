# model.py
import pickle
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Model load
# -----------------------------
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# Column helpers for inference
# -----------------------------
def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _ensure_datetime(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")


# -----------------------------
# Align schema and derive features
# (robust to upper or lower case timestamp columns)
# -----------------------------
def align_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Map possible variants to canonical uppercase names used in training
    etd_col = _first_existing(df, ["ETD", "etd"])
    atd_col = _first_existing(df, ["ATD", "atd"])
    eta_col = _first_existing(df, ["ETA", "eta"])
    ata_col = _first_existing(df, ["ATA", "ata"])

    # Coerce datetimes
    for src, dst in [(etd_col, "ETD"), (atd_col, "ATD"), (eta_col, "ETA"), (ata_col, "ATA")]:
        if src is not None:
            df["__tmp__" + dst] = pd.to_datetime(df[src], errors="coerce")
            df[dst] = df["__tmp__" + dst]
            df.drop(columns=["__tmp__" + dst], inplace=True, errors="ignore")
        else:
            # create empty if missing
            df[dst] = pd.NaT

    _ensure_datetime(df, ["ETD", "ATD", "ETA", "ATA"])

    # Compute transit_time_days if missing (training used ATD and ATA)
    if "transit_time_days" not in df.columns:
        if df["ATD"].notna().any() or df["ATA"].notna().any():
            df["transit_time_days"] = (df["ATA"] - df["ATD"]).dt.days
        else:
            df["transit_time_days"] = np.nan

    # Derived features that training used
    pol_col = _first_existing(df, ["POL", "pol"])
    pod_col = _first_existing(df, ["POD", "pod"])
    if pol_col is not None and pod_col is not None:
        df["lane"] = df[pol_col].astype(str) + "->" + df[pod_col].astype(str)
    else:
        # keep lane present so the column selector does not fail, imputer will handle NaN
        df["lane"] = np.nan

    # Time features from ETD preferred over ATD when ETD missing
    base = df["ETD"].where(df["ETD"].notna(), df["ATD"])
    df["ETD_month"] = base.dt.month
    df["ETD_weekday"] = base.dt.weekday

    # Vessel presence feature used in training
    vessel_col = _first_existing(df, ["Vessel", "vessel"])
    df["has_vessel"] = df[vessel_col].notna().astype(int) if vessel_col else 0

    return df


# -----------------------------
# Read the expected schema from the fitted pipeline
# -----------------------------
def _expected_columns_from_pipeline(model) -> Tuple[List[str], List[str]]:
    """
    Extract numeric and categorical column lists from the saved ColumnTransformer.
    Assumes names 'pre' for the preprocessor, and transformer names 'num' and 'cat'.
    """
    pre = model.named_steps.get("pre")
    if pre is None:
        # No preprocessor wrapped, assume direct model that accepts numeric matrix
        return [], []

    num_cols = []
    cat_cols = []
    # pre.transformers is a list of tuples: (name, transformer, column_selector)
    for name, _, cols in getattr(pre, "transformers", []):
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    # Fallback to fitted attribute if needed
    if not num_cols or not cat_cols:
        for name, _, cols in getattr(pre, "transformers_", []):
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)

    return num_cols, cat_cols


def _ensure_expected_columns(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """
    Make sure all expected columns exist with plausible dtypes.
    Imputers in the pipeline will take care of NaN.
    """
    df = df.copy()

    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        # try to coerce to numeric when present
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in cat_cols:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df), dtype="object")
        else:
            df[c] = df[c].astype("object")

    return df


# -----------------------------
# Public API
# -----------------------------
def build_features(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Build and return the exact feature frame the pipeline expects, in the right dtypes.
    """
    df2 = align_and_derive(df)
    num_cols, cat_cols = _expected_columns_from_pipeline(model)
    # If the model has no preprocessor, return what we have
    if not num_cols and not cat_cols:
        return df2

    df2 = _ensure_expected_columns(df2, num_cols, cat_cols)
    # Slice to the expected order
    return df2[num_cols + cat_cols]


def predict_delay(df: pd.DataFrame, model) -> pd.Series:
    X = build_features(df, model)
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="predicted_delay_days")
