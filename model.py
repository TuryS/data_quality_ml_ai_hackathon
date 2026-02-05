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

# -----------------------------
# Classifier-based probability inference (new)
# -----------------------------
from pathlib import Path
from typing import Dict, Union

class ClassifierBundle:
    """
    Holds one or more binary classifiers keyed by their threshold (in days).
    Expect each classifier to implement:
      - .named_steps.get("pre")  (optional, used by build_features)
      - .predict_proba(X) -> [n_samples, 2] with class 1 at column index 1
    """
    def __init__(self, by_threshold: Dict[float, object]):
        self.by_threshold = by_threshold or {}

    def available_thresholds(self):
        return sorted(self.by_threshold.keys())


def load_classifiers_from_dir(classifiers_dir: str) -> ClassifierBundle:
    """
    Loads pickled classifiers with filenames like:
      clf_T1.pkl, clf_T0.pkl, clf_T0.5.pkl
    Returns a ClassifierBundle keyed by float threshold in days.
    """
    by_T: Dict[float, object] = {}
    p = Path(classifiers_dir)
    if not p.exists():
        raise FileNotFoundError(f"classifiers_dir not found: {classifiers_dir}")
    for f in p.glob("clf_T*.pkl"):
        # extract numeric part after 'clf_T'
        name = f.stem  # 'clf_T1' -> '1'
        try:
            t_str = name.split("T", 1)[1]
            T = float(t_str)
        except Exception:
            continue
        by_T[T] = load_model(str(f))
    if not by_T:
        raise ValueError(f"No classifiers found in {classifiers_dir}. Expected files like clf_T1.pkl")
    return ClassifierBundle(by_T)


def _features_for_any_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Reuse the same feature assembly that your regressor expects.
    If the classifier has its own 'pre' with different columns, it will still work,
    because ColumnTransformer will select what it needs.
    """
    return build_features(df, model)


def predict_prob_over_thresholds_with_classifiers(
    df: pd.DataFrame,
    bundle: ClassifierBundle,
    thresholds_days: Union[float, list]
) -> pd.DataFrame:
    """
    For each threshold T provided, find a matching classifier and return a DataFrame
    with probability columns named p_over_T{T}.
    """
    if isinstance(thresholds_days, (int, float)):
        thresholds = [float(thresholds_days)]
    else:
        thresholds = [float(t) for t in thresholds_days]

    out = pd.DataFrame(index=df.index)

    for T in thresholds:
        clf = bundle.by_threshold.get(T)
        if clf is None:
            # If a classifier for T is missing, create a NaN column to make behavior explicit
            out[f"p_over_T{T:g}"] = np.nan
            continue

        X = _features_for_any_model(df, clf)
        # predict_proba returns columns in the order of classes; we assume class 1 is "delay > T"
        proba = clf.predict_proba(X)
        if proba.shape[1] == 2:
            p1 = proba[:, 1]
        else:
            # fall back if a different classifier shape; safer to compute as 1 - first
            p1 = 1.0 - proba[:, 0]

        out[f"p_over_T{T:g}"] = pd.Series(p1, index=df.index).clip(1e-6, 1 - 1e-6)

    return out


def predict_delay_and_probs(
    df: pd.DataFrame,
    regressor_model,                         # your existing delay regressor pipeline
    classifier_bundle: ClassifierBundle,     # loaded with load_classifiers_from_dir(...)
    thresholds_days: Union[float, list] = 1.0
) -> pd.DataFrame:
    """
    Convenience function that returns both:
      - predicted_delay_days (from regressor)
      - p_over_T{T} columns (from classifiers)
    """
    # 1) Regressor remains unchanged
    delay = predict_delay(df, regressor_model)

    # 2) Classifier-based probabilities
    probs = predict_prob_over_thresholds_with_classifiers(df, classifier_bundle, thresholds_days)

    # 3) Combine
    out = pd.DataFrame(index=df.index)
    out["predicted_delay_days"] = delay
    for c in probs.columns:
        out[c] = probs[c]
    return out

# ============================================
# PROBABILITIES FROM REGRESSOR + RESIDUAL + ISOTONIC
# ============================================
from pathlib import Path

class CalibratedProbArtifacts:
    def __init__(self, base_model, resid_model, calibrators):
        self.base_model = base_model
        self.resid_model = resid_model
        self.calibrators = calibrators  # dict {threshold(float): IsotonicRegression}

def load_calibrated_prob_artifacts(
    base_model_path: str = "artifacts/delay_model.pkl",
    resid_model_path: str = "artifacts/resid_model.pkl",
    calibrators_dir: str = "artifacts/calibrators"
) -> CalibratedProbArtifacts:
    base = load_model(base_model_path)
    resid = load_model(resid_model_path)
    calibs = {}

    p = Path(calibrators_dir)
    if not p.exists():
        raise FileNotFoundError(f"calibrators_dir not found: {calibrators_dir}")
    # expected file names like: iso_T1.pkl or iso_T1.0.pkl
    for f in p.glob("iso_T*.pkl"):
        stem = f.stem  # e.g., 'iso_T1.0'
        t_str = stem.split("T", 1)[1]
        try:
            T = float(t_str)
        except ValueError:
            continue
        calibs[T] = load_model(str(f))

    if not calibs:
        raise ValueError(f"No calibrators found in {calibrators_dir}")
    return CalibratedProbArtifacts(base, resid, calibs)

def predict_prob_over_thresholds_from_regression(
    df: pd.DataFrame,
    artifacts: CalibratedProbArtifacts,
    thresholds_days
) -> pd.DataFrame:
    """Returns columns p_over_T{T} using score=(mu - T)/sigma and isotonic mapping."""
    if isinstance(thresholds_days, (int, float)):
        thresholds = [float(thresholds_days)]
    else:
        thresholds = [float(t) for t in thresholds_days]

    # 1) mean delay (mu)
    mu = predict_delay(df, artifacts.base_model)

    # 2) features + sigma from residual model
    X = build_features(df, artifacts.base_model)
    sigma = pd.Series(artifacts.resid_model.predict(X), index=df.index).clip(lower=0.25)  # ~6h floor

    out = pd.DataFrame(index=df.index)
    out["predicted_delay_days"] = mu

    # 3) calibrated probabilities per threshold
    for T in thresholds:
        iso = artifacts.calibrators.get(T)
        col = f"p_over_T{T:g}"
        if iso is None:
            out[col] = np.nan
            continue
        score = (mu - T) / sigma
        p = iso.predict(score.values)
        out[col] = pd.Series(np.clip(p, 1e-6, 1 - 1e-6), index=df.index)
    return out
