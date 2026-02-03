import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse timestamps
    date_cols = [
        "eta", "ata",
        "etd", "atd",
        "estimated_delivery", "actual_delivery"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"atd", "ata"}.issubset(df.columns):
        df["transit_time_days"] = (df["ata"] - df["atd"]).dt.days

    if {"estimated_delivery", "actual_delivery"}.issubset(df.columns):
        df["delay_days"] = (
            df["actual_delivery"] - df["estimated_delivery"]
        ).dt.days

    return df


def basic_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["flag_negative_transit"] = df["transit_time_days"] < 0
    df["flag_negative_delay"] = df["delay_days"] < -5

    return df