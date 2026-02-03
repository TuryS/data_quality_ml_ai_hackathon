import pickle
import pandas as pd

FEATURES = [
    "port_congestion",
    "carrier_reliability",
    "transit_time_days"
]

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].fillna(0)

def predict_delay(df: pd.DataFrame, model) -> pd.Series:
    X = build_features(df)
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="predicted_delay_days")