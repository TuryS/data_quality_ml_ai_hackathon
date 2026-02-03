import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


FEATURES = [
    "port_congestion",
    "carrier_reliability",
    "transit_time_days"
]


def prepare_training_data(df):
    df = df.copy()
    df["transit_time_days"] = (df["ata"] - df["atd"]).dt.days
    df["delay_days"] = (df["actual_delivery"] - df["estimated_delivery"]).dt.days

    df = df.dropna(subset=FEATURES + ["delay_days"])

    X = df[FEATURES]
    y = df["delay_days"]

    return X, y


def train_and_save_model(data_path="data/shipments.csv",
                         model_path="artifacts/delay_model.pkl"):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["atd", "ata", "estimated_delivery", "actual_delivery"])

    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    mae = np.mean(np.abs(model.predict(X_test) - y_test))
    print(f"Model MAE: {mae:.2f} days")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_and_save_model()