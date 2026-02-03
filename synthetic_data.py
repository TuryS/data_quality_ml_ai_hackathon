import pandas as pd
import numpy as np
from datetime import timedelta
import os


def generate_synthetic_shipments(n=2000, seed=42):
    np.random.seed(seed)

    carriers = ["Maersk", "MSC", "CMA CGM", "Hapag-Lloyd", "ONE"]
    lanes = ["Asia-Europe", "Asia-US", "Europe-US"]

    origin_ports = ["Shanghai", "Shenzhen", "Singapore", "Rotterdam", "Dubai"]
    destination_ports = ["LA", "New York", "Hamburg", "Felixstowe", "Jebel Ali"]

    rows = []

    for _ in range(n):
        carrier = np.random.choice(carriers)

        atd = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            np.random.randint(0, 365), unit="days"
        )

        transit_time = np.random.normal(20, 5)  # base
        congestion = np.random.uniform(0, 1)
        reliability = np.random.uniform(0.6, 0.95)

        adjusted_transit = transit_time + (congestion * 8) - ((reliability - 0.6) * 5)

        ata = atd + timedelta(days=max(1, int(adjusted_transit)))

        est_delivery = ata - timedelta(days=np.random.randint(0, 5))
        act_delivery = ata + timedelta(
            days=np.random.randint(-2, 8)
        )  # can be early or late

        rows.append(
            {
                "carrier": carrier,
                "origin_port": np.random.choice(origin_ports),
                "destination_port": np.random.choice(destination_ports),
                "atd": atd,
                "ata": ata,
                "estimated_delivery": est_delivery,
                "actual_delivery": act_delivery,
                "port_congestion": round(congestion, 3),
                "carrier_reliability": round(reliability, 3),
            }
        )

    return pd.DataFrame(rows)


def save_synthetic_data(path="data/shipments.csv", rows=2000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_synthetic_shipments(rows)
    df.to_csv(path, index=False)
    print(f"Saved synthetic dataset to {path}")


if __name__ == "__main__":
    save_synthetic_data()