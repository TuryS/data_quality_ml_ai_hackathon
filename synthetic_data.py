import pandas as pd
import numpy as np
from datetime import timedelta
import os


def generate_synthetic_shipments(n=2000, seed=42):
    np.random.seed(seed)

    # Carrier-specific realism parameters
    carrier_profiles = {
        "Maersk": {
            "base_transit": 19,
            "transit_std": 3,
            "reliability_mean": 0.90,
            "congestion_sensitivity": 4,
            "late_bias": (0, 4),   # early to mild late
        },
        "MSC": {
            "base_transit": 22,
            "transit_std": 4,
            "reliability_mean": 0.80,
            "congestion_sensitivity": 5,
            "late_bias": (-1, 6),
        },
        "CMA CGM": {
            "base_transit": 21,
            "transit_std": 4.5,
            "reliability_mean": 0.78,
            "congestion_sensitivity": 6,
            "late_bias": (0, 8),
        },
        "Hapag-Lloyd": {
            "base_transit": 20,
            "transit_std": 3.5,
            "reliability_mean": 0.85,
            "congestion_sensitivity": 5,
            "late_bias": (-2, 5),
        },
        "ONE": {
            "base_transit": 23,
            "transit_std": 5,
            "reliability_mean": 0.75,
            "congestion_sensitivity": 7,
            "late_bias": (1, 10),
        },
    }

    origin_ports = ["Shanghai", "Shenzhen", "Singapore", "Rotterdam", "Dubai"]
    destination_ports = ["LA", "New York", "Hamburg", "Felixstowe", "Jebel Ali"]

    rows = []

    for _ in range(n):
        carrier = np.random.choice(list(carrier_profiles.keys()))
        profile = carrier_profiles[carrier]

        # realistic ATD
        atd = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            np.random.randint(0, 365), unit="days"
        )

        # base transit + variability
        transit_time = np.random.normal(
            profile["base_transit"], profile["transit_std"]
        )

        # congestion (dynamic)
        congestion = np.random.uniform(0, 1)

        # reliability affects early/late probability
        reliability = np.clip(
            np.random.normal(profile["reliability_mean"], 0.05), 0.5, 0.98
        )

        # adjusted transit incorporating congestion and reliability
        adjusted_transit = (
            transit_time
            + congestion * profile["congestion_sensitivity"]
            - ((reliability - 0.5) * 4)
        )

        transit_days = max(1, int(adjusted_transit))
        ata = atd + timedelta(days=transit_days)

        # estimated delivery = ATA - random 3–7 days
        est_delivery = ata - timedelta(days=np.random.randint(3, 7))

        # actual delivery early/late distribution per carrier
        early, late = profile["late_bias"]
        act_delivery = ata + timedelta(days=np.random.randint(early, late + 1))

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
