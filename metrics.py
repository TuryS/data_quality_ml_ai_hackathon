import pandas as pd

def compute_quality_score(df: pd.DataFrame) -> float:
    mandatory_fields = [
        "carrier", "origin_port", "destination_port",
        "atd", "ata", "estimated_delivery", "actual_delivery"
    ]

    present_ratio = df[mandatory_fields].notnull().mean().mean()
    anomaly_penalty = (
        df["flag_negative_transit"].mean() +
        df["flag_negative_delay"].mean()
    )

    score = max(0.0, present_ratio - anomaly_penalty)
    return round(score, 2)


def kpi_summary(df: pd.DataFrame) -> dict:
    return {
        "avg_transit_time": round(df["transit_time_days"].mean(), 2),
        "avg_delay_days": round(df["delay_days"].mean(), 2),
        "on_time_ratio": round((df["delay_days"] <= 0).mean(), 2),
        "shipments": len(df),
    }


def delay_by_carrier(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("carrier", as_index=False)
        .agg(
            avg_delay=("delay_days", "mean"),
            shipments=("delay_days", "count")
        )
        .sort_values("avg_delay")
    )


def kpis_per_carrier(df: pd.DataFrame) -> dict:
    result = {}
    for carrier, grp in df.groupby("carrier"):
        result[carrier] = kpi_summary(grp)
    return result
