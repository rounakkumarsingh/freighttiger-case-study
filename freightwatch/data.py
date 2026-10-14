"""Load and validate the shipment records."""
import pandas as pd

from .config import SHIPMENTS_CSV

REQUIRED = [
    "shipment_id", "origin", "destination", "route_type", "quantity_tonnes",
    "distance_km", "freight_cost_inr", "shipment_date",
]


def load_shipments(path=SHIPMENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"shipment file missing columns: {missing}")
    for col in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
        if (pd.to_numeric(df[col], errors="coerce") <= 0).any():
            raise ValueError(f"non-positive or non-numeric values in {col}")
    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]
    # Weeks run Monday to Sunday; week_of is that week's Monday.
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    return df
