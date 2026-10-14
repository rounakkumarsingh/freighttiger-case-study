"""Weekly cost per tonne-km and the two comparison baselines defined in the brief."""
import pandas as pd

from .config import HISTORY_WINDOW_WEEKS


def weekly_cost_per_tonne_km(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (route, week_of): sum(cost) / sum(tonnes * km)."""
    d = df.assign(tonne_km=df["quantity_tonnes"] * df["distance_km"])
    g = d.groupby(["route", "route_type", "week_of"], as_index=False).agg(
        cost=("freight_cost_inr", "sum"), tonne_km=("tonne_km", "sum"), shipments=("shipment_id", "count"),
    )
    g["cpt"] = g["cost"] / g["tonne_km"]
    return g.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_baselines(weekly: pd.DataFrame, window: int = HISTORY_WINDOW_WEEKS) -> pd.DataFrame:
    """Add own-history mean (strictly prior weeks) and same-week peer mean (excluding self)."""
    w = weekly.copy()
    # shift(1) so the current week never sees itself or the future.
    w["hist_avg"] = w.groupby("route")["cpt"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    w["hist_weeks_used"] = w.groupby("route")["cpt"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).count()
    )
    grp = w.groupby(["route_type", "week_of"])["cpt"]
    n = grp.transform("count")
    total = grp.transform("sum")
    w["peer_avg"] = ((total - w["cpt"]) / (n - 1)).where(n > 1)
    w["vs_own_pct"] = (w["cpt"] / w["hist_avg"] - 1) * 100
    w["vs_peer_pct"] = (w["cpt"] / w["peer_avg"] - 1) * 100
    return w
