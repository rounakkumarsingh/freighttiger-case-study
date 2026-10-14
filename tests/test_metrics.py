import pandas as pd
import pytest

from freightwatch.data import load_shipments
from freightwatch.flagging import select_candidates
from freightwatch.metrics import add_baselines, weekly_cost_per_tonne_km


@pytest.fixture(scope="module")
def weekly():
    return add_baselines(weekly_cost_per_tonne_km(load_shipments()))


def row(w, route, week):
    return w[(w.route == route) & (w.week_of == pd.Timestamp(week))].iloc[0]


# The three rows in sample_output_format_v2.csv are the ground truth for definitions.
@pytest.mark.parametrize("route,week,cpt,own,peer", [
    ("Delhi-Jaipur", "2024-11-11", 4.17, 35.5, 21.0),
    ("Ahmedabad-Mumbai", "2025-01-20", 3.29, 29.5, 22.5),
    ("Mumbai-Pune", "2025-09-15", 3.98, 9.2, 23.6),
])
def test_matches_sample_rows(weekly, route, week, cpt, own, peer):
    r = row(weekly, route, week)
    assert round(r.cpt, 2) == cpt
    assert round(r.vs_own_pct, 1) == own
    assert round(r.vs_peer_pct, 1) == peer


def test_week_is_monday(weekly):
    assert (weekly.week_of.dt.weekday == 0).all()


def test_sunday_belongs_to_previous_monday():
    df = load_shipments()
    sun = df[df.shipment_date.dt.weekday == 6].iloc[0]
    assert sun.week_of == sun.shipment_date - pd.Timedelta(days=6)


def test_first_week_has_no_history_and_early_weeks_use_fewer(weekly):
    first = weekly[weekly.route == "Mumbai-Pune"].iloc[0]
    assert pd.isna(first.hist_avg)
    third = weekly[weekly.route == "Mumbai-Pune"].iloc[3]
    assert third.hist_weeks_used == 3


def test_history_has_no_lookahead(weekly):
    r = weekly[weekly.route == "Chennai-Bangalore"]
    i = 20
    prior = r.cpt.iloc[i - 8:i].mean()
    assert r.hist_avg.iloc[i] == pytest.approx(prior)


def test_peer_excludes_self(weekly):
    w = pd.Timestamp("2025-01-20")
    med = weekly[(weekly.week_of == w) & (weekly.route_type == "Medium")]
    me = med[med.route == "Ahmedabad-Mumbai"].iloc[0]
    others = med[med.route != "Ahmedabad-Mumbai"].cpt.mean()
    assert me.peer_avg == pytest.approx(others)


def test_candidates_include_sample_rows_and_are_rising(weekly):
    c = select_candidates(weekly)
    keys = set(zip(c.route, c.week_of.dt.strftime("%Y-%m-%d")))
    assert ("Delhi-Jaipur", "2024-11-11") in keys
    assert ("Ahmedabad-Mumbai", "2025-01-20") in keys
    assert ("Mumbai-Pune", "2025-09-15") in keys
    assert (c.vs_own_pct > 0).all()
    assert len(c) == 19
