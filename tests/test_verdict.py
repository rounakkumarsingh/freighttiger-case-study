import pandas as pd
import pytest
from types import SimpleNamespace

from freightwatch.notes import load_notes
from freightwatch.verdict import check_note, decide

NOTES = load_notes()
DECOYS = ["N004", "N005", "N006", "N007", "N008", "N009", "N010"]


def row(route, week, own=20.0, peer=10.0):
    return SimpleNamespace(route=route, week_of=pd.Timestamp(week), vs_own_pct=own, vs_peer_pct=peer)


def test_flood_note_justifies_both_flood_weeks():
    for wk in ("2025-02-24", "2025-03-03"):
        assert check_note(NOTES["N001"], row("Chennai-Bangalore", wk)) is None


def test_flood_note_does_not_cover_week_after_it_ended():
    assert check_note(NOTES["N001"], row("Chennai-Bangalore", "2025-03-10")) is not None


def test_festival_note_justifies_its_week_only():
    assert check_note(NOTES["N002"], row("Ahmedabad-Mumbai", "2025-01-20")) is None
    assert check_note(NOTES["N002"], row("Ahmedabad-Mumbai", "2025-01-27")) is not None


def test_note_for_wrong_route_is_rejected():
    assert "route" in check_note(NOTES["N002"], row("Mumbai-Pune", "2025-01-20"))


@pytest.mark.parametrize("nid", DECOYS)
@pytest.mark.parametrize("route", ["Mumbai-Pune", "Delhi-Jaipur", "Chennai-Bangalore", "Mumbai-Delhi", "Kolkata-Bhubaneswar"])
@pytest.mark.parametrize("week", ["2024-07-29", "2025-03-10", "2025-06-23", "2025-11-17", "2025-12-08"])
def test_decoy_notes_never_justify(nid, route, week):
    assert check_note(NOTES[nid], row(route, week)) is not None


def test_network_wide_diesel_note_cannot_explain_peer_gap():
    r = row("Mumbai-Pune", "2025-06-23", own=6.2, peer=20.6)
    assert check_note(NOTES["N003"], r) is not None
    assert check_note(NOTES["N003"], row("Mumbai-Pune", "2025-06-23", own=6.2, peer=5.0)) is None


def test_no_applicable_note_means_unexplained():
    v = decide(row("Delhi-Jaipur", "2024-11-11"), list(NOTES.values()))
    assert not v.justified and v.note is None
