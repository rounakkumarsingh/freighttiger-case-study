import pandas as pd
from types import SimpleNamespace

from freightwatch.explain import _valid
from freightwatch.notes import load_notes
from freightwatch.verdict import Verdict

NOTES = load_notes()
ROW = SimpleNamespace(week_of=pd.Timestamp("2025-03-03"))
JUST = Verdict(True, NOTES["N001"], {})
UNEX = Verdict(False, None, {"N009": "x"})


def test_refusal_is_rejected_for_both_verdicts():
    refusal = "I'm sorry, but I can't comply with that request."
    assert not _valid(refusal, ROW, JUST, None)
    assert not _valid(refusal, ROW, UNEX, NOTES["N009"])


def test_justified_must_cite_exactly_the_matched_note():
    ok = "Justified: the flood in note N001 (2025-02-24) raised costs."
    assert _valid(ok, ROW, JUST, None)
    assert not _valid(ok.replace("N001", "N002"), ROW, JUST, None)
    assert not _valid(ok + " Also N003.", ROW, JUST, None)


def test_dates_we_did_not_supply_are_rejected():
    assert not _valid("Justified: note N001 (2025-02-24) and 2025-04-01.", ROW, JUST, None)


def test_unexplained_cannot_claim_justified_or_invent_a_cause():
    assert _valid("Unexplained: N009 (2025-03-17) says nothing about a rise.", ROW, UNEX, NOTES["N009"])
    assert not _valid("Justified: diesel prices rose.", ROW, UNEX, NOTES["N009"])
    assert not _valid("Unexplained: see N001.", ROW, UNEX, NOTES["N009"])
