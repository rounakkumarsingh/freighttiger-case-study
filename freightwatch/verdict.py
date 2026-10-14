"""Deterministic verdict: a note justifies a flag only if every gate passes.

The LLM never decides this. Retrieval proposes notes; these gates dispose.
"""
from dataclasses import dataclass

from .config import FLAG_PEER_PCT
from .notes import Note

MAGNITUDE_SLACK_PCT = 5.0   # own-history rise may exceed a stated max by this much


@dataclass(frozen=True)
class Verdict:
    justified: bool
    note: Note | None
    rejections: dict            # note_id -> why it was rejected (for the audit trail)


def check_note(note: Note, row) -> str | None:
    """Return None if the note justifies this row, otherwise the reason it does not."""
    if not note.covers_route(row.route):
        return "does not apply to this route"
    if not note.overlaps_week(row.week_of):
        return "is dated outside this week"
    if note.effect != "increase":
        return f"says costs were not raised (effect: {note.effect})"
    if note.scope == "all":
        # A network-wide shock lifts peers too, so it cannot explain a gap against peers.
        if row.vs_peer_pct >= FLAG_PEER_PCT:
            return "is network-wide, so it cannot explain a large gap versus peer routes"
        if note.magnitude_pct and row.vs_own_pct > note.magnitude_pct[1] + MAGNITUDE_SLACK_PCT:
            return "states a smaller rise than observed"
    return None


def decide(row, ranked_notes: list[Note]) -> Verdict:
    """ranked_notes: retrieval output, best first. First note passing all gates wins."""
    rejections = {}
    for note in ranked_notes:
        why = check_note(note, row)
        if why is None:
            return Verdict(True, note, rejections)
        rejections[note.note_id] = why
    return Verdict(False, None, rejections)
