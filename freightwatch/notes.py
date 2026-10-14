"""Notes: raw text from context_notes.csv plus structured facts (route scope, window, effect)."""
import json
from dataclasses import dataclass

import pandas as pd

from .config import NOTES_CSV, NOTES_STRUCTURED_JSON, STEP_CHANGE_WEEKS


@dataclass(frozen=True)
class Note:
    note_id: str
    date: pd.Timestamp
    applies_to: str
    text: str
    scope: str                 # "route" | "all" | "none"
    routes: tuple
    effect: str                # "increase" | "none" | "decrease"
    start: pd.Timestamp
    end: pd.Timestamp          # inclusive last day the note covers
    magnitude_pct: tuple | None

    def covers_route(self, route: str) -> bool:
        return self.scope == "all" or route in self.routes

    def overlaps_week(self, week_of: pd.Timestamp) -> bool:
        week_end = week_of + pd.Timedelta(days=6)
        return self.start <= week_end and self.end >= week_of


def load_notes(raw_path=NOTES_CSV, structured_path=NOTES_STRUCTURED_JSON) -> dict[str, Note]:
    raw = pd.read_csv(raw_path)
    facts = json.loads(structured_path.read_text())
    missing = set(raw["note_id"]) - set(facts)
    if missing:
        raise ValueError(f"notes without structured facts: {sorted(missing)}")
    notes = {}
    for r in raw.itertuples():
        f = facts[r.note_id]
        start = pd.Timestamp(f["start"])
        if f["end"]:
            end = pd.Timestamp(f["end"])
        elif f["effect"] == "increase" and f["scope"] == "all":
            end = start + pd.Timedelta(weeks=STEP_CHANGE_WEEKS) - pd.Timedelta(days=1)
        else:
            end = start + pd.Timedelta(days=6)   # one-off: its own week
        mag = tuple(f["magnitude_pct"]) if f["magnitude_pct"] else None
        notes[r.note_id] = Note(r.note_id, pd.Timestamp(r.date), r.applies_to, r.note,
                                f["scope"], tuple(f["routes"]), f["effect"], start, end, mag)
    return notes
