"""Write the `reason` text. The verdict is already fixed; the LLM only words it, and is validated."""
import re

from .llm import LLM
from .notes import Note
from .verdict import Verdict

SYSTEM = (
    "You write one short explanation (max 2 sentences) for a freight cost analyst. "
    "Use ONLY the facts given. Do not invent causes, numbers or dates. "
    "Start with the verdict word (Justified or Unexplained). Mention a note id and date only if given, exactly as given."
)
NOTE_ID = re.compile(r"\bN\d{3}\b")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _closest(row, v: Verdict, ranked: list[Note]) -> Note | None:
    """Closest in time among notes that name this route (or all routes); id breaks ties."""
    covering = [n for n in ranked if n.covers_route(row.route) and n.note_id in v.rejections]
    return min(covering, key=lambda n: (abs((n.date - row.week_of).days), n.note_id), default=None)


def template_reason(row, v: Verdict, ranked: list[Note]) -> str:
    if v.justified:
        n = v.note
        return (f"Justified: matches note {n.note_id} dated {n.date.date()}: {n.text} "
                f"The cost rise has a clear explanation.")
    near = _closest(row, v, ranked)
    if near is None:
        return ("Unexplained: no matching note found for this route or date range. "
                "Cost rise looks unexplained and worth a human review.")
    return (f"Unexplained: no note justifies this rise. The closest note ({near.note_id}, "
            f"{near.date.date()}) was ruled out because it {v.rejections[near.note_id]}. "
            f"Flagged for human review.")


def _valid(text: str, row, v: Verdict, near: Note | None) -> bool:
    """Reject refusals, wrong verdicts, and any note id or date we did not supply."""
    ids, dates = set(NOTE_ID.findall(text)), set(DATE.findall(text))
    week = str(row.week_of.date())
    if v.justified:
        return ("justified" in text.lower() and "unexplained" not in text.lower()
                and ids == {v.note.note_id} and dates <= {str(v.note.date.date()), week})
    allowed_ids = {near.note_id} if near else set()
    allowed_dates = {week} | ({str(near.date.date())} if near else set())
    return "unexplained" in text.lower() and "justified" not in text.lower().replace("not justified", "") \
        and ids <= allowed_ids and dates <= allowed_dates


def build_reason(row, v: Verdict, ranked: list[Note], llm: LLM | None) -> str:
    fallback = template_reason(row, v, ranked)
    if llm is None:
        return fallback
    near = None if v.justified else _closest(row, v, ranked)
    base = f"Route {row.route}, week of {row.week_of.date()}, cost rose {row.vs_own_pct:+.1f}% vs its own history."
    if v.justified:
        facts = (f"Verdict: justified. {base} Supporting note {v.note.note_id} "
                 f"({v.note.date.date()}): \"{v.note.text}\"")
    else:
        ctx = (f" The closest note, {near.note_id} ({near.date.date()}), was ruled out because it "
               f"{v.rejections[near.note_id]}." if near else " No note covers this route.")
        facts = (f"Verdict: unexplained. {base} No note justifies the rise.{ctx} "
                 f"Say it is unexplained and needs human review. Do not suggest any cause.")
    text = llm.complete(SYSTEM, facts)
    return text if _valid(text, row, v, near) else fallback
