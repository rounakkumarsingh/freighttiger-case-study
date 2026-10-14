"""Stretch: answer plain-English questions using only the assistant's own results (out/output.csv)."""
import calendar
import re

import pandas as pd

from .config import OUT_DIR
from .explain import NOTE_ID
from .llm import LLM

SYSTEM = ("Answer the analyst's question using ONLY the findings provided. Cite week_of dates and note ids "
          "exactly as given. If the findings do not answer it, say so. Max 3 sentences.")
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def parse_question(q: str, routes: list[str]) -> tuple[list[str], int | None, int | None]:
    low = q.lower()
    found = [r for r in routes if r.lower() in low or all(p in low for p in r.lower().split("-"))]
    month = next((n for m, n in MONTHS.items() if re.search(rf"\b{m}\b", low)), None)
    year = next((int(y) for y in re.findall(r"\b(20\d\d)\b", low)), None)
    return found, month, year


def answer(question: str, llm: LLM | None = None, out_path=OUT_DIR / "output.csv") -> str:
    df = pd.read_csv(out_path, keep_default_na=False, parse_dates=["week_of"])
    all_routes = sorted(pd.read_csv(out_path.parent / "weekly_metrics.csv").route.unique())
    routes, month, year = parse_question(question, all_routes)
    hit = df
    if routes:
        hit = hit[hit.route.isin(routes)]
    if month:
        hit = hit[hit.week_of.dt.month == month]
    if year:
        hit = hit[hit.week_of.dt.year == year]
    if hit.empty or hit is df:
        scope = f" for {', '.join(routes)}" if routes else ""
        return f"No finding{scope}: the assistant flagged no matching route/week, so it has no rise to explain."
    facts = "\n".join(f"- {r.route}, week of {r.week_of.date()}: {r.cost_per_tonne_km} INR/t-km, "
                      f"{r.vs_own_history}; verdict {r.flagged}; note {r.matched_note_id or 'none'}. {r.reason}"
                      for r in hit.itertuples())
    if llm is None:
        return facts
    text = llm.complete(SYSTEM, f"Question: {question}\nFindings:\n{facts}", max_tokens=250)
    allowed = set(hit.matched_note_id) | set(NOTE_ID.findall(facts))
    return text if set(NOTE_ID.findall(text)) <= allowed else facts
