"""End to end: shipments -> weekly metrics -> candidates -> retrieval -> verdict -> output.csv."""
import json

import pandas as pd

from .config import OUT_DIR, TOP_K_NOTES
from .data import load_shipments
from .explain import build_reason
from .flagging import select_candidates
from .llm import LLM
from .metrics import add_baselines, weekly_cost_per_tonne_km
from .notes import load_notes
from .retrieval import NoteIndex
from .verdict import decide

OUTPUT_COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
                  "flagged", "matched_note_id", "reason"]


def compute_weekly() -> pd.DataFrame:
    return add_baselines(weekly_cost_per_tonne_km(load_shipments()))


def run(use_llm: bool = True, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    weekly = compute_weekly()
    candidates = select_candidates(weekly)
    notes = load_notes()
    index = NoteIndex(notes)
    llm = LLM(use_cache=use_cache) if use_llm and LLM.available() else None

    rows = []
    for c in candidates.itertuples():
        ranked = index.search(c.route, c.week_of, TOP_K_NOTES)
        v = decide(c, ranked)
        rows.append({
            "route": c.route,
            "week_of": c.week_of.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": f"{c.cpt:.2f}",
            "vs_own_history": f"{c.vs_own_pct:+.1f}% vs this route's past average",
            "vs_similar_routes": f"{c.vs_peer_pct:+.1f}% vs similar-length routes this week",
            "flagged": "No (justified)" if v.justified else "Yes",
            "matched_note_id": v.note.note_id if v.justified else "",
            "reason": build_reason(c, v, ranked, llm),
        })
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    log = llm.usage.as_dict() if llm else {"llm_calls": 0, "note": "ran without LLM (template reasons)"}
    log["candidates"] = len(out)
    log["retrieval_backend"] = index.backend
    return out, weekly, log


def write_outputs(out: pd.DataFrame, weekly: pd.DataFrame, log: dict, out_dir=OUT_DIR) -> None:
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "output.csv", index=False)
    cols = ["route", "route_type", "week_of", "shipments", "cpt", "hist_avg", "hist_weeks_used",
            "peer_avg", "vs_own_pct", "vs_peer_pct"]
    metrics = weekly[cols].copy()
    num = metrics.select_dtypes("number").columns
    metrics[num] = metrics[num].round(4)
    metrics.to_csv(out_dir / "weekly_metrics.csv", index=False, date_format="%Y-%m-%d")
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2))
