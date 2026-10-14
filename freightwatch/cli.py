"""Command line: run | eval | repro."""
import argparse
import sys

import pandas as pd

from .config import GOLD_LABELS_CSV, OUT_DIR
from .pipeline import run, write_outputs


def cmd_run(args) -> int:
    out, weekly, log = run(use_llm=not args.no_llm, use_cache=not args.no_cache)
    write_outputs(out, weekly, log)
    print(f"wrote {len(out)} rows to {OUT_DIR / 'output.csv'}; log: {log}")
    return 0


def cmd_ask(args) -> int:
    from .llm import LLM
    from .qa import answer
    print(answer(args.question, LLM() if LLM.available() and not args.no_llm else None))
    return 0


def cmd_eval(args) -> int:
    out, _, _ = run(use_llm=False)
    gold = pd.read_csv(GOLD_LABELS_CSV, keep_default_na=False)
    m = out.merge(gold, on=["route", "week_of"], how="outer", suffixes=("", "_gold"), indicator=True)
    spurious, missed = m[m._merge == "left_only"], m[m._merge == "right_only"]
    both = m[m._merge == "both"]
    verdict_ok = (both.flagged == both.flagged_gold).sum()
    cite_ok = (both.matched_note_id == both.matched_note_id_gold).sum()
    fabricated = both[(both.matched_note_id != "") & (both.matched_note_id != both.matched_note_id_gold)]
    print(f"flag set: {len(both)} matched, {len(spurious)} spurious, {len(missed)} missed (gold={len(gold)})")
    print(f"verdict accuracy: {verdict_ok}/{len(both)}   citation accuracy: {cite_ok}/{len(both)}")
    print(f"fabricated/wrong citations: {len(fabricated)}")
    ok = len(spurious) == len(missed) == len(fabricated) == 0 and verdict_ok == cite_ok == len(both)
    print("EVAL PASS" if ok else "EVAL FAIL")
    return 0 if ok else 1


def cmd_repro(args) -> int:
    runs = []
    for i in range(3):
        out, weekly, log = run(use_llm=not args.no_llm, use_cache=False)
        runs.append((out, weekly))
        write_outputs(out, weekly, log, OUT_DIR / f"repro_run{i + 1}")
    base_out, base_w = runs[0]
    fixed = [c for c in base_out.columns if c != "reason"]
    same = all(o[fixed].equals(base_out[fixed]) and w.equals(base_w) for o, w in runs[1:])
    print("3 runs identical on all columns except reason (wording)" if same else "RUNS DIFFER")
    return 0 if same else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="freightwatch")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--no-llm", action="store_true"); r.add_argument("--no-cache", action="store_true")
    r.set_defaults(fn=cmd_run)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--no-llm", action="store_true")
    a.set_defaults(fn=cmd_ask)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    rp = sub.add_parser("repro"); rp.add_argument("--no-llm", action="store_true"); rp.set_defaults(fn=cmd_repro)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
