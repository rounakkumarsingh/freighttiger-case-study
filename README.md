# FreightWatch

Flags freight routes whose cost per tonne-km is rising unusually, then says whether a note in `context_notes.csv` genuinely explains it. No supporting note means "unexplained", never a guess.

```
cd freightwatch && uv venv && uv pip install pandas groq pytest chromadb
echo 'GROQ_API_KEY=...' > .env                   # optional: without it reasons use a fixed template
python -m freightwatch.cli run                  # -> out/output.csv, out/weekly_metrics.csv, out/run_log.json
python -m freightwatch.cli eval                 # score against eval/gold_labels.csv
python -m freightwatch.cli repro                # 3 runs, diff everything except reason wording
python -m freightwatch.cli ask "why did Chennai-Bangalore get pricier in March 2025?"
python -m pytest
```

## How it works
1. **Weekly metric** (`data.py`, `metrics.py`): `week_of` = Monday of the shipment date. Cost per tonne-km = sum(cost) / sum(tonnes x km) over the route-week. Confirmed against the three rows in `sample_output_format_v2.csv` (4.17 / 3.29 / 3.98); the mean of per-shipment ratios does not match them.
2. **Baselines**, exactly as the brief defines: own history is the mean of the up-to-8 weeks strictly before this one (no look-ahead); peers are the same-week mean of the other routes with the same `route_type`, excluding the route itself. All three sample rows' percentages reproduce (+35.5/+21.0, +29.5/+22.5, +9.2/+23.6).
   - **Fewer than 8 prior weeks**: weeks 2-8 of every route use all prior weeks available (`hist_weeks_used` in `weekly_metrics.csv` shows how many); week 1 has no history, so it has no own-history figure and is never flagged. Nothing is padded or extrapolated.
3. **Flag candidates** (`flagging.py`): rising vs own history AND (rise >= 15% OR gap vs peers >= 20%). This yields 19 rows and includes all three sample rows. Sensitivity of the candidate count: 15/20 -> 19, 12/20 -> 19, 15/18 -> 20, 20/25 -> 13. The thresholds are my judgement; the weekly noise is about 4% (std of the history comparison), so 15% is far above noise.
4. **Retrieval** (`retrieval.py`): each candidate queries the notes; all 10 notes are ranked (the corpus is tiny, so retrieval orders and the gates filter).
5. **Verdict** (`verdict.py`), deterministic, the LLM has no say: a note justifies a row only if it (a) names this route or "All Routes", (b) its dated window overlaps the week, (c) it describes a cost *increase*, and (d) a network-wide note (e.g. diesel) cannot justify a large gap versus peers, since peers got the same shock. Note windows come from `data/notes_structured.json`, which I wrote by hand from the note text (e.g. N001 ends Mar 8, so the week of Mar 10 is *not* covered). Otherwise `matched_note_id` is blank and the row is `Yes`.
6. **Reason text** (`explain.py`): the LLM (gpt-oss-20b on Groq, temperature 0, seed 7, reasoning_effort low) only words the fixed verdict. Output is validated: a justified reason must cite exactly the matched note id and only known dates; an unexplained reason must cite no note. Failing validation falls back to a template.

## Result
19 candidate rows: 3 justified (N002 for Ahmedabad-Mumbai 2025-01-20; N001 for Chennai-Bangalore 2025-02-24 and 03-03) and 16 unexplained. The decoys behave: Chennai-Bangalore 03-10/03-17 stay unexplained (flood ended Mar 8; N009 says back to normal), Delhi-Jaipur Nov 2024 has no note, and the Mumbai-Pune creep Jun-Dec 2025 is unexplained (N006 says demand was stable; N003 diesel cannot explain a gap versus peers).

## How I checked it
- `pytest` (193 tests): metric vs the sample rows, Monday week boundaries, no look-ahead, peer excludes self, and every non-justifying note (N004-N010) tested against 5 routes x 5 weeks to prove it can never justify a flag.
- `eval/gold_labels.csv`: 19 rows labelled by me from reading the notes before looking at the system output. `eval` scores flag set, verdict, and citation: 19/19, 0 spurious, 0 missed, 0 wrong citations. This is a small hand-built set, so it shows the gates behave as I intended, not that my reading of the notes is the only valid one.
- Caught while building: the Q&A answered about the wrong route when a route had no flags; fixed and covered by a test.

## Reproducibility
`repro` runs the pipeline 3 times with the response cache off. Result: `output.csv` columns other than `reason` and `weekly_metrics.csv` are identical across runs, and the note ids cited inside the reason text are identical too. Reason *wording* differs slightly between runs (allowed by the brief); `out/repro_run{1,2,3}/` hold the three outputs. Verdicts and note ids never depend on the LLM (the gates decide), so temperature cannot move them. The LLM runs at temperature 0, seed 7, `reasoning_effort=low`, and a cached rerun is byte-identical to the run that filled the cache.

## LLM and cost (one full run over the shipment file)
Model: `openai/gpt-oss-20b` on Groq (open weights, Apache-2.0). `llama-3.3-70b-versatile` was my first choice but Groq no longer serves it (404), so I listed the key's models and switched. gpt-oss is a reasoning model: without `reasoning_effort=low` it spent its whole token budget thinking and returned an empty string, which the wrapper now treats as an error.

| run | LLM calls | input tokens | output tokens | cost at published rate |
|---|---|---|---|---|
| cold (`out/run_log_cold.json`) | 19 | 4,084 | 1,910 | $0.0009 |
| cached rerun (`out/run_log_cached.json`) | 0 (19 cache hits) | 0 | 0 | $0 |

Rate used: $0.075 / 1M input, $0.30 / 1M output (Groq's gpt-oss-20b page). Output tokens vary by a few percent run to run (1,910 to 1,986 seen). One call per flagged row; the per-week metrics, retrieval and verdicts use no LLM.

## Things the LLM got wrong and how that is handled
- On the first full run, 13 of 16 unexplained rows came back as "I'm sorry, but I can't comply with that request." My first validator accepted them because a refusal cites no note or date. Fixed two ways: the prompt now gives the closest ruled-out note and why, and the validator requires the verdict word, rejects the opposite verdict, and rejects any note id or date that was not supplied. Anything failing validation falls back to a template (1 of 19 in the final run). Covered by `tests/test_explain.py`.
- The model writes non-breaking hyphens inside dates; text is normalised to ASCII before validating and saving.

## Remaining limitations
- `notes_structured.json` (each note's route scope, window and effect) is hand-written, not LLM-extracted. A new note needs an entry added.
- `eval/gold_labels.csv` is my own 19-row labelling, so it checks the gates behave as I intended, not that my reading is the only valid one.
- Retrieval uses Chroma (local MiniLM embeddings) with a TF-IDF fallback if Chroma is unavailable; `run_log.json` records which was used. With only 10 notes all are ranked and the gates do the filtering; at scale you would prefilter on route and date.
- The flag thresholds (15% vs own history, 20% vs peers) are my judgement; see the sensitivity table above.
