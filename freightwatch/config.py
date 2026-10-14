"""Central configuration: paths, thresholds, model settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export "), v.strip().strip("'\""))


_load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

SHIPMENTS_CSV = DATA_DIR / "shipment_records.csv"
NOTES_CSV = DATA_DIR / "context_notes.csv"
NOTES_STRUCTURED_JSON = DATA_DIR / "notes_structured.json"
GOLD_LABELS_CSV = ROOT / "eval" / "gold_labels.csv"

# Baselines (fixed by the brief).
HISTORY_WINDOW_WEEKS = 8

# Flag rule: rising vs own history AND (history rise >= HIST or peer gap >= PEER).
FLAG_HISTORY_PCT = 15.0
FLAG_PEER_PCT = 20.0

# A one-off event note with no stated end covers its own week;
# a step-change note (e.g. fuel) with no end covers this many weeks.
STEP_CHANGE_WEEKS = 8

# LLM: open-weights model on Groq's free tier, fully deterministic settings.
# (llama-3.3-70b-versatile was retired from Groq; gpt-oss-20b is Apache-2.0 open weights.)
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0
LLM_SEED = 7
LLM_REASONING_EFFORT = "low"   # without this the model spends its token budget thinking and returns ""
LLM_MAX_TOKENS = 400
# Published Groq rate for this model, USD per 1M tokens (console.groq.com/docs/model/openai/gpt-oss-20b).
PRICE_INPUT_PER_M = 0.075
PRICE_OUTPUT_PER_M = 0.30

# Corpus is 10 notes, so rank them all: retrieval orders, the verdict gates filter.
# At larger scale, prefilter on route/date metadata and lower this.
TOP_K_NOTES = 10
