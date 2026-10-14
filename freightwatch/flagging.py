"""Candidate rule: cost is rising vs own history AND is large vs history or vs peers."""
import pandas as pd

from .config import FLAG_HISTORY_PCT, FLAG_PEER_PCT


def select_candidates(w: pd.DataFrame, hist_pct=FLAG_HISTORY_PCT, peer_pct=FLAG_PEER_PCT) -> pd.DataFrame:
    rising = w["vs_own_pct"] > 0
    unusual = (w["vs_own_pct"] >= hist_pct) | (w["vs_peer_pct"] >= peer_pct)
    return w[rising & unusual].reset_index(drop=True)
