"""Note retrieval. Chroma (vector DB) when installed, else a dependency-free TF-IDF cosine index.

Both rank all candidate notes best-first with deterministic tie-breaking. Retrieval only
orders notes; the deterministic gates in verdict.py decide whether any of them justify a flag.
"""
import math
import re
from collections import Counter

from .notes import Note

TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _doc(n: Note) -> str:
    return f"{n.applies_to}: {n.text}"


def _query(route: str, week_of) -> str:
    return f"cost increase on {route} route in the week of {week_of.date()}"


class TfidfIndex:
    backend = "tfidf-numpy"

    def __init__(self, notes: dict[str, Note]):
        self.notes = notes
        self.ids = sorted(notes)
        docs = {i: Counter(_tokens(_doc(notes[i]))) for i in self.ids}
        df = Counter(t for c in docs.values() for t in c)
        n = len(self.ids)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vecs = {i: self._vec(c) for i, c in docs.items()}

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {t: c * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, route: str, week_of, k: int) -> list[Note]:
        q = self._vec(Counter(_tokens(_query(route, week_of))))
        scored = [(-round(sum(w * self.vecs[i].get(t, 0.0) for t, w in q.items()), 6), i) for i in self.ids]
        return [self.notes[i] for _, i in sorted(scored)[:k]]


class ChromaIndex:
    backend = "chromadb"

    def __init__(self, notes: dict[str, Note]):
        import chromadb
        self.notes = notes
        self.col = chromadb.EphemeralClient().get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})
        ids = sorted(notes)
        self.col.upsert(ids=ids, documents=[_doc(notes[i]) for i in ids],
                        metadatas=[{"date": str(notes[i].date.date())} for i in ids])

    def search(self, route: str, week_of, k: int) -> list[Note]:
        res = self.col.query(query_texts=[_query(route, week_of)], n_results=min(k, len(self.notes)))
        hits = sorted(zip((round(d, 6) for d in res["distances"][0]), res["ids"][0]))
        return [self.notes[i] for _, i in hits]


def NoteIndex(notes: dict[str, Note]):
    try:
        return ChromaIndex(notes)
    except Exception:   # chromadb missing, or its embedding model unavailable offline
        return TfidfIndex(notes)
