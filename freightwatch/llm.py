"""Groq client wrapper: temperature 0, fixed seed, token accounting, on-disk response cache."""
import hashlib
import json
import os
from dataclasses import dataclass, field

from .config import (LLM_MAX_TOKENS, LLM_MODEL, LLM_REASONING_EFFORT, LLM_SEED, LLM_TEMPERATURE,
                     OUT_DIR, PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M)

CACHE_PATH = OUT_DIR / "llm_cache.json"


_PUNCT = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"} | {"\u202f": " ", "\u00a0": " ", "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


def clean(text: str) -> str:
    """Models emit non-breaking hyphens/spaces inside dates; normalise so validation and CSV are plain ASCII."""
    return text.translate(_PUNCT).strip()


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self) -> float:
        return (self.input_tokens * PRICE_INPUT_PER_M + self.output_tokens * PRICE_OUTPUT_PER_M) / 1e6

    def as_dict(self) -> dict:
        return {"model": LLM_MODEL, "temperature": LLM_TEMPERATURE, "seed": LLM_SEED, "reasoning_effort": LLM_REASONING_EFFORT,
                "llm_calls": self.calls, "cache_hits": self.cache_hits,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd_at_published_rate": round(self.cost_usd(), 6)}


@dataclass
class LLM:
    use_cache: bool = True
    usage: Usage = field(default_factory=Usage)
    _client: object = None
    _cache: dict = None

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    def _load_cache(self) -> dict:
        if self._cache is None:
            self._cache = json.loads(CACHE_PATH.read_text()) if (self.use_cache and CACHE_PATH.exists()) else {}
        return self._cache

    def complete(self, system: str, user: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
        key = hashlib.sha256(json.dumps([LLM_MODEL, system, user]).encode()).hexdigest()
        cache = self._load_cache()
        if self.use_cache and key in cache:
            self.usage.cache_hits += 1
            return cache[key]
        if self._client is None:
            from groq import Groq
            self._client = Groq()
        resp = self._client.chat.completions.create(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE, seed=LLM_SEED, max_tokens=max_tokens,
            reasoning_effort=LLM_REASONING_EFFORT,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self.usage.calls += 1
        self.usage.input_tokens += resp.usage.prompt_tokens
        self.usage.output_tokens += resp.usage.completion_tokens
        text = clean(resp.choices[0].message.content or "")
        if not text:
            raise RuntimeError(f"empty completion (finish_reason={resp.choices[0].finish_reason})")
        if self.use_cache:
            cache[key] = text
            OUT_DIR.mkdir(exist_ok=True)
            CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True))
        return text
