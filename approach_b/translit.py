"""Indic transliteration for v2 retrieval only.

The matcher never imports this module. Positional mining keeps every token after
abbreviation expansion. Unseen tokens go through an akshara-to-Latin character
model and then snap to a real Source-1 vocabulary item.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import text

INDIC = text.INDIC_CHAR
EXPAND = text.EXPAND
LEGAL = text.LEGAL

_VIRAMA = 0x4D
_NUKTA = 0x3C
_SCRIPT_BASES = text._SCRIPT_BASES
_CONSONANT = set(range(0x15, 0x3A))
_INDEP_VOWEL = set(range(0x05, 0x15))
_VOWEL_SIGN = set(range(0x3E, 0x4D))
_NASAL = {0x01, 0x02, 0x03}


def latin_tokens(name: str) -> list[str]:
    return [EXPAND.get(tok, tok) for tok in text.tokens(name)]


def native_tokens(name: str) -> list[str]:
    return [tok for tok in (name or "").split() if tok]


def mine_positional(pairs: list[tuple[str, str]], min_support: int = 2, min_purity: float = 0.5) -> dict[str, str]:
    """Zip Latin and native tokens by position when counts match. Keep majority maps."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for latin, native in pairs:
        src = latin_tokens(latin)
        tgt = native_tokens(native)
        if len(src) != len(tgt) or not tgt:
            continue
        for a, b in zip(tgt, src):
            if INDIC.search(a) and b:
                votes[a][b] += 1
    out: dict[str, str] = {}
    for native, counts in votes.items():
        latin, support = counts.most_common(1)[0]
        total = sum(counts.values())
        if support >= min_support and support / total >= min_purity:
            out[native] = latin
    return out


def _offset(char: str) -> tuple[int, int] | None:
    code = ord(char)
    for base in _SCRIPT_BASES:
        if base <= code < base + 0x80:
            return base, code - base
    return None


def akshara_split(token: str) -> list[str]:
    """Group a Brahmic token into aksharas (conjunct + vowel sign)."""
    if not token:
        return []
    parts: list[str] = []
    buf: list[str] = []
    pending_virama = False
    for char in token:
        info = _offset(char)
        if info is None:
            if buf:
                parts.append("".join(buf))
                buf = []
            pending_virama = False
            if char.strip():
                parts.append(char)
            continue
        _, off = info
        if off == _VIRAMA:
            buf.append(char)
            pending_virama = True
            continue
        if off == _NUKTA or off in _VOWEL_SIGN or off in _NASAL:
            if not buf:
                buf.append(char)
            else:
                buf.append(char)
            pending_virama = False
            continue
        start_new = bool(buf) and not pending_virama and (off in _CONSONANT or off in _INDEP_VOWEL)
        if start_new:
            parts.append("".join(buf))
            buf = [char]
        else:
            buf.append(char)
        pending_virama = False
    if buf:
        parts.append("".join(buf))
    return parts or [token]


def _char_trigrams(word: str) -> set[str]:
    padded = f"  {word} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def char_jaccard(left: str, right: str) -> float:
    a, b = _char_trigrams(left), _char_trigrams(right)
    return len(a & b) / max(len(a | b), 1)


def train_m2m(pairs: list[tuple[str, str]], iterations: int = 6) -> dict[str, dict[str, float]]:
    """Monotonic many-to-many EM: 1–2 aksharas emit 1–4 Latin characters."""
    examples: list[tuple[list[str], str]] = []
    for native, latin in pairs:
        src = akshara_split(native)
        tgt = re.sub(r"[^a-z0-9]", "", (latin or "").lower())
        if src and tgt and len(src) <= 24 and len(tgt) <= 32:
            examples.append((src, tgt))
    table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 1.0))
    if not examples:
        return {}
    for _ in range(iterations):
        count: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total: dict[str, float] = defaultdict(float)
        for src, tgt in examples:
            _collect(src, tgt, table, count, total)
        table = defaultdict(lambda: defaultdict(float))
        for unit, chunks in count.items():
            z = sum(chunks.values()) or 1.0
            for chunk, value in chunks.items():
                table[unit][chunk] = value / z
    return {k: dict(v) for k, v in table.items()}


def _units(src: list[str], i: int) -> list[tuple[int, str]]:
    out = [(1, src[i - 1])]
    if i >= 2:
        out.append((2, src[i - 2] + src[i - 1]))
    return out


def _forward(src: list[str], tgt: str, table: dict[str, dict[str, float]]) -> list[list[float]]:
    n, m = len(src), len(tgt)
    alpha = [[0.0] * (m + 1) for _ in range(n + 1)]
    alpha[0][0] = 1.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc = 0.0
            for span, unit in _units(src, i):
                if i < span:
                    continue
                for k in range(1, min(4, j) + 1):
                    chunk = tgt[j - k : j]
                    acc += alpha[i - span][j - k] * table.get(unit, {}).get(chunk, 1e-8)
            alpha[i][j] = acc
    return alpha


def _collect(src: list[str], tgt: str, table, count, total) -> None:
    n, m = len(src), len(tgt)
    alpha = _forward(src, tgt, table)
    if alpha[n][m] <= 0:
        return
    beta = [[0.0] * (m + 1) for _ in range(n + 1)]
    beta[n][m] = 1.0
    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            if beta[i][j] == 0:
                continue
            for span, unit in _units(src, i):
                if i < span:
                    continue
                for k in range(1, min(4, j) + 1):
                    chunk = tgt[j - k : j]
                    p = table.get(unit, {}).get(chunk, 1e-8)
                    mass = alpha[i - span][j - k] * p * beta[i][j] / alpha[n][m]
                    if mass <= 0:
                        continue
                    count[unit][chunk] += mass
                    total[unit] += mass
                    beta[i - span][j - k] += beta[i][j] * p


def decode_nbest(token: str, table: dict[str, dict[str, float]], nbest: int = 5) -> list[str]:
    src = akshara_split(token)
    if not src:
        return []
    layers: dict[int, dict[str, float]] = {0: {"": 0.0}}
    n = len(src)
    for i in range(n):
        if i not in layers:
            continue
        for roman, logp in list(layers[i].items()):
            choices = [(1, src[i])]
            if i + 1 < n:
                choices.append((2, src[i] + src[i + 1]))
            for span, unit in choices:
                emissions = table.get(unit) or {text.brahmic_romanize(unit): 1.0}
                ranked = sorted(emissions.items(), key=lambda kv: -kv[1])[:8]
                bucket = layers.setdefault(i + span, {})
                for chunk, p in ranked:
                    score = logp + math.log(max(p, 1e-12))
                    cand = roman + chunk
                    prev = bucket.get(cand)
                    if prev is None or score > prev:
                        bucket[cand] = score
        keep = sorted(layers[i].items(), key=lambda kv: -kv[1])[: max(nbest * 4, 16)]
        layers[i] = dict(keep)
    final = layers.get(n) or {text.brahmic_romanize(token): 0.0}
    ordered = [word for word, _ in sorted(final.items(), key=lambda kv: -kv[1]) if word]
    return ordered[:nbest] or [text.brahmic_romanize(token)]


class VocabSnapper:
    def __init__(self, words: list[str]):
        self.words = []
        self.trigram: dict[str, list[int]] = defaultdict(list)
        self.skeleton: dict[str, list[int]] = defaultdict(list)
        seen: set[str] = set()
        for word in words:
            token = (word or "").lower()
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            index = len(self.words)
            self.words.append(token)
            grams = _char_trigrams(token)
            for gram in list(grams)[:12]:
                if len(self.trigram[gram]) < 80:
                    self.trigram[gram].append(index)
            self.skeleton[text.phonetic_skeleton(token)].append(index)

    def snap(self, roman: str, candidates: list[str] | None = None) -> str:
        pool = [roman] + [c for c in (candidates or []) if c != roman]
        best_word, best_score = roman, -1.0
        for cand in pool:
            skel = text.phonetic_skeleton(cand)
            hits: set[int] = set(self.skeleton.get(skel, ()))
            for gram in _char_trigrams(cand):
                hits.update(self.trigram.get(gram, ()))
            if not hits:
                score = 0.0
                word = cand
            else:
                word, score = max(((self.words[i], char_jaccard(cand, self.words[i])) for i in hits), key=lambda kv: kv[1])
            if skel and skel in self.skeleton and score < 0.55:
                word = self.words[self.skeleton[skel][0]]
                score = max(score, 0.55)
            if score > best_score:
                best_word, best_score = word, score
        if best_score >= 0.35:
            return best_word
        return roman


def apply_token(
    token: str,
    dictionary: dict[str, str],
    table: dict[str, dict[str, float]],
    snapper: VocabSnapper | None,
    cache: dict[str, str],
) -> str:
    if token in cache:
        return cache[token]
    if token in dictionary:
        cache[token] = dictionary[token]
        return cache[token]
    if not INDIC.search(token):
        mapped = EXPAND.get(token.lower(), token.lower())
        cache[token] = mapped
        return mapped
    guesses = decode_nbest(token, table, nbest=5) if table else [text.brahmic_romanize(token)]
    mapped = snapper.snap(guesses[0], guesses) if snapper else guesses[0]
    cache[token] = mapped
    return mapped


def romanize_name(
    name: str,
    dictionary: dict[str, str],
    table: dict[str, dict[str, float]],
    snapper: VocabSnapper | None,
    cache: dict[str, str],
) -> str:
    if not name:
        return ""
    if not INDIC.search(name):
        return " ".join(latin_tokens(name))
    parts = [apply_token(tok, dictionary, table, snapper, cache) for tok in native_tokens(name)]
    return " ".join(p for p in parts if p)


def skeleton_key(name: str) -> str:
    toks = [text.phonetic_skeleton(tok) for tok in latin_tokens(name) if tok not in LEGAL and len(tok) >= 2]
    return " ".join(t for t in toks if t)
