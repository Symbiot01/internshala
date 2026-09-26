"""Text cleanup, romanization, and Ditto serialization.

Romanization uses anyascii: a fixed per-character table (ISC license). It never
looks anything up, so it stays inside the challenge's no-external-data rule.
"""

from __future__ import annotations

import re
import unicodedata

from anyascii import anyascii

_NULL = re.compile(r"<\s*null\s*>", re.IGNORECASE)
_REPEATED_PUNCT = re.compile(r"([^\w\s])\1+")
_SPACES = re.compile(r"\s+")
_EMPTY_SLOTS = re.compile(r"(?:\s*,)+\s*(?=,)|^\s*,\s*|\s*,\s*$")
_FIRST_NUMBER = re.compile(r"\d+[a-z]?(?:[/-]\d+)?")
_POSTAL = re.compile(r"(?<!\d)\d{5,6}(?!\d)")


def clean(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = _NULL.sub(" ", value)
    value = _REPEATED_PUNCT.sub(r"\1", value)
    value = _EMPTY_SLOTS.sub("", value)
    return _SPACES.sub(" ", value).strip().casefold()


def has_non_latin_letters(value: str) -> bool:
    """True for scripts such as Devanagari or Tamil; accented Latin stays False."""
    return any(char.isalpha() and ord(char) > 0x24F for char in value)


def romanize(value: str) -> str:
    """ASCII form used for retrieval; also strips accents (bláck -> black)."""
    return _SPACES.sub(" ", anyascii(value)).strip().lower()


def prepare_record(name: str, address: str) -> tuple[str, str, str, str]:
    """Return (name, address, name_roman, address_roman) for one record."""
    clean_name, clean_address = clean(name), clean(address)
    return clean_name, clean_address, romanize(clean_name), romanize(clean_address)


def tag_numbers(address: str) -> str:
    """Ditto domain-knowledge span typing: mark the house number and postal codes."""
    address = _POSTAL.sub(lambda match: f"[ZIP] {match.group(0)} [/ZIP]", address)
    match = _FIRST_NUMBER.search(address)
    if match and "[ZIP]" not in address[max(0, match.start() - 6) : match.start()]:
        address = f"{address[: match.start()]}[NUM] {match.group(0)} [/NUM]{address[match.end():]}"
    return address


def _show(original: str, roman: str) -> str:
    """Show both scripts only when the original is not Latin."""
    return f"{original} ({roman})" if has_non_latin_letters(original) and roman else original


def summarize_address(address: str, idf: dict[str, float], keep_words: int) -> str:
    """Ditto-style TF-IDF summary: keep the highest-IDF tokens in their original order."""
    tokens = address.split()
    if len(tokens) <= keep_words:
        return address
    # Unknown tokens are treated as rare (high IDF) so Indic/OOV words are kept.
    scored = [(idf.get(token, 8.0), index) for index, token in enumerate(tokens)]
    keep = {index for _, index in sorted(scored, reverse=True)[:keep_words]}
    return " ".join(token for index, token in enumerate(tokens) if index in keep)


def serialize(
    name: str,
    address: str,
    name_roman: str,
    address_roman: str,
    country: str,
    dk_tags: bool,
    include_country: bool = True,
    extras: dict[str, str] | None = None,
) -> str:
    """Ditto: [COL] attr [VAL] value for each attribute. Blank extras are omitted (MAR)."""
    shown_address = _show(address, address_roman)
    if dk_tags:
        shown_address = tag_numbers(shown_address)
    parts = [
        f"[COL] name [VAL] {_show(name, name_roman)}",
        f"[COL] address [VAL] {shown_address}",
    ]
    if extras:
        for key in ("state", "city", "house"):
            value = (extras.get(key) or "").strip()
            if value:
                parts.append(f"[COL] {key} [VAL] {value}")
    if include_country and country:
        parts.append(f"[COL] country [VAL] {country.lower()}")
    return " ".join(parts)


# --- Learned / static normalization (Phase 2) --------------------------------

TOKEN = re.compile(r"[a-z0-9]+")
INDIC_CHAR = re.compile(r"[\u0900-\u0d7f]")
DOMAIN = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9.-]*?)\.(?:com|in|net|org|co|biz|info|fr)(?:\b|/|$)"
)
LEGAL = frozenset(
    "private limited pvt ltd llc inc co company corp corporation the and pllc llp "
    "services service group enterprises industries international india l p c pc "
    "sas sarl sasu eurl sa sci snc sarlu".split()
)
EXPAND = {
    "pvt": "private", "ltd": "limited", "llc": "llc", "inc": "inc", "corp": "corporation",
    "co": "company", "bros": "brothers", "ent": "enterprises", "ind": "industries",
    "svc": "services", "trd": "trading", "assoc": "associates", "tech": "technology",
    "soln": "solutions", "mgmt": "management", "intl": "international", "dept": "department",
    "rd": "road", "st": "street", "ave": "avenue", "plz": "plaza", "blvd": "boulevard",
    "dr": "drive", "cir": "circle", "ct": "court", "ln": "lane", "apt": "apartment",
    "bldg": "building", "fl": "floor", "no": "no", "ngr": "nagar", "ctr": "center",
    "praivet": "private", "praibhet": "private", "piraivet": "private", "praivrr": "private",
    "limitet": "limited", "limirrd": "limited", "mited": "limited", "limtid": "limited",
}
STREET_STOP = frozenset(
    "no h flat plot door house unit ste suite floor the shop office room block "
    "st nd rd th street road avenue drive lane court circle plaza boulevard "
    "rue avenue boulevard place allee allée impasse chemin quai bd av r n".split()
)
FRENCH_STREET = {
    "r": "rue", "r.": "rue", "rue": "rue", "bd": "boulevard", "blvd": "boulevard",
    "av": "avenue", "av.": "avenue", "ave": "avenue", "pl": "place", "allee": "allee",
    "allée": "allee", "n": "no", "n°": "no", "no": "no",
}

_SCRIPT_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u",
    0x0A: "u", 0x0B: "r", 0x0C: "l", 0x0D: "e", 0x0E: "e",
    0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "au", 0x14: "a",
}
_SIGNS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
    0x43: "r", 0x44: "r", 0x45: "e", 0x46: "e", 0x47: "e",
    0x48: "ai", 0x49: "o", 0x4A: "au", 0x4B: "o", 0x4C: "l",
}
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "c", 0x1B: "ch", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh",
    0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
    0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "s",
    0x38: "s", 0x39: "h",
}
_MISC = {0x01: "", 0x02: "n", 0x03: "h"}
_VIRAMA, _NUKTA = 0x4D, 0x3C

SEED_ALIASES = {
    "mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka",
    "tn": "tamil nadu", "wb": "west bengal", "gj": "gujarat", "tg": "telangana",
    "hr": "haryana", "rj": "rajasthan", "kl": "kerala", "br": "bihar",
    "mp": "madhya pradesh", "ap": "andhra pradesh", "or": "odisha", "od": "odisha",
    "pb": "punjab", "orissa": "odisha", "keralam": "kerala", "poona": "pune",
    "greater bombay": "mumbai", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "bangalore": "bengaluru", "new delhi": "delhi",
    "hyderabad city region": "hyderabad", "k.v.rangareddy": "rangareddy",
    "tx": "texas", "oh": "ohio", "ny": "new york", "nc": "north carolina",
    "il": "illinois", "va": "virginia", "ma": "massachusetts", "az": "arizona",
    "in": "indiana", "wa": "washington", "md": "maryland", "ca": "california",
    "al": "alabama", "wi": "wisconsin", "mn": "minnesota", "ky": "kentucky",
    "ar": "arkansas", "mo": "missouri", "ok": "oklahoma", "pa": "pennsylvania",
    "ga": "georgia", "fl": "florida", "mi": "michigan", "nj": "new jersey",
    "hauts-de-france": "hauts-de-france", "nouvelle-aquitaine": "nouvelle-aquitaine",
    "pays de la loire": "pays de la loire", "nord": "nord", "gironde": "gironde",
    "loire-atlantique": "loire-atlantique", "pas-de-calais": "pas-de-calais",
}


def brahmic_romanize(value: str) -> str:
    """Fixed Unicode-block offset table over the nine Brahmic scripts. No lookup."""
    out: list[str] = []
    for char in value:
        code = ord(char)
        mapped: str | None = None
        for base in _SCRIPT_BASES:
            if base <= code < base + 0x80:
                offset = code - base
                if 0x66 <= offset <= 0x6F:
                    mapped = str(offset - 0x66)
                elif offset in _VOWELS:
                    mapped = _VOWELS[offset]
                elif offset in _CONSONANTS:
                    mapped = _CONSONANTS[offset]
                elif offset in _SIGNS:
                    mapped = _SIGNS[offset]
                elif offset in _MISC:
                    mapped = _MISC[offset]
                elif offset in (_VIRAMA, _NUKTA):
                    mapped = ""
                break
        out.append(char if mapped is None else mapped)
    return "".join(out)


def tokens(value: str) -> list[str]:
    return TOKEN.findall((value or "").lower())


def content_tokens(value: str) -> list[str]:
    return [token for token in tokens(value) if token not in LEGAL and len(token) >= 2]


def is_domain_name(name: str) -> bool:
    return bool(DOMAIN.search((name or "").lower().replace(" ", "")))


def domain_stem(name: str) -> str:
    match = DOMAIN.search((name or "").lower().replace(" ", ""))
    return match.group(1).replace("-", "") if match else ""


def phonetic_skeleton(word: str) -> str:
    word = (word or "").lower()
    for src, dst in (
        ("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"),
        ("dh", "d"), ("sh", "s"), ("ch", "c"), ("jh", "j"), ("w", "v"),
        ("z", "j"), ("q", "k"), ("x", "ks"), ("y", "i"),
    ):
        word = word.replace(src, dst)
    body = word[:1] + re.sub(r"[aeiou]", "", word[1:])
    return re.sub(r"(.)\1+", r"\1", body)


def segment_concat(stem: str, unigrams: dict[str, int], total: int) -> list[str]:
    """Viterbi word-break of a concatenated domain using corpus unigram log-probs."""
    stem = re.sub(r"[^a-z0-9]", "", (stem or "").lower())
    if not stem:
        return []
    vocab = max(len(unigrams), 1)
    n = len(stem)
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)

    def cost(piece: str) -> float:
        count = unigrams.get(piece, 0)
        if count:
            import math
            return -math.log((count + 0.1) / (total + vocab))
        return 12.0 + 2.5 * len(piece)

    for end in range(1, n + 1):
        for start in range(max(0, end - 20), end):
            candidate = best[start] + cost(stem[start:end])
            if candidate < best[end]:
                best[end] = candidate
                back[end] = start
    parts: list[str] = []
    cursor = n
    while cursor > 0:
        parts.append(stem[back[cursor]:cursor])
        cursor = back[cursor]
    return list(reversed(parts))


def apply_indic_dict(value: str, dictionary: dict[str, str], unigrams: dict[str, int] | None = None) -> str:
    """Replace native-script tokens; fall back to nearest known, then Brahmic table."""
    if not value or not INDIC_CHAR.search(value):
        return value
    keys = list(dictionary)
    skeletons = {phonetic_skeleton(key): key for key in keys} if keys else {}
    out: list[str] = []
    for token in value.split():
        if not INDIC_CHAR.search(token):
            out.append(token)
            continue
        if token in dictionary:
            out.append(dictionary[token])
            continue
        nearest = _nearest_native(token, keys)
        if nearest:
            out.append(dictionary[nearest])
            continue
        skel = phonetic_skeleton(brahmic_romanize(token))
        if skel in skeletons:
            out.append(dictionary[skeletons[skel]])
            continue
        out.append(brahmic_romanize(token))
    return " ".join(out)


def _nearest_native(token: str, keys: list[str], limit: int = 4000) -> str | None:
    if not keys:
        return None
    best_key, best_dist = None, 3
    for key in keys[:limit]:
        if abs(len(key) - len(token)) > 2:
            continue
        dist = _edit_distance(token, key, best_dist)
        if dist < best_dist:
            best_dist, best_key = dist, key
            if dist <= 1:
                break
    return best_key if best_dist <= 2 else None


def _edit_distance(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        row_min = i
        for j, b in enumerate(right, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            sub = previous[j - 1] + (a != b)
            value = min(insert, delete, sub)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def expand_tokens(value: str) -> str:
    return " ".join(EXPAND.get(token, token) for token in tokens(value))


def alias_lookup(value: str, aliases: dict[str, str]) -> str:
    key = (value or "").strip().lower()
    if not key:
        return ""
    if key in aliases:
        return aliases[key]
    roman = romanize(key)
    return aliases.get(roman, roman or key)


def parse_address(address: str, aliases: dict[str, str]) -> dict[str, str]:
    """Split a free-text address into house, street_core, city, state. Missing = empty."""
    cleaned = clean(address)
    if not cleaned:
        return {"house": "", "street_core": "", "city": "", "state": ""}
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    state = alias_lookup(parts[-1], aliases) if parts else ""
    city = ""
    if len(parts) >= 2:
        city = alias_lookup(parts[-2], aliases)
    house_match = _FIRST_NUMBER.search(cleaned)
    house = house_match.group(0).lstrip("0") or "0" if house_match else ""
    street_core = ""
    if house_match:
        after = TOKEN.findall(cleaned[house_match.end():])
        for token in after:
            mapped = FRENCH_STREET.get(token, token)
            if mapped not in STREET_STOP and len(mapped) >= 3:
                street_core = mapped
                break
    if not street_core:
        for token in TOKEN.findall(parts[0] if parts else cleaned):
            mapped = FRENCH_STREET.get(token, token)
            if mapped not in STREET_STOP and not mapped.isdigit() and len(mapped) >= 3:
                street_core = mapped
                break
    return {"house": house, "street_core": street_core, "city": city, "state": state}


def is_nonce_name(name: str) -> bool:
    toks = content_tokens(name)
    if is_domain_name(name) or INDIC_CHAR.search(name or ""):
        return False
    if any(token in LEGAL for token in tokens(name)):
        return False
    return 1 <= len(toks) <= 2 and all(part.isalpha() and 4 <= len(part) <= 16 for part in toks)


def normalize_name(name: str, tables: dict) -> str:
    dictionary = tables.get("indic_dict") or {}
    unigrams = tables.get("unigrams") or {}
    total = int(tables.get("unigram_total") or sum(unigrams.values()) or 1)
    mapped = apply_indic_dict(name, dictionary)
    roman = romanize(mapped)
    if is_domain_name(name) or is_domain_name(roman):
        stem = domain_stem(name) or domain_stem(roman)
        parts = segment_concat(stem, unigrams, total)
        if parts:
            roman = " ".join(parts)
    return expand_tokens(roman)


def normalize_address(address: str, tables: dict) -> str:
    if not (address or "").strip():
        return ""
    dictionary = tables.get("indic_dict") or {}
    aliases = tables.get("aliases") or SEED_ALIASES
    mapped = apply_indic_dict(address, dictionary)
    roman = expand_tokens(romanize(mapped))
    pieces = []
    for part in roman.split(","):
        key = part.strip()
        pieces.append(alias_lookup(key, aliases) if key else "")
    parsed = parse_address(roman, aliases)
    extras = [parsed[key] for key in ("house", "street_core", "city", "state") if parsed[key]]
    return _SPACES.sub(" ", " ".join(p for p in pieces if p) + " " + " ".join(extras)).strip()


def enrich_record(name: str, address: str, name_roman: str, address_roman: str, tables: dict) -> dict[str, str]:
    aliases = tables.get("aliases") or SEED_ALIASES
    name_norm = normalize_name(name, tables)
    address_norm = normalize_address(address, tables)
    parsed = parse_address(address_norm or address_roman, aliases)
    return {
        "name_norm": name_norm,
        "address_norm": address_norm,
        "blank_address": "1" if not (address or "").strip() else "0",
        "nonce_like": "1" if is_nonce_name(name_norm or name_roman) else "0",
        "domain_like": "1" if is_domain_name(name) or is_domain_name(name_roman) else "0",
        **parsed,
    }


def ocr_perturb(value: str, rng) -> str:
    table = str.maketrans({"0": "o", "o": "0", "1": "l", "l": "1", "5": "s", "s": "5", "8": "b"})
    chars = list(value)
    for index, char in enumerate(chars):
        if char.lower() in "0o1l5s8b" and rng.random() < 0.15:
            chars[index] = char.translate(table)
    return "".join(chars)
