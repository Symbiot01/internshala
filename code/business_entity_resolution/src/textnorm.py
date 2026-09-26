"""String cleanup using only the provided data and deterministic character rules.

No external gazetteers, geocoders, or business registries. The Brahmic
transliteration below is a fixed offset table over parallel Unicode blocks
(Devanagari U+0900, Bengali U+0980, Gurmukhi U+0A00, Gujarati U+0A80,
Oriya U+0B00, Tamil U+0B80, Telugu U+0C00, Kannada U+0C80, Malayalam U+0D00),
which share the same code-point layout. It performs no lookup.
"""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.[a-z]{2,}(?:\.[a-z]{2,})?$")

# Abbreviation equivalences. The core set comes from the problem statement;
# the extended set was mined from co-occurring tokens across gold training
# pairs (see scripts/mine_abbrev.py). Applied after lowercasing.
_REPLACEMENTS = (
    (re.compile(r"\bcorporation\b"), "corp"),
    (re.compile(r"\bincorporated\b"), "inc"),
    (re.compile(r"\blimited\b"), "ltd"),
    (re.compile(r"\bprivate\b"), "pvt"),
    (re.compile(r"\bcompany\b"), "co"),
    (re.compile(r"\bbrothers\b"), "bros"),
    (re.compile(r"\benterprises\b"), "ent"),
    (re.compile(r"\benterprise\b"), "ent"),
    (re.compile(r"\bindustries\b"), "ind"),
    (re.compile(r"\bindustry\b"), "ind"),
    (re.compile(r"\bservices\b"), "svc"),
    (re.compile(r"\bservice\b"), "svc"),
    (re.compile(r"\btraders\b"), "trd"),
    (re.compile(r"\btrading\b"), "trd"),
    (re.compile(r"\bassociates\b"), "assoc"),
    (re.compile(r"\btechnologies\b"), "tech"),
    (re.compile(r"\btechnology\b"), "tech"),
    (re.compile(r"\bsolutions\b"), "soln"),
    (re.compile(r"\bmanagement\b"), "mgmt"),
    (re.compile(r"\binternational\b"), "intl"),
    (re.compile(r"\bdepartment\b"), "dept"),
    (re.compile(r"\broad\b"), "rd"),
    (re.compile(r"\bstreet\b"), "st"),
    (re.compile(r"\bavenue\b"), "ave"),
    (re.compile(r"\bplaza\b"), "plz"),
    (re.compile(r"\bboulevard\b"), "blvd"),
    (re.compile(r"\bdrive\b"), "dr"),
    (re.compile(r"\bcircle\b"), "cir"),
    (re.compile(r"\bcourt\b"), "ct"),
    (re.compile(r"\blane\b"), "ln"),
    (re.compile(r"\bapartment\b"), "apt"),
    (re.compile(r"\bbuilding\b"), "bldg"),
    (re.compile(r"\bfloor\b"), "fl"),
    (re.compile(r"\bnumber\b"), "no"),
    (re.compile(r"\bshop\b"), "shp"),
    (re.compile(r"\bmarket\b"), "mkt"),
    (re.compile(r"\bnagar\b"), "ngr"),
    (re.compile(r"\btrail\b"), "trl"),
    (re.compile(r"\bblue\b"), "blu"),
    (re.compile(r"\bcenter\b"), "ctr"),
    (re.compile(r"\bcentre\b"), "ctr"),
    (re.compile(r"\bsainte?\b"), "st"),
    (re.compile(r"\brue\b"), "st"),
    # Transliteration-spelling variants mined from gold training pairs.
    (re.compile(r"\bpraivet\b"), "pvt"),
    (re.compile(r"\bpraibhet\b"), "pvt"),
    (re.compile(r"\bpiraivet\b"), "pvt"),
    (re.compile(r"\bpraivrr\b"), "pvt"),
    (re.compile(r"\blimitet\b"), "ltd"),
    (re.compile(r"\blimirrd\b"), "ltd"),
    (re.compile(r"\bmited\b"), "ltd"),
    (re.compile(r"\blimtid\b"), "ltd"),
    (re.compile(r"\binphra\b"), "infra"),
    (re.compile(r"\binphotek\b"), "infotech"),
    (re.compile(r"\btreding\b"), "trading"),
    (re.compile(r"\bbijnes\b"), "business"),
    (re.compile(r"\bhospitailiti\b"), "hospitality"),
    (re.compile(r"\bintrneshnl\b"), "international"),
    (re.compile(r"\bglobl\b"), "global"),
    (re.compile(r"\bdrim\b"), "dream"),
    (re.compile(r"\baiti\b"), "it"),
    (re.compile(r"\belelpi\b"), "llp"),
    (re.compile(r"\bmharastr\b"), "maharashtra"),
    (re.compile(r"\bprdesh\b"), "pradesh"),
    (re.compile(r"\bgujrat\b"), "gujarat"),
    (re.compile(r"\btelngan\b"), "telangana"),
    (re.compile(r"\bkrnatk\b"), "karnataka"),
    (re.compile(r"\bhriyana\b"), "haryana"),
    (re.compile(r"\brajsthan\b"), "rajasthan"),
    (re.compile(r"\bmdhy\b"), "madhya"),
    (re.compile(r"\buttr\b"), "uttar"),
    (re.compile(r"\bkeralam\b"), "kerala"),
    (re.compile(r"\bkerln\b"), "kerala"),
    (re.compile(r"\btmilnatu\b"), "tamil nadu"),
)

# US state abbreviations, applied to addresses only: several codes collide
# with ordinary words (in, or, me, la, de, co, ms, mt, md, hi) and are excluded.
_STATE_REPLACEMENTS = (
    (re.compile(r"\btx\b"), "texas"),
    (re.compile(r"\boh\b"), "ohio"),
    (re.compile(r"\but\b"), "utah"),
    (re.compile(r"\bia\b"), "iowa"),
    (re.compile(r"\bks\b"), "kansas"),
    (re.compile(r"\bky\b"), "kentucky"),
    (re.compile(r"\bnv\b"), "nevada"),
    (re.compile(r"\bnj\b"), "new jersey"),
    (re.compile(r"\bnm\b"), "new mexico"),
    (re.compile(r"\bny\b"), "new york"),
    (re.compile(r"\bnc\b"), "north carolina"),
    (re.compile(r"\bnd\b"), "north dakota"),
    (re.compile(r"\bsc\b"), "south carolina"),
    (re.compile(r"\bsd\b"), "south dakota"),
    (re.compile(r"\btn\b"), "tennessee"),
    (re.compile(r"\bvt\b"), "vermont"),
    (re.compile(r"\bwi\b"), "wisconsin"),
    (re.compile(r"\bwy\b"), "wyoming"),
    (re.compile(r"\bnh\b"), "new hampshire"),
    (re.compile(r"\bri\b"), "rhode island"),
    (re.compile(r"\bwv\b"), "west virginia"),
    (re.compile(r"\baz\b"), "arizona"),
    (re.compile(r"\bak\b"), "alaska"),
    (re.compile(r"\bct\b"), "connecticut"),
    (re.compile(r"\bfl\b"), "florida"),
    (re.compile(r"\bga\b"), "georgia"),
    (re.compile(r"\bpa\b"), "pennsylvania"),
    (re.compile(r"\bne\b"), "nebraska"),
    (re.compile(r"\bid\b"), "idaho"),
    (re.compile(r"\bok\b"), "oklahoma"),
    (re.compile(r"\bmn\b"), "minnesota"),
    (re.compile(r"\bar\b"), "arkansas"),
    (re.compile(r"\bmo\b"), "missouri"),
    (re.compile(r"\bva\b"), "virginia"),
    (re.compile(r"\bca\b"), "california"),
)

_STOP = frozenset(
    {
        "the",
        "and",
        "of",
        "near",
        "at",
        "in",
        "for",
        "a",
        "an",
        "to",
        "by",
        "opp",
        "opposite",
        "behind",
        "null",
        "www",
        "com",
        "http",
        "https",
    }
)

_ORDINALS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th",
}


def _compiled_to_map(pairs: tuple[tuple[re.Pattern[str], str], ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pattern, replacement in pairs:
        token = pattern.pattern.replace(r"\b", "")
        if token == "sainte?":
            mapping["saint"] = replacement
            mapping["sainte"] = replacement
        else:
            mapping[token] = replacement
    return mapping


_ABBREV = _compiled_to_map(_REPLACEMENTS)
_STATES = _compiled_to_map(_STATE_REPLACEMENTS)

# Brahmic script base offsets sharing one parallel layout.
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

_MISC = {0x01: "", 0x02: "n", 0x03: "h"}  # chandrabindu, anusvara, visarga
_VIRAMA = 0x4D
_NUKTA = 0x3C


def transliterate(text: str) -> str:
    """Map Brahmic-script characters to Latin with a fixed offset table."""
    out: list[str] = []
    for char in text:
        code = ord(char)
        mapped: str | None = None
        for base in _SCRIPT_BASES:
            if base <= code < base + 0x80:
                offset = code - base
                if 0x66 <= offset <= 0x6F:  # script digits
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


def _domain_stem(name: str) -> str:
    match = _DOMAIN.match(name.strip().lower())
    return match.group(1).replace("-", " ") if match else ""


def _strip_diacritics(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def _needs_transliteration(text: str) -> bool:
    return any(0x0900 <= ord(char) <= 0x0D7F for char in text)


def _remap(words: list[str], table: dict[str, str]) -> list[str]:
    out: list[str] = []
    for word in words:
        replacement = table.get(word)
        if replacement is None:
            out.append(word)
        else:
            out.extend(replacement.split(" "))
    return out


def _finish(words: list[str]) -> list[str]:
    cleaned: list[str] = []
    for word in words:
        if word.isdigit():
            word = word.lstrip("0") or "0"
        if word:
            cleaned.append(word)
    return cleaned


def normalize(text: str, is_address: bool = False) -> str:
    raw = text or ""
    if any(ord(char) > 127 for char in raw):
        value = unicodedata.normalize("NFKC", raw)
        if _needs_transliteration(value):
            value = transliterate(value)
        value = _strip_diacritics(value)
    else:
        value = raw
    value = value.casefold().replace("&", " and ")
    if "." in value:
        stem = _domain_stem(value)
        if stem:
            value = f"{value} {stem}"
    words = _NON_ALNUM.sub(" ", value).split()
    words = [_ORDINALS.get(word, word) for word in words]
    if is_address:
        words = _remap(words, _STATES)
    words = _remap(words, _ABBREV)
    return " ".join(_finish(words))


def _norm_name(text: str) -> str:
    return normalize(text, is_address=False)


def _norm_address(text: str) -> str:
    return normalize(text, is_address=True)


def content_tokens(normalized: str) -> list[str]:
    return [token for token in normalized.split() if len(token) >= 2 and token not in _STOP]


def tokens(text: str, is_address: bool = False) -> list[str]:
    return [
        token
        for token in normalize(text, is_address=is_address).split()
        if len(token) >= 2 and token not in _STOP
    ]
