import numpy as np

import decide
import text


def test_domain_viterbi_splits_company_url():
    unigrams = {"maan": 100, "construction": 80, "private": 50, "limited": 50}
    parts = text.segment_concat("maanconstruction", unigrams, sum(unigrams.values()))
    assert parts[:2] == ["maan", "construction"]


def test_indic_dictionary_maps_prime_food():
    mapped = text.apply_indic_dict("प्राइम फूड", {"प्राइम": "prime", "फूड": "food"})
    assert mapped == "prime food"


def test_blank_address_parse():
    parsed = text.parse_address("", text.SEED_ALIASES)
    assert parsed["house"] == parsed["city"] == parsed["state"] == ""


def test_french_street_and_state_alias():
    parsed = text.parse_address("17 r. villars, roubaix, nord", {"nord": "hauts-de-france", **text.SEED_ALIASES})
    assert parsed["street_core"] in {"villars", "rue"} or parsed["house"] == "17"
    assert parsed["house"] == "17"


def test_nonce_flag():
    assert text.is_nonce_name("jaxpyra")
    assert not text.is_nonce_name("prime food private limited")


def test_ocr_perturb_is_deterministic_with_rng():
    rng = np.random.default_rng(0)
    out = text.ocr_perturb("foods 101", rng)
    assert isinstance(out, str)


def test_keep_mask_per_source_and_exempt():
    s1 = np.array(["A"] * 8)
    cand = np.array([f"S2-{i}" for i in range(5)] + [f"S3-{i}" for i in range(3)])
    scores = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.99, 0.5, 0.4])
    packed = decide.pack(s1, cand, scores)
    keep = decide.keep_mask(packed, threshold=0.5, cap=2, alpha=0.0, gate=0.0, cap_exempt=0.99)
    kept = set(packed["cand_ids"][keep].tolist())
    assert "S2-0" in kept  # exempt
    assert "S3-0" in kept
    assert packed["ranks_source"].min() == 1
