import translit


def test_positional_maps_prime_food():
    pairs = [("prime food private limited", "प्राइम फूड प्राइवेट लिमिटेड")] * 3
    dictionary = translit.mine_positional(pairs)
    assert dictionary["प्राइम"] == "prime"
    assert dictionary["फूड"] == "food"


def test_positional_keeps_legal_tokens():
    pairs = [("green eastern producer private limited", "గ్రీన్ ఈస్టర్న్ ప్రొడ్యూసర్ ప్రైవేట్ లిమిటెడ్")] * 3
    dictionary = translit.mine_positional(pairs)
    assert dictionary.get("ప్రైవేట్") == "private"
    assert dictionary.get("లిమిటెడ్") == "limited"


def test_golden_dictionary_lookups():
    pairs = [
        ("prime food private limited", "प्राइम फूड प्राइवेट लिमिटेड"),
        ("prime food private limited", "ప్రైమ్ ఫుడ్ ప్రైవేట్ లిమిటెడ్"),
        ("prime food private limited", "பிரைம் புட் பிரைவேட் லிமிடெட்"),
        ("prime food private limited", "প্রাইম ফুড প্রাইভেট লিমিটেড"),
    ] * 3
    dictionary = translit.mine_positional(pairs)
    assert translit.romanize_name("प्राइम फूड", dictionary, {}, None, {}) == "prime food"
    assert dictionary.get("ప్రైవేట్") == "private" or dictionary.get("பிரைவேட்") == "private"
    assert dictionary.get("লিমিটেড") == "limited"


def test_akshara_split_nonempty():
    parts = translit.akshara_split("प्राइवेट")
    assert len(parts) >= 2
    assert "".join(parts) == "प्राइवेट"


def test_unseen_snaps_to_vocab():
    dictionary = translit.mine_positional([("private limited", "प्राइवेट लिमिटेड")] * 4)
    table = translit.train_m2m([(k, v) for k, v in dictionary.items()] * 4)
    snapper = translit.VocabSnapper(["private", "limited", "prime", "foods", "consulting"])
    mapped = translit.apply_token("प्राइवेट", {}, table, snapper, {})
    assert mapped == "private"
