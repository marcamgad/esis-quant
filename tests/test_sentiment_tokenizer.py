from models.sentiment.tokenizer import ArabicFinancialTokenizer, preprocess


def test_preprocess_removes_diacritics():
    assert preprocess("السَّلَامُ عَلَيْكُمْ") == "السلام عليكم"


def test_preprocess_normalizes_alef_variants():
    assert preprocess("آحمد") == preprocess("أحمد") == preprocess("إحمد") == "احمد"


def test_preprocess_normalizes_teh_marbuta():
    assert preprocess("شركة") == "شركه"


def test_preprocess_removes_tatweel():
    assert preprocess("شـــركة") == "شركه"


def test_preprocess_normalizes_ya_variants():
    assert preprocess("علئ") == preprocess("على") == preprocess("علي") == "علي"


def test_preprocess_keeps_digits_and_percent_signs():
    # Both Arabic and ASCII percent/decimal forms are deliberately kept
    # (see tokenizer.py _KEEP_PATTERN docstring note).
    result = preprocess("بمعدل خصم 24.7% اليوم")
    assert "24.7%" in result


def test_preprocess_collapses_multiple_spaces_and_strips():
    assert preprocess("  شركة    كبيرة   ") == "شركه كبيره"


def test_bpe_train_merges_most_frequent_pair_first():
    # word 'اا' freq=3 (2 occurrences in first string + 1 in second),
    # word 'با' freq=1 -> pair ('ا','ا') count=3 vs ('ب','ا') count=1,
    # so the first (and only, since vocab_size caps it at exactly 1 merge)
    # merge learned must be ('ا','ا').
    corpus = ["اا اا", "اا", "با"]
    n_special = len(ArabicFinancialTokenizer().special_tokens)
    n_chars = 2  # ا, ب
    tok = ArabicFinancialTokenizer(vocab_size=n_special + n_chars + 1)
    tok.train_bpe(corpus)
    assert tok.merges == [("ا", "ا")]
    assert "اا" in tok.vocab


def test_bpe_train_stops_early_when_no_more_pairs_available():
    # Single-character words only -- no adjacent pairs exist to merge,
    # even with a huge target vocab_size, train_bpe must stop gracefully.
    tok = ArabicFinancialTokenizer(vocab_size=8000)
    tok.train_bpe(["ا ب ا ب"])
    assert tok.merges == []


def test_encode_structure_cls_sep_padding_and_attention_mask():
    tok = ArabicFinancialTokenizer(vocab_size=len(ArabicFinancialTokenizer().special_tokens) + 2 + 1)
    tok.train_bpe(["اا اا", "اا", "با"])
    enc = tok.encode("اا با", max_length=10)

    assert len(enc["input_ids"]) == 10
    assert len(enc["attention_mask"]) == 10
    assert enc["input_ids"][0] == tok.special_tokens["[CLS]"]

    sep_idx = enc["input_ids"].index(tok.special_tokens["[SEP]"])
    assert all(m == 1 for m in enc["attention_mask"][: sep_idx + 1])
    assert all(m == 0 for m in enc["attention_mask"][sep_idx + 1 :])
    assert all(t == tok.special_tokens["[PAD]"] for t in enc["input_ids"][sep_idx + 1 :])


def test_encode_truncates_long_text_and_keeps_sep_last():
    tok = ArabicFinancialTokenizer(vocab_size=len(ArabicFinancialTokenizer().special_tokens) + 2 + 1)
    tok.train_bpe(["اا اا", "اا", "با"])
    long_text = " ".join(["اا"] * 20)
    enc = tok.encode(long_text, max_length=8)
    assert len(enc["input_ids"]) == 8
    assert enc["input_ids"][-1] == tok.special_tokens["[SEP]"]
    assert enc["input_ids"][0] == tok.special_tokens["[CLS]"]


def test_encode_unknown_symbol_maps_to_unk():
    tok = ArabicFinancialTokenizer(vocab_size=len(ArabicFinancialTokenizer().special_tokens) + 2 + 1)
    tok.train_bpe(["اا اا", "اا", "با"])
    enc = tok.encode("جح", max_length=10)  # characters never seen in training corpus
    unk_id = tok.special_tokens["[UNK]"]
    # Everything between CLS and SEP should be UNK since ج/ح are out-of-vocab.
    sep_idx = enc["input_ids"].index(tok.special_tokens["[SEP]"])
    middle = enc["input_ids"][1:sep_idx]
    assert all(t == unk_id for t in middle)
    assert len(middle) > 0


def test_save_and_load_roundtrip(tmp_path):
    tok = ArabicFinancialTokenizer(vocab_size=len(ArabicFinancialTokenizer().special_tokens) + 2 + 1)
    tok.train_bpe(["اا اا", "اا", "با"])
    path = str(tmp_path / "tok.json")
    tok.save(path)

    loaded = ArabicFinancialTokenizer.load(path)
    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.encode("اا با") == tok.encode("اا با")
