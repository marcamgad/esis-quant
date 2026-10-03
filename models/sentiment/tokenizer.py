"""
Arabic financial BPE tokenizer, implemented from scratch (no HuggingFace
tokenizers / sentencepiece). Byte-Pair Encoding per Sennrich et al. 2016:
starts from a character-level vocabulary and greedily merges the most
frequent adjacent symbol pair, vocab_size - len(special_tokens) - n_chars
times, recording each merge in the order learned so encoding can replay
the same merges deterministically.
"""

import json
import logging
import re
from collections import Counter

import config

logger = logging.getLogger("esis_quant.sentiment.tokenizer")

_DIACRITICS_PATTERN = re.compile(r"[ً-ٟ]")
_ALEF_VARIANTS_PATTERN = re.compile(r"[آأإ]")  # آ أ إ -> ا
_TEH_MARBUTA_PATTERN = re.compile(r"ة")  # ة -> ه
_WAW_VARIANT_PATTERN = re.compile(r"ؤ")  # ؤ -> و
_TATWEEL_PATTERN = re.compile(r"ـ")  # ـ (removed, not replaced)
_YA_VARIANTS_PATTERN = re.compile(r"[ئى]")  # ئ ى -> ي

# Arabic letters (post-normalization range) + Arabic-Indic and ASCII
# digits + Arabic percent/decimal separators + ASCII '%'/'.' (kept in
# addition to the spec's Arabic-only decimal/percent chars: real scraped
# headlines use ASCII '%'/'.' for numbers like "24.7%", and destroying
# that would throw away real information a T-bill-rate/return-magnitude
# signal might need -- a deliberate, disclosed deviation from a strict
# reading of "digits, percent (٪), decimal (٫)", not an oversight).
_KEEP_PATTERN = re.compile(r"[^ء-ي٠-٩٪٫%.0-9\s]")
_MULTISPACE_PATTERN = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """Apply the 9-step Arabic normalization pipeline, in order:
    1. remove diacritics (harakat)
    2. normalize alef variants (آأإ) -> ا
    3. normalize teh marbuta (ة) -> ه
    4. normalize waw variant (ؤ) -> و
    5. remove tatweel (ـ)
    6. normalize ya variants (ئى) -> ي
    7. remove non-Arabic except digits/percent/decimal (see _KEEP_PATTERN note)
    8. collapse multiple spaces
    9. strip leading/trailing whitespace
    """
    text = _DIACRITICS_PATTERN.sub("", text)
    text = _ALEF_VARIANTS_PATTERN.sub("ا", text)
    text = _TEH_MARBUTA_PATTERN.sub("ه", text)
    text = _WAW_VARIANT_PATTERN.sub("و", text)
    text = _TATWEEL_PATTERN.sub("", text)
    text = _YA_VARIANTS_PATTERN.sub("ي", text)
    text = _KEEP_PATTERN.sub("", text)
    text = _MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def _merge_pair_in_word(symbols: list, a: str, b: str) -> list:
    """Replace every adjacent (a, b) occurrence in `symbols` with the
    merged token a+b, scanning left to right (non-overlapping).
    """
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class ArabicFinancialTokenizer:
    def __init__(self, vocab_size: int = None):
        self.vocab_size = vocab_size or config.SENTIMENT_VOCAB_SIZE
        self.special_tokens = dict(config.SENTIMENT_SPECIAL_TOKENS)
        self.vocab = dict(self.special_tokens)  # token string -> id
        self.merges = []  # ordered list of (a, b) tuples, in learned order
        self._merge_ranks = {}

    def train_bpe(self, corpus: list):
        """Train BPE merges on `corpus` (list of raw Arabic strings).
        Each string is preprocessed first, then split on whitespace into
        "words"; BPE operates within each word (merges never cross a
        whitespace boundary), matching standard BPE practice.
        """
        word_freq = Counter()
        for text in corpus:
            for word in preprocess(text).split():
                word_freq[word] += 1

        if not word_freq:
            raise ValueError("train_bpe: corpus produced zero words after preprocessing")

        splits = {word: list(word) for word in word_freq}
        charset = sorted({ch for word in word_freq for ch in word})

        vocab = dict(self.special_tokens)
        next_id = len(vocab)
        for ch in charset:
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1

        merges = []
        while len(vocab) < self.vocab_size:
            pair_freq = Counter()
            for word, freq in word_freq.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pair_freq[(symbols[i], symbols[i + 1])] += freq
            if not pair_freq:
                logger.info("train_bpe: no more adjacent pairs to merge, stopping at vocab size %d", len(vocab))
                break

            best_pair = max(pair_freq.items(), key=lambda kv: kv[1])[0]
            a, b = best_pair
            merged_token = a + b
            for word in splits:
                splits[word] = _merge_pair_in_word(splits[word], a, b)
            merges.append(best_pair)
            vocab[merged_token] = next_id
            next_id += 1

        self.vocab = vocab
        self.merges = merges
        self._merge_ranks = {pair: i for i, pair in enumerate(merges)}
        logger.info("train_bpe: learned %d merges, final vocab size %d", len(merges), len(vocab))

    def _apply_merges(self, symbols: list) -> list:
        """Encode-time BPE: repeatedly merge whichever adjacent pair
        present in `symbols` has the LOWEST rank (earliest learned),
        matching how the merges were prioritized during training.
        """
        symbols = list(symbols)
        while len(symbols) > 1:
            pairs_present = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self._merge_ranks[p], i) for i, p in enumerate(pairs_present) if p in self._merge_ranks]
            if not ranked:
                break
            _, best_idx = min(ranked)
            a, b = symbols[best_idx], symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [a + b] + symbols[best_idx + 2:]
        return symbols

    def encode(self, text: str, max_length: int = None) -> dict:
        """Preprocess -> BPE-tokenize -> add [CLS]/[SEP] -> pad/truncate.
        Returns {"input_ids": list[int] length max_length, "attention_mask": list[int] length max_length}.
        """
        max_length = max_length or config.SENTIMENT_MAX_SEQ_LEN
        cls_id = self.special_tokens["[CLS]"]
        sep_id = self.special_tokens["[SEP]"]
        unk_id = self.special_tokens["[UNK]"]
        pad_id = self.special_tokens["[PAD]"]

        text = preprocess(text)
        token_ids = [cls_id]
        for word in text.split():
            for symbol in self._apply_merges(list(word)):
                token_ids.append(self.vocab.get(symbol, unk_id))
        token_ids.append(sep_id)

        if len(token_ids) > max_length:
            token_ids = token_ids[: max_length - 1] + [sep_id]

        attention_mask = [1] * len(token_ids)
        pad_len = max_length - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges, "vocab_size": self.vocab_size}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ArabicFinancialTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.vocab = data["vocab"]
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok._merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        return tok
