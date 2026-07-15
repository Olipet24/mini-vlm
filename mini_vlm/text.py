"""Minimal word-level tokenizer for VQA questions. Built from scratch
(no external tokenizer dependency) since questions are short and the
vocabulary only needs to cover this project's train split."""
import json
import re
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"


def _words(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class Tokenizer:
    def __init__(self, word2idx: dict) -> None:
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    @classmethod
    def build(cls, questions, min_freq: int = 1) -> "Tokenizer":
        from collections import Counter

        counts = Counter()
        for q in questions:
            counts.update(_words(q))
        vocab = [PAD, UNK] + sorted(w for w, c in counts.items() if c >= min_freq)
        return cls({w: i for i, w in enumerate(vocab)})

    def encode(self, text: str, max_len: int) -> list:
        ids = [self.word2idx.get(w, self.word2idx[UNK]) for w in _words(text)]
        ids = ids[:max_len]
        ids += [self.word2idx[PAD]] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: Path) -> None:
        json.dump(self.word2idx, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        return cls(json.load(open(path)))
