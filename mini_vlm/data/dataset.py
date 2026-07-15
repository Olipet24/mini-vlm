'''After build_dataset.py, we turn this into a pytorch dataset using the following script.'''

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mini_vlm.text import Tokenizer

MAX_QUESTION_LEN = 16


class VQAFeatureDataset(Dataset):
    """Reads a processed VQA split (train/val/test .json) and joins each
    question against its pre-cached, frozen-encoder feature tensor."""

    def __init__(
        self,
        split_path: Path,
        features_dir: Path,
        tokenizer: Tokenizer,
        answer_vocab: list,
        max_question_len: int = MAX_QUESTION_LEN,
    ) -> None:
        self.records = json.load(open(split_path))
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.answer_vocab = answer_vocab
        self.max_question_len = max_question_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        feat_path = self.features_dir / (Path(r["image_filename"]).stem + ".pt")
        features = torch.load(feat_path).float()  # [576, 7, 7]
        question_ids = torch.tensor(
            self.tokenizer.encode(r["question"], self.max_question_len), dtype=torch.long
        )
        answer_idx = torch.tensor(r["answer_idx"], dtype=torch.long)
        return {
            "features": features,
            "question_ids": question_ids,
            "answer_idx": answer_idx,
            "question": r["question"],
            "answer": r["answer"],
            "image_filename": r["image_filename"],
        }


def collate(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "question_ids": torch.stack([b["question_ids"] for b in batch]),
        "answer_idx": torch.stack([b["answer_idx"] for b in batch]),
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "image_filename": [b["image_filename"] for b in batch],
    }
