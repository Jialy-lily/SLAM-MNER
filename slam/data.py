import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor, AutoTokenizer

TYPE2ID = {"PER": 0, "LOC": 1, "ORG": 2, "OTHER": 3, "MISC": 3}
ID2TYPE = {0: "PER", 1: "LOC", 2: "ORG", 3: "MISC"}
BOUNDARY2ID = {"O": 0, "B-ENT": 1, "I-ENT": 2}
ID2BOUNDARY = {v: k for k, v in BOUNDARY2ID.items()}


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(path: str, image_root: Optional[str]) -> str:
    if image_root is None or os.path.isabs(path):
        return path
    return os.path.join(image_root, path)


def build_boundary_word_labels(words: List[str], gold_entities: List[Dict[str, Any]]) -> List[int]:
    labels = [BOUNDARY2ID["O"]] * len(words)
    for ent in gold_entities:
        s = int(ent["token_start"])
        e = int(ent["token_end"])
        if 0 <= s < e <= len(words):
            labels[s] = BOUNDARY2ID["B-ENT"]
            for i in range(s + 1, e):
                labels[i] = BOUNDARY2ID["I-ENT"]
    return labels


def build_spans(gold_entities: List[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    spans = []
    for ent in gold_entities:
        s = int(ent["token_start"])
        e = int(ent["token_end"])
        t = TYPE2ID.get(ent["type"], TYPE2ID["OTHER"])
        if s < e:
            spans.append((s, e, t))
    return spans


def build_complete_word_token_maps(
    word_ids: List[Optional[int]], full_word_ids: List[Optional[int]]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Map only words whose complete subword sequence survives truncation."""
    retained_counts = Counter(wid for wid in word_ids if wid is not None)
    full_counts = Counter(wid for wid in full_word_ids if wid is not None)
    word_to_token_start = {}
    word_to_token_end = {}
    for token_idx, wid in enumerate(word_ids):
        if wid is None or retained_counts[wid] != full_counts[wid]:
            continue
        word_to_token_start.setdefault(wid, token_idx)
        word_to_token_end[wid] = token_idx
    return word_to_token_start, word_to_token_end


def build_limited_supervision_subset(samples: List[Dict[str, Any]], ratio: float, seed: int = 42) -> List[int]:
    rng = random.Random(seed)
    buckets: Dict[Tuple[int, str], List[int]] = {}
    for i, x in enumerate(samples):
        num_images = min(len(x.get("image_paths", [])), 4)
        types = sorted({e["type"] for e in x.get("gold_entities", [])})
        type_key = "-".join(types) if types else "NONE"
        key = (num_images, type_key)
        buckets.setdefault(key, []).append(i)
    chosen = []
    for ids in buckets.values():
        k = max(1, int(round(len(ids) * ratio)))
        chosen.extend(rng.sample(ids, min(k, len(ids))))
    return sorted(chosen)


class MNERConvertedDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        text_model_name: str = "xlm-roberta-large",
        vision_model_name: str = "openai/clip-vit-base-patch16",
        max_length: int = 128,
        max_images: int = 4,
        image_root: Optional[str] = None,
    ):
        self.samples = load_json(json_path)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(vision_model_name, use_fast=True)
        self.max_length = max_length
        self.max_images = max_images
        self.image_root = image_root

    def __len__(self):
        return len(self.samples)

    def _load_images(self, image_paths: List[str]):
        images = []
        valid_mask = []
        for p in image_paths[: self.max_images]:
            rp = resolve_image_path(p, self.image_root)
            try:
                img = Image.open(rp).convert("RGB")
                images.append(img)
                valid_mask.append(1)
            except Exception:
                images.append(Image.new("RGB", (224, 224), color=0))
                valid_mask.append(0)
        while len(images) < self.max_images:
            images.append(Image.new("RGB", (224, 224), color=0))
            valid_mask.append(0)
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return pixel_values, torch.tensor(valid_mask, dtype=torch.long)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        words = item["text"].split()
        gold_entities = item.get("gold_entities", [])

        boundary_word_labels = build_boundary_word_labels(words, gold_entities)
        spans = build_spans(gold_entities)

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        full_encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        word_to_token_start, word_to_token_end = build_complete_word_token_maps(
            word_ids, full_encoding.word_ids(batch_index=0)
        )
        token_boundary_labels = []
        token_first_mask = []
        prev_wid = None
        for token_idx, wid in enumerate(word_ids):
            if wid is None:
                token_boundary_labels.append(-100)
                token_first_mask.append(0)
                continue
            if wid != prev_wid:
                token_boundary_labels.append(boundary_word_labels[wid])
                token_first_mask.append(1)
            else:
                token_boundary_labels.append(-100)
                token_first_mask.append(0)
            prev_wid = wid

        pixel_values, image_mask = self._load_images(item.get("image_paths", []))

        return {
            "id": item.get("id", str(idx)),
            "text": item["text"],
            "words": words,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "boundary_labels": torch.tensor(token_boundary_labels, dtype=torch.long),
            "token_first_mask": torch.tensor(token_first_mask, dtype=torch.long),
            "word_to_token_start": word_to_token_start,
            "word_to_token_end": word_to_token_end,
            "spans": spans,
            "pixel_values": pixel_values,
            "image_mask": image_mask,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "words": [x["words"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "boundary_labels": torch.stack([x["boundary_labels"] for x in batch], dim=0),
        "token_first_mask": torch.stack([x["token_first_mask"] for x in batch], dim=0),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch], dim=0),
        "image_mask": torch.stack([x["image_mask"] for x in batch], dim=0),
        "spans": [x["spans"] for x in batch],
        "word_to_token_start": [x["word_to_token_start"] for x in batch],
        "word_to_token_end": [x["word_to_token_end"] for x in batch],
    }


def make_loader(dataset, batch_size, shuffle=False, subset_indices: Optional[List[int]] = None):
    ds = Subset(dataset, subset_indices) if subset_indices is not None else dataset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
