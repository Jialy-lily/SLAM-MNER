import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch

from .data import BOUNDARY2ID, ID2TYPE


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    out = dict(batch)
    for k in ["input_ids", "attention_mask", "boundary_labels", "token_first_mask", "pixel_values", "image_mask"]:
        out[k] = batch[k].to(device)
    return out


def decode_word_boundary_preds(logits: torch.Tensor, token_first_mask: torch.Tensor, words_batch: List[List[str]]):
    pred_token = logits.argmax(dim=-1).detach().cpu()
    token_first_mask = token_first_mask.detach().cpu()
    word_preds = []
    for b in range(pred_token.size(0)):
        preds = []
        for t in range(pred_token.size(1)):
            if token_first_mask[b, t].item() == 1:
                preds.append(int(pred_token[b, t].item()))
        preds = preds[: len(words_batch[b])]
        word_preds.append(preds)
    return word_preds


def decode_spans_from_boundary(word_boundary_preds: List[List[int]]) -> List[List[Tuple[int, int]]]:
    all_spans = []
    for preds in word_boundary_preds:
        spans = []
        i = 0
        n = len(preds)
        while i < n:
            if preds[i] == BOUNDARY2ID["B-ENT"]:
                s = i
                i += 1
                while i < n and preds[i] == BOUNDARY2ID["I-ENT"]:
                    i += 1
                spans.append((s, i))
            else:
                i += 1
        all_spans.append(spans)
    return all_spans


def classify_predicted_spans(model, model_out: Dict[str, Any], batch: Dict[str, Any], pred_spans: List[List[Tuple[int, int]]]):
    text_tokens = model_out["text_tokens"]
    slots = model_out["slots"]
    outputs = []
    for b, spans in enumerate(pred_spans):
        curr = []
        if not spans:
            outputs.append(curr)
            continue
        reps = []
        valid_spans = []
        for span in spans:
            rep = model.get_span_rep(text_tokens[b], batch["word_to_token_start"][b], batch["word_to_token_end"][b], span)
            if rep is not None:
                reps.append(rep)
                valid_spans.append(span)
        if not reps:
            outputs.append(curr)
            continue
        reps = torch.stack(reps, dim=0)
        per_slots = slots[b].unsqueeze(0).expand(reps.size(0), -1, -1)
        type_logits, bind_attn, _, _ = model.classify_mentions(reps, per_slots)
        type_ids = type_logits.argmax(dim=-1).detach().cpu().tolist()
        bind_ids = bind_attn.argmax(dim=-1).detach().cpu().tolist()
        for span, t, sidx in zip(valid_spans, type_ids, bind_ids):
            curr.append((span[0], span[1], int(t), int(sidx)))
        outputs.append(curr)
    return outputs


def compute_entity_metrics(pred_entities, gold_entities):
    pred_set = set(pred_entities)
    gold_set = set(gold_entities)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, tp, fp, fn


def boundary_accuracy(boundary_logits, boundary_labels):
    pred = boundary_logits.argmax(dim=-1)
    mask = boundary_labels.ne(-100)
    correct = (pred.eq(boundary_labels) & mask).sum().item()
    total = mask.sum().item()
    return correct, total


def train_one_epoch(model, loader, optimizer, loss_computer, device):
    model.train()
    total_loss = 0.0
    total_steps = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        loss, _ = loss_computer(out, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        total_steps += 1
    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(model, loader, loss_computer, device, save_predictions: str = None):
    model.eval()
    total_tp = total_fp = total_fn = 0
    total_b_correct = total_b_total = 0
    total_loss = 0.0
    steps = 0
    pred_dump = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        loss, _ = loss_computer(out, batch)
        total_loss += float(loss.item())
        steps += 1

        corr, tot = boundary_accuracy(out["boundary_logits"], batch["boundary_labels"])
        total_b_correct += corr
        total_b_total += tot

        word_boundary_preds = decode_word_boundary_preds(out["boundary_logits"], batch["token_first_mask"], batch["words"])
        pred_spans = decode_spans_from_boundary(word_boundary_preds)
        pred_entities = classify_predicted_spans(model, out, batch, pred_spans)

        for i in range(len(batch["ids"])):
            gold = [(s, e, t) for (s, e, t) in batch["spans"][i]]
            pred = [(s, e, t) for (s, e, t, _) in pred_entities[i]]
            _, _, _, tp, fp, fn = compute_entity_metrics(pred, gold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            pred_dump.append({
                "id": batch["ids"][i],
                "text": batch["texts"][i],
                "gold_entities": [
                    {"token_start": s, "token_end": e, "type": ID2TYPE[t]} for (s, e, t) in gold
                ],
                "pred_entities": [
                    {"token_start": s, "token_end": e, "type": ID2TYPE[t], "slot": slot_idx} for (s, e, t, slot_idx) in pred_entities[i]
                ],
            })

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    metrics = {
        "loss": total_loss / max(steps, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "boundary_acc": total_b_correct / max(total_b_total, 1),
    }
    if save_predictions is not None:
        save_dir = os.path.dirname(save_predictions)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_predictions, "w", encoding="utf-8") as f:
            json.dump(pred_dump, f, ensure_ascii=False, indent=2)
    return metrics
