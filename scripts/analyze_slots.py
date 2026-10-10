"""Slot interventions and composed evidence routing from paper Section 4.8.

The checkpoint should come from the 10% limited-supervision experiment for
Table 7. This script does not train a model or supply the paper's unreported
slot-sampling seeds. Pass five explicit seeds and the random-choice seed.
"""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys


def intervention_metrics(mention_count, baseline_correct, altered_correct,
                         probability_drop_sum, correct_to_incorrect):
    """Micro measures; accuracy/probability drops are percentage points."""
    if mention_count <= 0:
        raise ValueError("At least one retained gold mention is required.")
    return {
        "accuracy_drop_pp": 100.0 * (baseline_correct - altered_correct) / mention_count,
        "gold_probability_drop_pp": 100.0 * probability_drop_sum / mention_count,
        "flip_rate_percent": (
            100.0 * correct_to_incorrect / baseline_correct if baseline_correct else None
        ),
        "intervened_correct_count": altered_correct,
        "correct_to_incorrect_count": correct_to_incorrect,
    }


def compose_evidence_routing(alpha, beta, valid_mask, text_length, patches_per_image):
    """Equation (14): mix every slot, mask invalid atoms, then normalize.

    Inputs are ordinary sequences so this numerical accounting can be checked
    without pretrained encoders. The top slot is not used to replace the sum.
    """
    atom_count = len(valid_mask)
    if not alpha or len(alpha) != len(beta):
        raise ValueError("Mention and evidence attention must have the same nonzero slot count.")
    if any(len(row) != atom_count for row in beta):
        raise ValueError("Every slot attention row must cover every evidence atom.")
    if patches_per_image <= 0 or not 0 <= text_length <= atom_count:
        raise ValueError("Invalid text length or image patch count.")
    if (atom_count - text_length) % patches_per_image:
        raise ValueError("Visual atoms must form complete per-image patch groups.")
    if any(not math.isfinite(float(x)) or x < 0 for x in alpha):
        raise ValueError("Mention attention must be finite and nonnegative.")
    if any(not math.isfinite(float(x)) or x < 0 for row in beta for x in row):
        raise ValueError("Evidence attention must be finite and nonnegative.")
    mass = [
        sum(alpha[i] * beta[i][j] for i in range(len(alpha))) if valid_mask[j] else 0.0
        for j in range(atom_count)
    ]
    total = sum(mass)
    if total <= 0:
        raise ValueError("No positive routing mass remains over valid evidence atoms.")
    mass_percent = [100.0 * x / total for x in mass]
    image_patch_mass = [
        mass_percent[start:start + patches_per_image]
        for start in range(text_length, atom_count, patches_per_image)
    ]
    return {
        "text_mass_percent": sum(mass_percent[:text_length]),
        "image_mass_percent": [sum(x) for x in image_patch_mass],
        "text_subword_mass_percent": mass_percent[:text_length],
        "image_patch_mass_percent": image_patch_mass,
    }


def summarize_seeds(seed_results, std_ddof=1):
    """Aggregate seed-level measures rather than treating repetitions as seeds."""
    if std_ddof not in (0, 1):
        raise ValueError("std_ddof must be zero or one.")
    if len(seed_results) <= std_ddof:
        raise ValueError("Not enough seeds for the requested standard deviation.")
    summary = {}
    for name in ("bottom", "random", "top"):
        summary[name] = {}
        for metric in ("accuracy_drop_pp", "gold_probability_drop_pp", "flip_rate_percent"):
            values = [r["interventions"][name][metric] for r in seed_results]
            if any(x is None for x in values):
                summary[name][metric] = {"mean": None, "std": None}
            else:
                std = statistics.stdev(values) if std_ddof else statistics.pstdev(values)
                summary[name][metric] = {"mean": statistics.mean(values), "std": std}
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--slot_seeds", type=int, nargs=5, required=True,
                        help="Five distinct evaluation slot-sampling seeds; paper values are not provided.")
    parser.add_argument("--random_seed", type=int, required=True,
                        help="Independent seed for uniform random slot choices, held fixed across slot seeds.")
    parser.add_argument("--random_repeats", type=int, default=20,
                        help="Random interventions per mention (paper protocol: 20).")
    parser.add_argument("--std_ddof", type=int, choices=(0, 1), default=1,
                        help="Standard deviation convention; paper does not specify the denominator.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--routing_ids", nargs="*", default=[],
                        help="Post IDs for full Equation (14) routing exports.")
    parser.add_argument("--routing_seed", type=int,
                        help="Explicit slot seed to export routing from; must occur in --slot_seeds.")
    return parser


def zero_slots(model, span_reprs, per_span_slots, slot_indices):
    """Zero the representation, then recompute attention and entity typing.

    This deliberately differs from training's mask-and-renormalize attention.
    The remaining latent slots and the baseline draw are held fixed.
    """
    import torch

    changed_slots = per_span_slots.clone()
    rows = torch.arange(changed_slots.size(0), device=changed_slots.device)
    changed_slots[rows, slot_indices] = 0
    logits, _, _, _ = model.classify_mentions(span_reprs, changed_slots)
    return logits


def _add_intervention(totals, name, base_correct, base_gold_probability, gold, logits, weight=1.0):
    import torch

    changed_probability = torch.softmax(logits, dim=-1).gather(1, gold[:, None]).squeeze(1)
    changed_correct = logits.argmax(dim=-1).eq(gold)
    totals[name]["correct"] += weight * changed_correct.sum().item()
    totals[name]["probability_drop"] += weight * (
        base_gold_probability - changed_probability
    ).double().sum().item()
    totals[name]["flips"] += weight * (base_correct & ~changed_correct).sum().item()


def _export_routing(out, batch, dataset, seed, requested_ids, patches_per_image, id_to_sample):
    from slam.data import ID2TYPE

    exports = []
    pack = out["span_pack"]
    if pack is None:
        return exports
    for j, owner in enumerate(pack["owner_index"]):
        post_id = str(batch["ids"][owner])
        if post_id not in requested_ids:
            continue
        span = pack["owner_spans"][j]
        alpha = pack["bind_attn"][j].detach().cpu().tolist()
        beta = out["slot_to_atom_attn"][owner].detach().cpu().tolist()
        routing = compose_evidence_routing(
            alpha, beta, out["atom_mask"][owner].detach().cpu().tolist(),
            batch["input_ids"].size(1), patches_per_image,
        )
        sample = id_to_sample[post_id]
        tokens = dataset.tokenizer.convert_ids_to_tokens(batch["input_ids"][owner].cpu().tolist())
        top_index = max(range(len(alpha)), key=alpha.__getitem__)
        exports.append({
            "id": post_id,
            "slot_seed": seed,
            "text": batch["texts"][owner],
            "token_start": int(span[0]),
            "token_end": int(span[1]),
            "mention": " ".join(batch["words"][owner][span[0]:span[1]]),
            "gold_type": ID2TYPE[int(span[2])],
            "predicted_type": ID2TYPE[int(out["type_logits"][j].argmax().item())],
            "top_slot_index": top_index,
            "top_slot_number": top_index + 1,
            "mention_slot_attention": alpha,
            "image_paths": sample.get("image_paths", [])[:dataset.max_images],
            "image_validity": batch["image_mask"][owner].cpu().tolist(),
            "text_subwords": tokens,
            **routing,
        })
    return exports


def analyze(model, loader, device, slot_seeds, random_seed, random_repeats=20,
            routing_ids=(), routing_seed=None, dataset=None, patches_per_image=16):
    import torch
    from slam.utils import move_batch_to_device, set_seed

    if random_repeats < 1:
        raise ValueError("random_repeats must be positive.")
    requested = set(map(str, routing_ids))
    id_to_sample = {}
    if requested:
        for idx, sample in enumerate(dataset.samples):
            post_id = str(sample.get("id", str(idx)))
            if post_id in id_to_sample:
                raise ValueError("Routing requires unique dataset post IDs.")
            id_to_sample[post_id] = sample
        missing = requested.difference(id_to_sample)
        if missing:
            raise ValueError(f"Routing post IDs are absent from the dataset: {sorted(missing)}")

    seed_results = []
    routing_exports = []
    model.eval()
    with torch.no_grad():
        for seed in slot_seeds:
            set_seed(seed)
            choice_rng = random.Random(random_seed)
            mention_count = baseline_correct_count = gold_count = 0
            totals = {name: {"correct": 0.0, "probability_drop": 0.0, "flips": 0.0}
                      for name in ("bottom", "random", "top")}
            for batch in loader:
                gold_count += sum(len(spans) for spans in batch["spans"])
                batch = move_batch_to_device(batch, device)
                out = model(batch)
                pack = out["span_pack"]
                if pack is None:
                    continue
                gold = pack["span_types"]
                base_logits = out["type_logits"]
                base_correct = base_logits.argmax(dim=-1).eq(gold)
                base_probability = torch.softmax(base_logits, dim=-1).gather(1, gold[:, None]).squeeze(1)
                n, slots = pack["bind_attn"].shape
                mention_count += n
                baseline_correct_count += base_correct.sum().item()
                for name, indices in (("top", pack["bind_attn"].argmax(dim=-1)),
                                      ("bottom", pack["bind_attn"].argmin(dim=-1))):
                    logits = zero_slots(model, pack["span_reprs"], pack["per_span_slots"], indices)
                    _add_intervention(totals, name, base_correct, base_probability, gold, logits)
                for _ in range(random_repeats):
                    indices = torch.tensor([choice_rng.randrange(slots) for _ in range(n)],
                                           dtype=torch.long, device=gold.device)
                    logits = zero_slots(model, pack["span_reprs"], pack["per_span_slots"], indices)
                    _add_intervention(totals, "random", base_correct, base_probability, gold, logits,
                                      weight=1.0 / random_repeats)
                if requested and seed == routing_seed:
                    routing_exports.extend(_export_routing(
                        out, batch, dataset, seed, requested, patches_per_image, id_to_sample,
                    ))
            if not mention_count:
                raise ValueError("No retained gold mentions were available for oracle-span analysis.")
            seed_results.append({
                "slot_seed": seed,
                "gold_mentions": gold_count,
                "retained_gold_mentions": mention_count,
                "excluded_gold_mentions": gold_count - mention_count,
                "baseline_correct_count": baseline_correct_count,
                "baseline_oracle_typing_accuracy_percent": 100.0 * baseline_correct_count / mention_count,
                "interventions": {
                    name: intervention_metrics(
                        mention_count, baseline_correct_count, values["correct"],
                        values["probability_drop"], values["flips"],
                    ) for name, values in totals.items()
                },
            })
    return seed_results, routing_exports


def main():
    args = build_parser().parse_args()
    if len(set(args.slot_seeds)) != 5:
        raise ValueError("Pass five distinct slot-sampling seeds.")
    if args.batch_size < 1 or args.random_repeats < 1:
        raise ValueError("batch_size and random_repeats must be positive.")
    if args.routing_ids and args.routing_seed not in args.slot_seeds:
        raise ValueError("--routing_ids requires --routing_seed from the five --slot_seeds.")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from slam.data import MNERConvertedDataset, make_loader
    from slam.model import LatentSlotMNER

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    train_args = checkpoint.get("args", {})
    config = {
        "text_name": train_args.get("text_model", "xlm-roberta-large"),
        "vision_name": train_args.get("vision_model", "openai/clip-vit-base-patch16"),
        "hidden_size": train_args.get("hidden_size", 768),
        "num_slots": train_args.get("num_slots", 8),
        "slot_iters": train_args.get("slot_iters", 3),
        "pooled_hw": train_args.get("pooled_hw", 4),
        "freeze_vision": train_args.get("freeze_vision", True),
        "use_slot_induction": train_args.get("use_slot_induction", True),
    }
    if not config["use_slot_induction"]:
        raise ValueError("Section 4.8 slot analysis requires a model with slot induction enabled.")
    model = LatentSlotMNER(**config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dataset = MNERConvertedDataset(
        args.test_json, text_model_name=config["text_name"], vision_model_name=config["vision_name"],
        max_length=train_args.get("max_length", 128), max_images=train_args.get("max_images", 4),
        image_root=args.image_root,
    )
    loader = make_loader(dataset, args.batch_size, shuffle=False)
    seed_results, routing = analyze(
        model, loader, device, args.slot_seeds, args.random_seed, args.random_repeats,
        args.routing_ids, args.routing_seed, dataset, config["pooled_hw"] ** 2,
    )
    output = {
        "protocol": {
            "paper_section": "4.8",
            "checkpoint": args.ckpt,
            "test_json": args.test_json,
            "checkpoint_training_ratio": train_args.get("low_resource_ratio"),
            "table_7_requires_training_ratio": 0.1,
            "slot_seeds": args.slot_seeds,
            "paper_slot_seeds_provided": False,
            "random_choice_seed": args.random_seed,
            "random_choice_seed_provided_by_paper": False,
            "random_choice_distribution": "uniform over all slots, with replacement, independently per mention",
            "random_choice_seed_reset_for_each_slot_seed": True,
            "random_repeats_per_mention": args.random_repeats,
            "std_ddof": args.std_ddof,
            "paper_std_ddof_specified": False,
            "span_source": "retained gold word spans; no boundary predictions",
            "intervention": "zero selected slot representation and recompute mention-slot attention and entity type",
            "selection": "original mention-slot attention; top=argmax, bottom=argmin; first index breaks ties",
            "flip_rate_denominator": "baseline-correct retained gold mentions",
            "device": str(device),
            "batch_size": args.batch_size,
            "routing_seed": args.routing_seed,
            "routing_formula": "sum_i alpha[u,i] * beta[i,j], masked and renormalized over valid atoms",
            "routing_text_source": "all text positions marked valid by the model, including valid special tokens",
            "model_config": config,
        },
        "per_seed": seed_results,
        "mean_std_over_slot_seeds": summarize_seeds(seed_results, args.std_ddof),
        "routing": routing,
        "routing_posts_without_retained_mentions": sorted(
            set(map(str, args.routing_ids)) - {x["id"] for x in routing}
        ),
    }
    target = Path(args.output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output["mean_std_over_slot_seeds"], indent=2, allow_nan=False))
    print(f"Saved per-seed statistics and routing to {target}")


if __name__ == "__main__":
    main()
