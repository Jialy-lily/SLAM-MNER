import argparse
import json
import os

import torch

from slam.data import MNERConvertedDataset, build_limited_supervision_subset, make_loader
from slam.model import LatentSlotMNER, LossComputer, LossConfig
from slam.utils import evaluate, set_seed, train_one_epoch


def build_parser():
    parser = argparse.ArgumentParser(description="Train SLAM for multi-image multimodal NER.")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--text_model", type=str, default="xlm-roberta-large")
    parser.add_argument("--vision_model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--slot_iters", type=int, default=3)
    parser.add_argument("--pooled_hw", type=int, default=4)
    parser.add_argument("--low_resource_ratio", type=float, default=1.0)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--lambda_type", type=float, default=1.0)
    parser.add_argument("--lambda_aux", type=float, default=0.2)
    parser.add_argument("--lambda_mask", type=float, default=0.2)
    parser.add_argument("--lambda_bind_sparsity", type=float, default=0.05)
    parser.add_argument("--lambda_slot_diversity", type=float, default=0.05)
    parser.add_argument("--slot_masking_margin", type=float, default=0.15)
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = MNERConvertedDataset(
        args.train_json,
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        max_length=args.max_length,
        max_images=args.max_images,
        image_root=args.image_root,
    )
    val_set = MNERConvertedDataset(
        args.val_json,
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        max_length=args.max_length,
        max_images=args.max_images,
        image_root=args.image_root,
    )

    train_subset_indices = None
    val_subset_indices = None
    if args.low_resource_ratio < 1.0:
        train_subset_indices = build_limited_supervision_subset(
            train_set.samples,
            ratio=args.low_resource_ratio,
            seed=args.seed,
        )
        val_subset_indices = build_limited_supervision_subset(
            val_set.samples,
            ratio=args.low_resource_ratio,
            seed=args.seed,
        )

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, subset_indices=train_subset_indices)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, subset_indices=val_subset_indices)

    model = LatentSlotMNER(
        text_name=args.text_model,
        vision_name=args.vision_model,
        hidden_size=args.hidden_size,
        num_slots=args.num_slots,
        slot_iters=args.slot_iters,
        pooled_hw=args.pooled_hw,
        freeze_vision=args.freeze_vision,
    ).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    loss_computer = LossComputer(
        model,
        LossConfig(
            lambda_type=args.lambda_type,
            lambda_aux=args.lambda_aux,
            lambda_mask=args.lambda_mask,
            lambda_bind_sparsity=args.lambda_bind_sparsity,
            lambda_slot_diversity=args.lambda_slot_diversity,
            slot_masking_margin=args.slot_masking_margin,
        ),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_computer, device)
        val_metrics = evaluate(model, val_loader, loss_computer, device)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_p={val_metrics['precision']:.4f} | "
            f"val_r={val_metrics['recall']:.4f} | val_bacc={val_metrics['boundary_acc']:.4f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)

    print(f"Training finished. Best val F1 = {best_f1:.4f}")
    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
