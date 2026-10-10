import argparse
import json

import torch

from slam.data import MNERConvertedDataset, make_loader
from slam.model import LatentSlotMNER, LossComputer, LossConfig
from slam.utils import evaluate, set_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained SLAM checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_predictions", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--text_model", type=str, default="xlm-roberta-large")
    parser.add_argument("--vision_model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--slot_iters", type=int, default=3)
    parser.add_argument("--pooled_hw", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Slot-sampling seed; defaults to the checkpoint training seed, or 42.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {})
    eval_seed = args.seed if args.seed is not None else train_args.get("seed", 42)

    model = LatentSlotMNER(
        text_name=train_args.get("text_model", args.text_model),
        vision_name=train_args.get("vision_model", args.vision_model),
        hidden_size=train_args.get("hidden_size", args.hidden_size),
        num_slots=train_args.get("num_slots", args.num_slots),
        slot_iters=train_args.get("slot_iters", args.slot_iters),
        pooled_hw=train_args.get("pooled_hw", args.pooled_hw),
        freeze_vision=train_args.get("freeze_vision", True),
        use_slot_induction=train_args.get("use_slot_induction", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    test_set = MNERConvertedDataset(
        args.test_json,
        text_model_name=train_args.get("text_model", args.text_model),
        vision_model_name=train_args.get("vision_model", args.vision_model),
        max_length=train_args.get("max_length", args.max_length),
        max_images=train_args.get("max_images", args.max_images),
        image_root=args.image_root,
    )
    test_loader = make_loader(test_set, args.batch_size, shuffle=False)

    loss_computer = LossComputer(
        model,
        LossConfig(
            lambda_type=train_args.get("lambda_type", 1.0),
            lambda_aux=train_args.get("lambda_aux", 0.2),
            lambda_mask=train_args.get("lambda_mask", 0.2),
            lambda_bind_sparsity=train_args.get("lambda_bind_sparsity", 0.05),
            lambda_slot_diversity=train_args.get("lambda_slot_diversity", 0.05),
            slot_masking_margin=train_args.get("slot_masking_margin", 0.15),
        ),
    )
    # Reset after initialization so encoder construction cannot consume the
    # random stream used for stochastic Gaussian slot sampling.
    set_seed(eval_seed)
    metrics = evaluate(model, test_loader, loss_computer, device, save_predictions=args.save_predictions)
    metrics["seed"] = eval_seed
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
