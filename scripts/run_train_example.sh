#!/usr/bin/env bash
python train.py \
  --train_json /path/to/train.json \
  --val_json /path/to/val.json \
  --image_root /path/to/images \
  --output_dir outputs/run1 \
  --text_model xlm-roberta-large \
  --vision_model openai/clip-vit-base-patch16 \
  --batch_size 4 \
  --epochs 10 \
  --lr 3e-5 \
  --seed 42 \
  --lambda_mask 0.2 \
  --freeze_vision
