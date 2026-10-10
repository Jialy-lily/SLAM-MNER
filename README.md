# SLAM: Slot-based Latent Alignment for Multi-Image Multimodal Named Entity Recognition

This repository contains the official implementation of **SLAM**, a slot-based latent alignment framework for **multi-image multimodal named entity recognition (MNER)** under full-resource and limited-supervision settings. SLAM is designed to organize noisy cross-image evidence and select mention-specific slot evidence for entity type prediction.

It includes the model implementation, data loading utilities, training and evaluation entry points, paper analysis tools, a sample data record, and regression tests.

This repository does not include MNER-MI data or images. Please obtain the dataset from the original benchmark release and convert it to the required JSON format.

## Authors and Contact

Jia Huang, Shuli Zhao, Yaru Zhao, and Binyang Li.

**Corresponding author:** Yaru Zhao ([zhaoyaru@uir.edu.cn](mailto:zhaoyaru@uir.edu.cn)).

**Repository:** [Jialy-lily/SLAM-MNER](https://github.com/Jialy-lily/SLAM-MNER).

**Paper status:** Accepted at DAI 2026. Final publication links and citation details will be added when available.

## Overview

![SLAM overview](assets/overview.png)

SLAM follows an evidence organization pipeline. Given a post with text and multiple associated images, the model first encodes text tokens and image patches as fine-grained multimodal evidence units. A boundary predictor proposes candidate mention spans from the textual representations. The unified token-patch evidence pool is then abstracted into a small set of latent semantic slots. Finally, each candidate mention selects the most relevant slots through mention-slot alignment, and the selected slot-aware context is used for entity type prediction.

The training objectives are marked as **training only** in the overview figure. They supervise boundary prediction, entity typing, contribution-aware slot masking, and lightweight sparsity/diversity constraints. During inference, SLAM uses the learned encoders, latent slots, mention-slot alignment module, and entity type classifier. The overview is Figure 2 of the accepted paper.

## Detailed Framework

![SLAM detailed framework](assets/detail_framework.png)

The detailed framework illustrates the full data flow:

1. Text tokens are encoded by a text encoder, while multiple images are encoded by a vision encoder and spatially pooled into patch-level visual evidence.
2. Textual token representations and visual patch representations are concatenated into a unified multimodal evidence pool.
3. Initial slots are independently sampled from a shared learnable diagonal Gaussian. Each slot attends over valid evidence atoms and is refined through GRU updates and a residual MLP. A learnable slot-index embedding is added after the final iteration.
4. Candidate mention spans are obtained from boundary labels. During training, gold spans are used for mention-level type supervision; during inference, decoded spans are used.
5. Each mention is pooled over its subword representations and softly aligned with all slots. The mention representation and weighted slot context are concatenated and passed through a Fusion FFN before type classification. Incomplete mentions after truncation are excluded from mention-level typing.
6. Training-only objectives include boundary loss, type loss, post-level auxiliary type prediction, contribution-aware slot masking, mention-slot sparsity, and slot diversity.

The detailed framework is Figure 3 of the accepted paper. The implementation follows Eqs. (4) and (6): evidence aggregation uses a slot-by-evidence attention matrix multiplied by evidence values, and mention pooling uses the word-to-subword mapping.

## Repository Structure

```text
SLAM-MNER/
├── assets/               # overview and detailed framework figures
├── configs/              # example notes and command snippets
├── examples/             # sample JSON data format
├── scripts/              # release checking and helper scripts
├── tests/                # regression tests without pretrained model downloads
├── slam/                 # core package
│   ├── data.py           # dataset loading and limited-supervision sampling
│   ├── model.py          # encoders, latent slots, mention-slot alignment, losses
│   └── utils.py          # training and evaluation utilities
├── train.py              # training entry point
├── eval.py               # evaluation entry point
├── .gitignore            # local data, checkpoints, outputs, and credentials
├── requirements.txt
└── README.md
```

## Environment

Python 3.10+ is recommended. Clone the repository and install dependencies with:

```bash
git clone https://github.com/Jialy-lily/SLAM-MNER.git
cd SLAM-MNER
pip install -r requirements.txt
```

Before releasing or running experiments, you can run the lightweight local check:

```bash
python scripts/check_release.py
```

This check validates Python syntax, the example data format, and common credential and local-path patterns, while allowing the listed public correspondence address. It does not download pretrained models or run training.

## Dataset Format

Each dataset file is a JSON list. Every item should follow the format below:

```json
{
  "id": "sample_0001",
  "text": "Met Emilia at Comic-Con! So cool!",
  "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"],
  "gold_entities": [
    {"token_start": 1, "token_end": 2, "type": "PER"},
    {"token_start": 3, "token_end": 4, "type": "MISC"}
  ]
}
```

Notes:

- `token_start` is inclusive and `token_end` is exclusive.
- Token indices are based on whitespace tokenization of `text`.
- Entity types follow `{PER, LOC, ORG, MISC}`.
- `image_paths` may be absolute paths or paths relative to `--image_root`.
- Verify that every image path exists before training. Missing or unreadable images are replaced by masked black placeholders.
- `examples/sample_record.json` illustrates the schema; its image filenames are placeholders.

## Training

### Full-Resource Training

```bash
python train.py \
  --train_json /path/to/train.json \
  --val_json /path/to/val.json \
  --image_root /path/to/images \
  --output_dir outputs/full \
  --text_model xlm-roberta-large \
  --vision_model openai/clip-vit-base-patch16 \
  --batch_size 4 \
  --epochs 10 \
  --lr 3e-5 \
  --seed 42 \
  --lambda_mask 0.2 \
  --freeze_vision
```

### Limited-Supervision Training

Use `--low_resource_ratio` to sample both the training and development sets at the requested ratio. Sampling is stratified by image count and entity-type combination, with at least one example per non-empty group, so the realized ratio may differ from the requested ratio. The command-line name is kept for limited-supervision experiments.

```bash
python train.py \
  --train_json /path/to/train.json \
  --val_json /path/to/val.json \
  --image_root /path/to/images \
  --output_dir outputs/limited10 \
  --low_resource_ratio 0.1 \
  --text_model xlm-roberta-large \
  --vision_model openai/clip-vit-base-patch16 \
  --batch_size 4 \
  --epochs 10 \
  --lr 3e-5 \
  --seed 42 \
  --lambda_mask 0.2 \
  --freeze_vision
```

Common limited-supervision ratios are `0.1`, `0.2`, and `0.4`.

### Paper Configuration

The commands above freeze the CLIP vision encoder, as in the accepted paper. The other default settings are:

| Setting | Value |
|---|---|
| Text / vision encoder | XLM-R-large / CLIP ViT-B/16 |
| Hidden size | 768 |
| Maximum subwords / images per post | 128 / 4 |
| Image pooling grid | 4 × 4 (16 patches per image) |
| Slots / refinement iterations | 8 / 3 |
| Slot MLP hidden size | 1024 |
| Fusion dropout | 0.1 |
| Batch size / epochs | 4 / 10 |
| Optimizer / learning rate / weight decay | AdamW / 3e-5 / 0.01 |
| Gradient clipping norm | 1.0 |
| Learning rate scheduler / warmup | None / none |
| Training and subset seed | 42 |
| Type / auxiliary / masking loss weights | 1.0 / 0.2 / 0.2 |
| Sparsity / diversity loss weights | 0.05 / 0.05 |
| Slot masking margin | 0.15 |

The test split remains complete in every supervision setting. Limited-supervision sampling is applied independently to train and development splits, stratified by capped image count and entity-type combination.

## Evaluation

```bash
python eval.py \
  --ckpt outputs/full/best_model.pt \
  --test_json /path/to/test.json \
  --image_root /path/to/images \
  --save_predictions outputs/full/predictions.json \
  --batch_size 4 \
  --seed 42
```

The evaluation script reports entity-level precision, recall, and F1. A prediction is counted as correct only when both the entity span and entity type match the gold annotation. It also reports word-level boundary accuracy, boundary-span precision/recall/F1 without type labels, and oracle typing accuracy on fully retained gold spans (Table 4). Metric values use the range 0–1; multiply by 100 to display percentages.

If `--seed` is omitted, evaluation uses the seed stored in the checkpoint, or 42 for older checkpoints without a seed. The selected seed is included in the JSON output. Initial slots remain stochastically sampled during evaluation, as specified in Eq. (3).

## Key Options

- `--num_slots`: number of latent semantic slots.
- `--slot_iters`: number of recurrent slot refinement iterations.
- `--pooled_hw`: spatial pooling size for each image.
- `--low_resource_ratio`: ratio of training and development examples used for limited-supervision experiments.
- `--freeze_vision`: freezes the CLIP vision encoder.
- `--lambda_mask`: weight for the contribution-aware slot masking objective.
- `--slot_masking_margin`: margin used by the slot masking objective.

## Training Objectives

The released training script keeps the main SLAM training objective. The loss weights can be adjusted with:

- `--lambda_type`: weight for mention-level entity type prediction.
- `--lambda_aux`: weight for post-level auxiliary type prediction.
- `--lambda_mask`: weight for contribution-aware slot masking.
- `--lambda_bind_sparsity`: weight for mention-slot sparsity regularization.
- `--lambda_slot_diversity`: weight for slot diversity regularization.

## Paper Ablations and Analysis

For Table 5, use the 10% limited-supervision training command and change only the listed option. Train each variant into a separate output directory, then evaluate its checkpoint on the full test split.

| Table 5 variant | Additional training option |
|---|---|
| Full model | None |
| w/o Slot Induction | `--no_slot_induction` |
| w/o Contribution-aware Slot Masking | `--lambda_mask 0` |
| w/o Slot Diversity | `--lambda_slot_diversity 0` |
| w/o Post-level Auxiliary Type Prediction | `--lambda_aux 0` |

The slot-induction ablation replaces recurrent slot construction with a masked mean over valid text and image atoms. The mean is used as the same global context for every mention, without slot-index embeddings. `eval.py` restores this choice from checkpoint arguments.

For Table 6, vary `--num_slots`, `--slot_iters`, `--pooled_hw`, `--lambda_bind_sparsity`, `--lambda_slot_diversity`, or `--slot_masking_margin` individually while retaining the other default settings and the 10% protocol.

### Slot Contribution and Evidence Routing

`scripts/analyze_slots.py` implements Section 4.8 using fully retained **gold mention spans**, so boundary prediction errors do not affect oracle typing analysis. It selects the top, bottom, or uniformly random slot using the original mention-slot attention, sets that slot representation to zero, and recomputes attention and entity typing. This differs from the training objective, which masks and renormalizes attention weights.

```bash
python scripts/analyze_slots.py \
  --ckpt outputs/limited10/best_model.pt \
  --test_json /path/to/test.json \
  --image_root /path/to/images \
  --output_json outputs/limited10/slot_analysis.json \
  --slot_seeds 42 43 44 45 46 \
  --random_seed 42 \
  --random_repeats 20 \
  --batch_size 4
```

These five seed values are an **example**, not recovered original experiment seeds. The paper specifies five slot-sampling seeds and 20 random repetitions per mention but does not list the seeds or the standard deviation denominator. The script requires explicit seeds and records all protocol choices; `--std_ddof 1` (default) uses sample standard deviation, and `--std_ddof 0` uses population standard deviation.

The output contains per-seed and mean/std values for oracle accuracy drop, gold-type probability drop (both in percentage points), and correct-to-incorrect flip rate. Flip rate is divided by the number of baseline-correct mentions; random interventions are averaged across the 20 repetitions for each mention.

To export Figure 5-style routing data for selected test posts, add `--routing_ids POST_ID --routing_seed 42`, with a routing seed included among the five slot seeds. Routing composes attention over **all** slots as `rho[u,j] = sum_i alpha[u,i] * beta[i,j]`, masks invalid atoms, renormalizes, and sums mass over text and each image. JSON includes mention-slot weights and subword/patch routing percentages. It does not reconstruct the paper's original example without its data, checkpoint, and sampling seed.

## Reproducibility Notes

The local release check validates syntax and packaging only; it does not establish reproduction of the paper results. Reproducing experiments requires the original dataset, images, pretrained encoders, and experiment settings. Dataset files and trained checkpoints are not distributed in this repository.

The model samples the initial latent slots during evaluation as well as training. Keep the evaluation seed, batch size, data order, model versions, and hardware environment fixed when comparing runs. The explicit evaluation seed controls a new evaluation run; it does not restore the RNG state of a historical training-time validation run.

The accepted paper reports full-resource F1 **82.62** and limited-supervision F1 **75.85 / 77.97 / 79.04** at **10% / 20% / 40%**. These are paper-reported results. This repository update has not rerun full training or independently verified those scores.

Regression checks can be run after installing the dependencies:

```bash
python -B -m unittest discover -s tests -v
python -B scripts/check_release.py
```

## License

No license has been selected yet. A license will be added after the code and accompanying assets have been reviewed. Public visibility alone does not grant an open-source license.
