Paper defaults and runnable command templates are documented in README.md.

Use the 10% limited-supervision command for Tables 5, 6, and 7.
For ablation training, change one option and use a separate output directory:
  --no_slot_induction
  --lambda_mask 0
  --lambda_slot_diversity 0
  --lambda_aux 0

Independent evaluation accepts --seed and restores the model configuration
from checkpoint arguments. Slot analysis is in scripts/analyze_slots.py.
Keep the dataset splits, subset seed, checkpoint, package versions, evaluation
seed, and batch size with every experiment record. Do not treat example seed
choices as the unreported original seeds of the paper's Table 7.
