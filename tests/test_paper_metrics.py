"""Regression checks for the paper's Table 4 metrics and Section 4.5 ablation.

Metric aggregation checks run with the Python standard library. The numerical
pooling check additionally needs PyTorch, but never downloads encoder weights.
Run with: python -m unittest discover -s tests -p 'test_paper_metrics.py' -v
"""

import argparse
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HAS_MODEL_DEPENDENCIES = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


def load_functions(relative_path, names, namespace, class_name=None):
    """Execute the actual source functions without loading pretrained backbones."""
    module = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    nodes = module.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    functions = []
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    if {node.name for node in functions} != set(names):
        raise AssertionError("Requested function is missing from source")
    exec(compile(ast.Module(body=functions, type_ignores=[]), relative_path, "exec"), namespace)
    return namespace


class EvaluationModel:
    def eval(self):
        pass

    def __call__(self, batch):
        return batch["out"]


class ZeroLoss:
    def item(self):
        return 0.0


def evaluation_metrics(batches):
    namespace = load_functions(
        "slam/utils.py",
        ["compute_entity_metrics", "evaluate"],
        {
            "move_batch_to_device": lambda batch, device: batch,
            "boundary_accuracy": lambda logits, labels: (0, 0),
            "oracle_typing_accuracy": lambda out: out["oracle_counts"],
            "decode_word_boundary_preds": lambda logits, mask, words: logits,
            "decode_spans_from_boundary": lambda predictions: predictions,
            "classify_predicted_spans": lambda model, out, batch, spans: out["typed_predictions"],
            "ID2TYPE": {0: "PER", 1: "LOC", 2: "ORG", 3: "MISC"},
        },
    )
    return namespace["evaluate"](
        EvaluationModel(), batches, lambda out, batch: (ZeroLoss(), {}), "cpu"
    )


def make_batch(gold, predicted_spans, typed_predictions, oracle_counts):
    size = len(gold)
    return {
        "ids": [str(i) for i in range(size)],
        "texts": ["Example text"] * size,
        "words": [["Example", "text"]] * size,
        "spans": gold,
        "boundary_labels": None,
        "token_first_mask": None,
        "out": {
            "boundary_logits": predicted_spans,
            "typed_predictions": typed_predictions,
            "oracle_counts": oracle_counts,
        },
    }


class PaperMetricsTests(unittest.TestCase):
    def test_boundary_span_f1_ignores_type_errors(self):
        batch = make_batch(
            [[(0, 1, 0)]], [[(0, 1)]], [[(0, 1, 1, 0)]], (1, 1)
        )
        metrics = evaluation_metrics([batch])
        self.assertEqual(metrics["boundary_span_f1"], 1.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["oracle_typing_acc"], 1.0)

    def test_micro_counts_and_unseen_gold_spans(self):
        # One correct span in the first batch. The second has a false positive
        # and three gold mentions, including spans not retained after truncation.
        batches = [
            make_batch([[(0, 1, 0)]], [[(0, 1)]], [[(0, 1, 0, 0)]], (1, 1)),
            make_batch(
                [[(2, 3, 0), (4, 5, 1), (6, 7, 2)]],
                [[(0, 1)]], [[(0, 1, 0, 0)]], (1, 2),
            ),
        ]
        metrics = evaluation_metrics(batches)
        self.assertEqual(metrics["boundary_span_precision"], 0.5)
        self.assertEqual(metrics["boundary_span_recall"], 0.25)
        self.assertAlmostEqual(metrics["boundary_span_f1"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["oracle_typing_acc"], 2.0 / 3.0)
        self.assertEqual(metrics["oracle_typing_total"], 3)

    def test_no_gold_mentions_is_defined(self):
        metrics = evaluation_metrics([make_batch([[]], [[]], [[]], (0, 0))])
        self.assertEqual(metrics["boundary_span_f1"], 0.0)
        self.assertEqual(metrics["oracle_typing_acc"], 0.0)
        self.assertEqual(metrics["oracle_typing_total"], 0)

    def test_slot_induction_enabled_unless_ablation_requested(self):
        namespace = load_functions("train.py", ["build_parser"], {"argparse": argparse})
        parser = namespace["build_parser"]()
        required = ["--train_json", "train.json", "--val_json", "dev.json",
                    "--image_root", "images", "--output_dir", "outputs"]
        self.assertTrue(parser.parse_args(required).use_slot_induction)
        self.assertFalse(parser.parse_args(required + ["--no_slot_induction"]).use_slot_induction)


@unittest.skipUnless(HAS_MODEL_DEPENDENCIES, "PyTorch and Transformers are required for tensor checks")
class PaperTorchMetricsTests(unittest.TestCase):
    def test_real_boundary_decode_and_oracle_type_predictions(self):
        import torch
        from slam.utils import evaluate

        batch = make_batch([[(0, 1, 0)]], [[(0, 1)]], [[(0, 1, 1, 0)]], (1, 1))
        for name in ("input_ids", "attention_mask", "boundary_labels", "token_first_mask"):
            batch[name] = torch.tensor([[1, 0]]) if name == "boundary_labels" else torch.ones(1, 2, dtype=torch.long)
        batch["pixel_values"] = torch.zeros(1, 1, 3, 2, 2)
        batch["image_mask"] = torch.ones(1, 1, dtype=torch.long)
        batch["out"].update({
            "boundary_logits": torch.tensor([[[0.0, 3.0, 0.0], [3.0, 0.0, 0.0]]]),
            "type_logits": torch.tensor([[3.0, 0.0, 0.0, 0.0]]),
            "span_pack": {"span_types": torch.tensor([0])},
        })
        # Stub only the model's predicted-span typing; boundary decoding,
        # gold-span typing counts, tensor transfers and aggregation are real.
        with patch("slam.utils.classify_predicted_spans", return_value=[[(0, 1, 1, 0)]]):
            metrics = evaluate(EvaluationModel(), [batch], lambda out, batch: (ZeroLoss(), {}), "cpu")
        self.assertEqual(metrics["boundary_acc"], 1.0)
        self.assertEqual(metrics["boundary_span_f1"], 1.0)
        self.assertEqual(metrics["oracle_typing_acc"], 1.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_oracle_accuracy_uses_all_gold_predictions_and_handles_no_spans(self):
        import torch
        from slam.utils import oracle_typing_accuracy

        counts = oracle_typing_accuracy({
            "type_logits": torch.tensor([[0., 2., 0., 0.], [0., 0., 2., 0.], [2., 0., 0., 0.]]),
            "span_pack": {"span_types": torch.tensor([1, 2, 3])},
        })
        self.assertEqual(counts, (2, 3))
        self.assertEqual(oracle_typing_accuracy({"type_logits": None, "span_pack": None}), (0, 0))


@unittest.skipUnless(HAS_MODEL_DEPENDENCIES, "PyTorch and Transformers are required for tensor checks")
class MeanPoolingAblationTests(unittest.TestCase):
    def make_model(self, num_slots):
        import torch
        from slam.model import LatentSlotMNER

        # Construct the module without pretrained encoders. The tested method
        # is the production implementation imported normally from slam.model.
        model = LatentSlotMNER.__new__(LatentSlotMNER)
        torch.nn.Module.__init__(model)
        model.use_slot_induction = False
        model.num_slots = num_slots
        return model

    def test_masked_mean_is_shared_and_does_not_use_slot_parameters(self):
        import torch
        # Deliberately omit slot_attention and slot_pos_embed: the ablation
        # must not use either module to produce the shared global context.
        model = self.make_model(num_slots=3)
        atoms = torch.tensor([[[2.0, 4.0], [1000.0, 1000.0], [6.0, 8.0]]], requires_grad=True)
        slots, attention = model.induce_slots(atoms, torch.tensor([[1, 0, 1]]))
        torch.testing.assert_close(slots, torch.tensor([[[4.0, 6.0]]]).expand(1, 3, 2))
        torch.testing.assert_close(attention, torch.tensor([[[0.5, 0.0, 0.5]]]).expand(1, 3, 3))
        slots[:, 0].sum().backward()
        torch.testing.assert_close(atoms.grad[:, 1], torch.zeros(1, 2))
        self.assertTrue(torch.isfinite(atoms.grad).all())

    def test_all_masked_evidence_returns_finite_zeros(self):
        import torch
        slots, attention = self.make_model(num_slots=2).induce_slots(
            torch.ones(1, 3, 2), torch.zeros(1, 3, dtype=torch.long),
        )
        self.assertEqual(torch.count_nonzero(slots).item(), 0)
        self.assertEqual(torch.count_nonzero(attention).item(), 0)


if __name__ == "__main__":
    unittest.main()
