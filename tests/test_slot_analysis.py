"""Hand-checkable accounting tests; no checkpoints or encoder downloads."""

import importlib.util
import math
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_slots.py"
SPEC = importlib.util.spec_from_file_location("analyze_slots", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class InterventionMetricsTests(unittest.TestCase):
    def test_micro_denominators_and_probability_points(self):
        # 8 baseline correct / 10 mentions, 7 altered correct, 2 flips
        # (one originally incorrect prediction also becomes correct).
        metrics = analysis.intervention_metrics(10, 8, 7, 0.6, 2)
        self.assertAlmostEqual(metrics["accuracy_drop_pp"], 10.0)
        self.assertAlmostEqual(metrics["gold_probability_drop_pp"], 6.0)
        self.assertAlmostEqual(metrics["flip_rate_percent"], 25.0)

    def test_random_average_counts_are_not_rounded(self):
        metrics = analysis.intervention_metrics(4, 3, 2.5, 0.2, 0.75)
        self.assertAlmostEqual(metrics["accuracy_drop_pp"], 12.5)
        self.assertAlmostEqual(metrics["flip_rate_percent"], 25.0)

    def test_no_correct_baselines_has_undefined_flip_rate(self):
        metrics = analysis.intervention_metrics(2, 0, 1, -0.1, 0)
        self.assertIsNone(metrics["flip_rate_percent"])
        self.assertEqual(metrics["accuracy_drop_pp"], -50.0)

    def test_no_mentions_is_an_error(self):
        with self.assertRaises(ValueError):
            analysis.intervention_metrics(0, 0, 0, 0, 0)


class RoutingTests(unittest.TestCase):
    def test_all_slots_are_composed_and_invalid_atom_removed(self):
        # alpha=[.25,.75], beta rows differ. Unnormalized mass is
        # [.225,.275,.35,.15]; atom 3 is invalid, leaving total .85.
        routing = analysis.compose_evidence_routing(
            [0.25, 0.75], [[0.6, 0.2, 0.2, 0.0], [0.1, 0.3, 0.4, 0.2]],
            [1, 1, 1, 0], text_length=2, patches_per_image=1,
        )
        self.assertAlmostEqual(routing["text_mass_percent"], 100 * 0.5 / 0.85)
        self.assertAlmostEqual(routing["image_mass_percent"][0], 100 * 0.35 / 0.85)
        self.assertEqual(routing["image_mass_percent"][1], 0.0)
        self.assertAlmostEqual(routing["text_mass_percent"] + sum(routing["image_mass_percent"]), 100)
        # Using only the most attended slot would produce text=50%, not this mixture.
        self.assertNotAlmostEqual(routing["text_mass_percent"], 50.0)

    def test_same_top_slot_can_have_different_composed_routing(self):
        beta = [[0.2, 0.4, 0.1, 0.3], [0.6, 0.1, 0.2, 0.1]]
        a = analysis.compose_evidence_routing([0.75, 0.25], beta, [1, 1, 0, 1], 2, 1)
        b = analysis.compose_evidence_routing([0.55, 0.45], beta, [1, 1, 0, 1], 2, 1)
        self.assertAlmostEqual(a["text_mass_percent"], 71.42857142857143)
        self.assertAlmostEqual(b["text_mass_percent"], 75.43859649122807)
        self.assertEqual(a["image_mass_percent"][0], 0)
        self.assertNotEqual(a["text_mass_percent"], b["text_mass_percent"])

    def test_grouping_each_image_patch_grid(self):
        routing = analysis.compose_evidence_routing(
            [1.0], [[0.2, 0.1, 0.2, 0.2, 0.3]], [1, 1, 1, 1, 1],
            text_length=1, patches_per_image=2,
        )
        self.assertAlmostEqual(routing["text_mass_percent"], 20)
        self.assertAlmostEqual(routing["image_mass_percent"][0], 30)
        self.assertAlmostEqual(routing["image_mass_percent"][1], 50)

    def test_no_valid_evidence_is_an_error(self):
        with self.assertRaises(ValueError):
            analysis.compose_evidence_routing([1.0], [[1.0]], [0], 1, 1)


class SeedSummaryTests(unittest.TestCase):
    def test_mean_std_are_over_seeds_with_declared_ddof(self):
        seed_results = []
        for value in [1, 2, 3, 4, 5]:
            metrics = {"accuracy_drop_pp": value, "gold_probability_drop_pp": 2 * value,
                       "flip_rate_percent": value / 2}
            seed_results.append({"interventions": {name: dict(metrics)
                                                   for name in ("top", "bottom", "random")}})
        result = analysis.summarize_seeds(seed_results, std_ddof=1)
        self.assertEqual(result["top"]["accuracy_drop_pp"]["mean"], 3)
        self.assertAlmostEqual(result["top"]["accuracy_drop_pp"]["std"], 2.5 ** 0.5)
        population = analysis.summarize_seeds(seed_results, std_ddof=0)
        self.assertAlmostEqual(population["top"]["accuracy_drop_pp"]["std"], 2 ** 0.5)

    def test_seed_flip_rates_are_not_pooled_across_denominators(self):
        seed_results = []
        for correct in [1, 2, 3, 4, 4]:
            metrics = analysis.intervention_metrics(5, correct, correct - 1, 0.2, 1)
            seed_results.append({"interventions": {name: dict(metrics)
                                                   for name in ("top", "bottom", "random")}})
        result = analysis.summarize_seeds(seed_results)
        self.assertAlmostEqual(result["random"]["flip_rate_percent"]["mean"], 46.66666666666667)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for tensor intervention tests")
class TorchInterventionTests(unittest.TestCase):
    def test_zero_representation_recomputes_attention_and_preserves_baseline(self):
        import torch

        class FakeClassifier:
            def classify_mentions(self, span_reprs, slots):
                keys = slots.squeeze(-1) + 0.5  # slot projection includes a bias
                attention = torch.softmax(span_reprs * keys, dim=-1)
                context = (attention * slots.squeeze(-1)).sum(dim=-1)
                self.last_attention = attention
                logits = torch.stack([context, torch.zeros_like(context)], dim=-1)
                return logits, attention, context, None

        model = FakeClassifier()
        spans = torch.ones(1, 1)
        slots = torch.tensor([[[2.0], [-1.0]]])
        original = slots.clone()
        base_logits, base_attention, _, _ = model.classify_mentions(spans, slots)
        top = base_attention.argmax(dim=-1)
        bottom = base_attention.argmin(dim=-1)
        altered = analysis.zero_slots(model, spans, slots, top)
        self.assertAlmostEqual(altered[0, 0].item(), -1 / (1 + math.e), places=6)
        self.assertAlmostEqual(model.last_attention[0, 0].item(), math.e / (1 + math.e), places=6)
        self.assertEqual(base_logits.argmax(dim=-1).item(), 0)
        self.assertEqual(altered.argmax(dim=-1).item(), 1)
        # Training's attention masking would yield a context of -1.
        self.assertNotAlmostEqual(altered[0, 0].item(), -1)
        altered_bottom = analysis.zero_slots(model, spans, slots, bottom)
        self.assertAlmostEqual(altered_bottom[0, 0].item(), 2 * math.exp(2) / (1 + math.exp(2)), places=6)
        self.assertTrue(torch.equal(slots, original))

    def test_random_repetitions_average_outcomes_not_logits(self):
        import torch

        totals = {"random": {"correct": 0.0, "probability_drop": 0.0, "flips": 0.0}}
        gold = torch.tensor([0])
        baseline_correct = torch.tensor([True])
        baseline_probability = torch.tensor([0.9])
        predictions = [[0.4, 0.5, 0.05, 0.05]] * 15 + [[0.8, 0.1, 0.05, 0.05]] * 5
        for probabilities in predictions:
            logits = torch.tensor([probabilities]).log()
            analysis._add_intervention(totals, "random", baseline_correct, baseline_probability,
                                       gold, logits, weight=1 / 20)
        metrics = analysis.intervention_metrics(1, 1, totals["random"]["correct"],
                                                totals["random"]["probability_drop"],
                                                totals["random"]["flips"])
        self.assertAlmostEqual(metrics["accuracy_drop_pp"], 75)
        self.assertAlmostEqual(metrics["gold_probability_drop_pp"], 40, places=4)
        self.assertAlmostEqual(metrics["flip_rate_percent"], 75)
        self.assertEqual(torch.tensor(predictions).mean(dim=0).argmax().item(), 0)

    def test_micro_statistics_include_recovered_predictions(self):
        import torch

        base = torch.tensor([[0.8, 0.1, 0.05, 0.05], [0.7, 0.1, 0.1, 0.1],
                             [0.6, 0.2, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1]])
        changed = torch.tensor([[0.4, 0.5, 0.05, 0.05], [0.65, 0.15, 0.1, 0.1],
                                [0.9, 0.04, 0.03, 0.03], [0.3, 0.25, 0.25, 0.2]])
        gold = torch.zeros(4, dtype=torch.long)
        totals = {"top": {"correct": 0.0, "probability_drop": 0.0, "flips": 0.0}}
        analysis._add_intervention(totals, "top", base.argmax(dim=-1).eq(gold), base[:, 0],
                                   gold, changed.log())
        metrics = analysis.intervention_metrics(4, 3, totals["top"]["correct"],
                                                totals["top"]["probability_drop"], totals["top"]["flips"])
        self.assertAlmostEqual(metrics["accuracy_drop_pp"], 0)
        self.assertAlmostEqual(metrics["gold_probability_drop_pp"], 1.25, places=4)
        self.assertAlmostEqual(metrics["flip_rate_percent"], 100 / 3)

    def test_routing_uses_post_owner_not_mention_index(self):
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch
        import torch

        data_module = ModuleType("slam.data")
        data_module.ID2TYPE = {0: "PER", 1: "LOC", 2: "ORG", 3: "MISC"}
        parent_module = ModuleType("slam")
        parent_module.__path__ = []
        batch = {
            "ids": ["p0", "p1"], "texts": ["A", "B"], "words": [["A"], ["B"]],
            "input_ids": torch.tensor([[0], [1]]), "image_mask": torch.ones(2, 1, dtype=torch.long),
        }
        out = {
            "span_pack": {"owner_index": [1, 0], "owner_spans": [(0, 1, 0), (0, 1, 0)],
                          "bind_attn": torch.tensor([[1.0, 0.0], [1.0, 0.0]])},
            "slot_to_atom_attn": torch.tensor([[[0.9, 0.1], [0.0, 1.0]],
                                               [[0.1, 0.9], [0.0, 1.0]]]),
            "atom_mask": torch.ones(2, 2, dtype=torch.long),
            "type_logits": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        }
        dataset = SimpleNamespace(max_images=1, tokenizer=SimpleNamespace(
            convert_ids_to_tokens=lambda ids: [str(i) for i in ids]))
        samples = {"p0": {"image_paths": ["a.jpg"]}, "p1": {"image_paths": ["b.jpg"]}}
        with patch.dict(sys.modules, {"slam": parent_module, "slam.data": data_module}):
            routing = analysis._export_routing(out, batch, dataset, 42, {"p0", "p1"}, 1, samples)
        self.assertEqual(routing[0]["id"], "p1")
        self.assertAlmostEqual(routing[0]["text_mass_percent"], 10, places=5)
        self.assertEqual(routing[0]["image_paths"], ["b.jpg"])
        self.assertEqual(routing[1]["id"], "p0")
        self.assertAlmostEqual(routing[1]["text_mass_percent"], 90, places=5)


if __name__ == "__main__":
    unittest.main()
