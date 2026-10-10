"""Exercise the evaluation entry point with lightweight encoder doubles."""
import argparse
import ast
import contextlib
import io
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class EvalSeedTests(unittest.TestCase):
    def run_eval(self, train_args, explicit_seed=None):
        events = []
        constructor_args = {}

        class Model:
            def __init__(self, **kwargs):
                constructor_args.update(kwargs)
                events.append("model")
                random.random()  # Simulate initialization consuming RNG.

            def to(self, device):
                return self

            def load_state_dict(self, state, strict):
                pass

        def dataset(*args, **kwargs):
            events.append("dataset")
            random.random()
            return object()

        def seed(value):
            events.append(("seed", value))
            random.seed(value)

        def evaluate(*args, **kwargs):
            events.append("evaluate")
            return {"f1": 0.5, "sampling_probe": random.random()}

        namespace = dict(
            argparse=argparse, json=json,
            torch=SimpleNamespace(
                device=lambda name: name, cuda=SimpleNamespace(is_available=lambda: False),
                load=lambda path, map_location: {"args": train_args, "model_state": {}},
            ),
            LatentSlotMNER=Model, MNERConvertedDataset=dataset,
            make_loader=lambda *args, **kwargs: object(),
            LossConfig=lambda **kwargs: kwargs,
            LossComputer=lambda *args: object(),
            set_seed=seed, evaluate=evaluate,
        )
        tree = ast.parse((ROOT / "eval.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "eval.py"), "exec"), namespace)
        argv = ["eval.py", "--ckpt", "model.pt", "--test_json", "test.json", "--image_root", "images"]
        if explicit_seed is not None:
            argv.extend(["--seed", str(explicit_seed)])
        output = io.StringIO()
        with patch("sys.argv", argv), contextlib.redirect_stdout(output):
            namespace["main"]()
        return json.loads(output.getvalue()), events, constructor_args

    def test_default_uses_checkpoint_seed_after_initialization(self):
        output, events, _ = self.run_eval({"seed": 123})
        self.assertEqual(output["seed"], 123)
        self.assertEqual(output["sampling_probe"], random.Random(123).random())
        self.assertEqual(events[-2:], [("seed", 123), "evaluate"])

    def test_old_checkpoint_defaults_to_42_and_keeps_slot_induction(self):
        output, _, kwargs = self.run_eval({})
        self.assertEqual(output["seed"], 42)
        self.assertEqual(output["sampling_probe"], random.Random(42).random())
        self.assertTrue(kwargs["use_slot_induction"])

    def test_explicit_zero_overrides_checkpoint_seed(self):
        output, _, kwargs = self.run_eval({"seed": 123, "use_slot_induction": False}, explicit_seed=0)
        self.assertEqual(output["seed"], 0)
        self.assertEqual(output["sampling_probe"], random.Random(0).random())
        self.assertFalse(kwargs["use_slot_induction"])

    def test_repeat_evaluation_resets_sampling_stream(self):
        first, _, _ = self.run_eval({"seed": 42})
        random.random()
        second, _, _ = self.run_eval({"seed": 42})
        self.assertEqual(first["sampling_probe"], second["sampling_probe"])


if __name__ == "__main__":
    unittest.main()
