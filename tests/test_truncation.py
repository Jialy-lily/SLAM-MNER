"""Exercise truncation without downloading encoders or importing PyTorch."""
import ast
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TinyTensor:
    def __init__(self, values):
        self.values = values

    def squeeze(self, _dim):
        return self

    def __getitem__(self, key):
        return TinyTensor(self.values[key])

    def mean(self, dim):
        assert dim == 0
        return [sum(row[col] for row in self.values) / len(self.values) for col in range(len(self.values[0]))]


class Encoding(dict):
    def __init__(self, word_ids):
        super().__init__(input_ids=TinyTensor([0] * len(word_ids)), attention_mask=TinyTensor([1] * len(word_ids)))
        self.ids = word_ids

    def word_ids(self, batch_index=0):
        return self.ids


class TruncationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = dict(Counter=Counter, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, Any=Any)
        cls.maps = staticmethod(load_function(ROOT / "slam/data.py", "build_complete_word_token_maps", namespace))
        load_function(ROOT / "slam/data.py", "build_boundary_word_labels", namespace)
        load_function(ROOT / "slam/data.py", "build_spans", namespace)
        namespace.update(
            BOUNDARY2ID={"O": 0, "B-ENT": 1, "I-ENT": 2},
            TYPE2ID={"PER": 0, "LOC": 1, "ORG": 2, "OTHER": 3, "MISC": 3},
            torch=SimpleNamespace(tensor=lambda values, dtype: TinyTensor(values), long="long"),
        )
        cls.get_item = staticmethod(load_function(ROOT / "slam/data.py", "__getitem__", namespace, "MNERConvertedDataset"))
        cls.span_rep = staticmethod(load_function(ROOT / "slam/model.py", "get_span_rep", {}, "LatentSlotMNER"))

    def test_complete_words_keep_original_token_positions(self):
        ids = [None, 0, 1, 1, 1, 2, None, None]
        starts, ends = self.maps(ids, ids[:-1])
        self.assertEqual(starts, {0: 1, 1: 2, 2: 5})
        self.assertEqual(ends, {0: 1, 1: 4, 2: 5})
        tokens = TinyTensor([[float(i), float(i)] for i in range(8)])
        self.assertEqual(self.span_rep(None, tokens, starts, ends, (1, 2, 0)), [3.0, 3.0])

    def test_dataset_skips_partially_retained_word_for_mention_typing(self):
        calls = []

        def tokenizer(words, **kwargs):
            calls.append(kwargs)
            if kwargs["truncation"]:
                return Encoding([None, 0, 1, 1, None])
            return Encoding([None, 0, 1, 1, 1, 2, None])

        dataset = SimpleNamespace(
            samples=[{"text": "Met Longname Here", "gold_entities": [{"token_start": 1, "token_end": 2, "type": "PER"}]}],
            tokenizer=tokenizer, max_length=5,
            _load_images=lambda paths: (None, None),
        )
        item = self.get_item(dataset, 0)
        self.assertEqual(item["word_to_token_start"], {0: 1})
        self.assertEqual(item["word_to_token_end"], {0: 1})
        # Preserve the existing first-subword boundary supervision.
        self.assertEqual(item["boundary_labels"].values, [-100, 0, 1, -100, -100])
        self.assertEqual(item["token_first_mask"].values, [0, 1, 1, 0, 0])
        self.assertTrue(any(call["truncation"] is False for call in calls))
        tokens = TinyTensor([[float(i), float(i)] for i in range(5)])
        self.assertIsNone(self.span_rep(None, tokens, item["word_to_token_start"], item["word_to_token_end"], (1, 2, 0)))
        self.assertIsNone(self.span_rep(None, tokens, item["word_to_token_start"], item["word_to_token_end"], (0, 2, 0)))

    def test_wholly_truncated_word_is_not_mapped(self):
        starts, ends = self.maps([None, 0, None], [None, 0, 1, 1, None])
        tokens = TinyTensor([[0.0], [1.0], [2.0]])
        self.assertIsNone(self.span_rep(None, tokens, starts, ends, (1, 2, 0)))
        self.assertEqual(self.span_rep(None, tokens, starts, ends, (0, 1, 0)), [1.0])


if __name__ == "__main__":
    unittest.main()
