"""Verify the paper's stochastic slot initialization without backbone downloads."""
import importlib.util
import unittest


HAS_MODEL_DEPENDENCIES = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_MODEL_DEPENDENCIES, "PyTorch and Transformers are required")
class SlotSamplingTests(unittest.TestCase):
    def test_eval_seed_controls_sampling_without_disabling_random_slots(self):
        import torch
        from slam.model import SlotAttention
        from slam.utils import set_seed

        set_seed(7)
        module = SlotAttention(num_slots=3, dim=4, iters=2, hidden_dim=8).eval()
        atoms = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.long)
        with torch.no_grad():
            set_seed(42)
            first, _ = module(atoms, mask)
            set_seed(42)
            repeated, _ = module(atoms, mask)
            set_seed(43)
            different, _ = module(atoms, mask)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, different))


if __name__ == "__main__":
    unittest.main()
