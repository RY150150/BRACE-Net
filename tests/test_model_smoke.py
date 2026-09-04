import unittest

import torch

from brace_net.models import build_model_from_cfg
from brace_net.utils import EasyConfig


class FinalModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = EasyConfig()
        config.load("configs/zpvehicles/brace_net.yaml", recursive=True)
        cls.model = build_model_from_cfg(config.model)

    def test_model_size_and_residual_initialization(self):
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertGreater(parameter_count, 800_000)
        self.assertLess(parameter_count, 850_000)
        self.assertAlmostEqual(self.model.triplane_logit_alpha.item(), 0.01, places=6)

        final_conv = [
            layer
            for layer in self.model.triplane_head.modules()
            if isinstance(layer, torch.nn.Conv1d)
        ][-1]
        self.assertEqual(torch.count_nonzero(final_conv.weight).item(), 0)
        if final_conv.bias is not None:
            self.assertEqual(torch.count_nonzero(final_conv.bias).item(), 0)

    def test_training_and_inference_outputs(self):
        batch_size, num_points = 2, 256
        positions = torch.rand(batch_size, num_points, 3)
        batch = {"pos": positions, "x": positions.transpose(1, 2).contiguous()}

        self.model.train()
        training_output = self.model(batch)
        self.assertEqual(tuple(training_output["seg_logits"].shape), (batch_size, 2, num_points))
        self.assertEqual(tuple(training_output["boundary_logits"].shape), (batch_size, 2, num_points))

        self.model.eval()
        with torch.no_grad():
            inference_output = self.model(batch)
        self.assertEqual(tuple(inference_output.shape), (batch_size, 2, num_points))


if __name__ == "__main__":
    unittest.main()
