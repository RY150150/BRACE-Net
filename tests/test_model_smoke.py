import unittest
from unittest.mock import Mock, patch

import torch

from brace_net.models import build_model_from_cfg
from brace_net.models.segmentation import base_seg
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

    def test_paper_initialization_order(self):
        config = EasyConfig()
        config.load("configs/zpvehicles/brace_net.yaml", recursive=True)
        calls = Mock()
        with patch.object(base_seg, "build_model_from_cfg", wraps=build_model_from_cfg) as builder:
            with patch.object(
                base_seg, "TriPlaneProjectionModule", wraps=base_seg.TriPlaneProjectionModule
            ) as projection:
                calls.attach_mock(builder, "build")
                calls.attach_mock(projection, "projection")
                build_model_from_cfg(config.model)

        order = [
            args[0].NAME if method == "build" else "TriPlane"
            for method, args, _ in calls.mock_calls
        ]
        self.assertEqual(
            order,
            ["BRACEEncoder", "HierarchicalPointDecoder", "TriPlane", "SegHead", "SegHead", "SegHead"],
        )

    def test_released_weight_loads_strictly(self):
        checkpoint = torch.load("weights/brace_net_final.pth", map_location="cpu")
        self.assertEqual(checkpoint["model_name"], "BRACE-Net")
        self.assertEqual(checkpoint["config"], "configs/zpvehicles/brace_net.yaml")
        self.assertEqual(checkpoint["seed"], 0)
        self.assertEqual(checkpoint["epoch"], 75)
        self.assertNotIn("optimizer", checkpoint)
        self.assertNotIn("scheduler", checkpoint)

        incompatible = self.model.load_state_dict(checkpoint["model"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

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
