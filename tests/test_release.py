import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTests(unittest.TestCase):
    def test_paper_model_configuration(self):
        config = yaml.safe_load((ROOT / "configs/zpvehicles/brace_net.yaml").read_text())
        model = config["model"]
        encoder = model["encoder_args"]
        boundary = config["boundary_aux"]

        self.assertEqual(model["NAME"], "BRACENet")
        self.assertEqual(encoder["NAME"], "BRACEEncoder")
        self.assertEqual(encoder["blocks"], [1, 1, 1, 1, 1])
        self.assertEqual(encoder["strides"], [1, 4, 4, 4, 4])
        self.assertEqual(encoder["geo_stages"], [0, 1, 2, 3, 4])
        self.assertEqual(model["triplane_grid_size"], 64)
        self.assertEqual(model["triplane_dropout"], 0.1)
        self.assertEqual(model["triplane_logit_alpha_init"], 0.01)
        self.assertEqual(boundary["knn_k"], 16)
        self.assertEqual(boundary["lambda_b"], 0.2)
        self.assertTrue(boundary["balanced_loss"])

    def test_paper_training_protocol(self):
        config = yaml.safe_load((ROOT / "configs/zpvehicles/default.yaml").read_text())
        self.assertEqual(config["epochs"], 80)
        self.assertEqual(config["batch_size"], 8)
        self.assertEqual(config["dataset"]["common"]["num_points"], 4096)
        self.assertFalse(config["dataset"]["common"]["augmentation"]["enabled"])
        self.assertEqual(config["optimizer"]["NAME"], "adamw")
        self.assertEqual(config["lr"], 0.001)
        self.assertEqual(config["sched"], "cosine")

    def test_paper_split_sizes(self):
        expected = {"train": 74, "val": 16, "test": 16}
        seen = set()
        for split, count in expected.items():
            names = [
                line.strip()
                for line in (ROOT / f"data/splits/{split}.txt").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(names), count)
            self.assertTrue(seen.isdisjoint(names))
            seen.update(names)
        self.assertEqual(len(seen), 106)

    def test_release_contains_no_weights_or_old_repository_branding(self):
        weight_suffixes = {".pth", ".pt", ".ckpt", ".onnx", ".bin"}
        forbidden = ("point" + "next", "open" + "points")
        excluded = {ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "licenses/third-party-backbone-MIT.txt"}

        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path in excluded:
                continue
            self.assertNotIn(path.suffix.lower(), weight_suffixes, str(path))
            if path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".txt", ".sh"}:
                text = path.read_text(errors="ignore").lower()
                for marker in forbidden:
                    self.assertNotIn(marker, text, str(path))


if __name__ == "__main__":
    unittest.main()
