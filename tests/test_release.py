import unittest
from pathlib import Path

import numpy as np
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

    def test_processed_dataset_is_complete_and_sanitized(self):
        sample_dir = ROOT / "data/processed/samples"
        sample_paths = sorted(sample_dir.glob("*.npz"))
        expected_names = {
            line.strip()
            for split in ("train", "val", "test")
            for line in (ROOT / f"data/processed/splits/{split}.txt").read_text().splitlines()
            if line.strip()
        }

        self.assertEqual(len(sample_paths), 106)
        self.assertEqual({path.stem for path in sample_paths}, expected_names)

        total_points = 0
        for path in sample_paths:
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"pos", "x", "y"}, str(path))
                self.assertEqual(data["pos"].shape, data["x"].shape, str(path))
                self.assertEqual(data["pos"].shape[1], 3, str(path))
                self.assertEqual(data["y"].shape, (data["pos"].shape[0],), str(path))
                self.assertEqual(data["pos"].dtype, np.float32, str(path))
                self.assertEqual(data["x"].dtype, np.float32, str(path))
                self.assertEqual(data["y"].dtype, np.uint8, str(path))
                self.assertTrue(np.isin(data["y"], [0, 1]).all(), str(path))
                total_points += data["y"].shape[0]
        self.assertEqual(total_points, 10_352_125)

    def test_release_contains_no_weights_or_old_repository_branding(self):
        weight_suffixes = {".pth", ".pt", ".ckpt", ".onnx", ".bin"}
        released_weight = ROOT / "weights/brace_net_final.pth"
        forbidden = ("point" + "next", "open" + "points")
        excluded = {ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "licenses/third-party-backbone-MIT.txt"}

        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path in excluded:
                continue
            if path.suffix.lower() in weight_suffixes:
                self.assertEqual(path, released_weight, str(path))
            if path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".txt", ".sh"}:
                text = path.read_text(errors="ignore").lower()
                for marker in forbidden:
                    self.assertNotIn(marker, text, str(path))


if __name__ == "__main__":
    unittest.main()
