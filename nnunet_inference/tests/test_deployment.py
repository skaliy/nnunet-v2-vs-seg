import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from nnunet_inference import deployment


def model_fixture(root):
    source = root / "source"
    source.mkdir()
    plans = dict(dataset_name=deployment.DATASET, plans_name=deployment.PLANS,
                 configurations={"3d_fullres": dict(patch_size=[48, 192, 192],
                     architecture=dict(network_class_name="dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                                       arch_kwargs=dict(features_per_stage=[32, 64, 128, 256, 320, 320])))})
    (source / "plans.json").write_text(json.dumps(plans))
    (source / "dataset.json").write_text(json.dumps(dict(channel_names={"0": "T1"},
        labels={"background": 0, "VS": 1}, numTraining=344, file_ending=".nii.gz")))
    for fold in range(5):
        path = source / f"fold_{fold}" / "checkpoint_best.pth"
        path.parent.mkdir()
        torch.save(dict(network_weights={"weight": torch.ones(2)}, trainer_name="nnUNetTrainer",
                        current_epoch=10 + fold, init_args=dict(configuration="3d_fullres", fold=fold),
                        inference_allowed_mirroring_axes=(0, 1, 2),
                        optimizer_state={"large": torch.ones(100)}), path)
    return source


class BundleTests(unittest.TestCase):
    def test_staging_preserves_original_weights_and_strips_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = model_fixture(root)
            before = deployment.file_sha256(source / "fold_0/checkpoint_best.pth")
            with patch.object(deployment, "build_predictor") as loader:
                manifest = deployment.prepare_bundle(source, root / "model")
                self.assertEqual(loader.call_count, 5)
            self.assertEqual(deployment.validate_deployment(root / "model"), manifest)
            checkpoint = torch.load(root / "model/fold_0/checkpoint_best.pth", weights_only=False)
            self.assertNotIn("optimizer_state", checkpoint)
            torch.testing.assert_close(checkpoint["network_weights"]["weight"], torch.ones(2))
            self.assertEqual(checkpoint["inference_allowed_mirroring_axes"], (0, 1, 2))
            self.assertEqual(deployment.file_sha256(source / "fold_0/checkpoint_best.pth"), before)
            (root / "model/fold_0/checkpoint_best.pth").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "file checksum"):
                deployment.validate_deployment(root / "model")

    def test_wrong_plan_partial_folds_and_wrong_fold_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = model_fixture(root)
            path = source / "fold_4/checkpoint_best.pth"
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                deployment.prepare_bundle(source, root / "model")
            self.assertFalse((root / "model").exists())
            (source / "plans.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "ResEnc-M"):
                deployment.validate_plan(source)

    def test_manifest_configuration_is_bound_to_bundle_digest(self):
        m = {"checkpoint": "checkpoint_best.pth", "folds": [0, 1, 2, 3, 4]}
        digest = deployment.bundle_sha256(m)
        m["checkpoint"] = "checkpoint_final.pth"
        self.assertNotEqual(deployment.bundle_sha256(m), digest)
