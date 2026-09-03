"""Unit tests for the pipeline CLI helpers (no model run here)."""
import unittest
from pathlib import Path

from nnunet_inference import pipeline


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = pipeline.parse_args(["in_dir", "out_dir", "--model-dir", "model"])
        self.assertEqual(args.input_dir, "in_dir")
        self.assertEqual(args.output_dir, "out_dir")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.checkpoint, "final")
        self.assertEqual(args.folds, "auto")
        self.assertFalse(args.disable_tta)
        self.assertEqual(args.model_dir, Path("model"))

    def test_output_dir_is_optional(self):
        # omitting output_dir -> None, which run_inference resolves to the
        # input series' parent (mask/ + vote_map/ written alongside the input)
        args = pipeline.parse_args(["in_dir", "--model-dir", "model"])
        self.assertEqual(args.input_dir, "in_dir")
        self.assertIsNone(args.output_dir)

    def test_flags(self):
        args = pipeline.parse_args(
            [
                "i", "o", "--model-dir", "model", "--device", "cpu",
                "--checkpoint", "best", "--folds", "0,1,2,3,4",
                "--no-tta",
            ]
        )
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.checkpoint, "best")
        self.assertEqual(args.folds, "0,1,2,3,4")
        self.assertTrue(args.disable_tta)

    def test_tta_can_be_enabled_explicitly(self):
        args = pipeline.parse_args(["i", "--model-dir", "model", "--tta"])
        self.assertFalse(args.disable_tta)

    def test_legacy_disable_tta_flag_is_kept(self):
        args = pipeline.parse_args(["i", "--model-dir", "model", "--disable-tta"])
        self.assertTrue(args.disable_tta)

    def test_checkpoint_map_covers_choices(self):
        self.assertEqual(pipeline.CHECKPOINTS["final"], "checkpoint_final.pth")
        self.assertEqual(pipeline.CHECKPOINTS["best"], "checkpoint_best.pth")

    def test_model_dir_is_required(self):
        with self.assertRaises(SystemExit):
            pipeline.parse_args(["in_dir"])


if __name__ == "__main__":
    unittest.main()
