"""Unit tests for the pipeline CLI helpers (no model run here)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nnunet_inference import pipeline


class TestSliceStackCheck(unittest.TestCase):
    def test_the_stack_is_checked_before_nnunet_sees_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").mkdir()
            model = root / "model"
            model.mkdir()
            (model / "MODEL_VERSION.txt").write_text("a" * 64)
            (model / "MODEL_NAME.txt").write_text("Synthetic")
            series = object()
            predictor = SimpleNamespace(allowed_mirroring_axes=None)
            with patch.object(pipeline, "resolve_folds", return_value=(0,)), \
                    patch.object(pipeline, "build_predictor", return_value=predictor), \
                    patch.object(pipeline.dicom_io, "read_series", return_value=series) as read, \
                    patch.object(pipeline.dicom_io, "check_slice_stack",
                                 side_effect=ValueError("not evenly spaced")) as check, \
                    patch.object(pipeline.dicom_io, "series_to_nifti") as to_nifti, \
                    patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "not evenly spaced"):
                    pipeline.run_inference(root / "input", root / "output", model_dir=model,
                                           device="cpu", disable_tta=True)
            read.assert_called_once_with(root / "input")
            check.assert_called_once_with(series, root / "input")
            to_nifti.assert_not_called()


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
