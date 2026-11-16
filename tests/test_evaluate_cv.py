"""training/evaluate_cv.py must score complete folds only and keep its columns."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import evaluate_cv  # noqa: E402

DATASET = "Dataset900_Synthetic"
MODEL = "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
COLUMNS = ["fold", "case_id", "dsc", "sensitivity", "precision", "ldr", "rve", "assd_mm",
           "hd95_mm", "nsd_tau0.5_mm", "nsd_tau1.0_mm", "nsd_tau2.0_mm", "surface_status",
           "spacing_mm"]
SPLITS = [dict(train=["c", "d"], val=["a", "b"]), dict(train=["a", "b"], val=["c", "d"])]


def fake_metrics(prediction, target, spacing, *, nsd_tolerances_mm):
    """Stands in for fastMONAI: only the overlap is real."""
    overlap = float((prediction * target).sum())
    panel = dict(dsc=2 * overlap / float(prediction.sum() + target.sum()), sensitivity=1.0,
                 precision=1.0, ldr=1.0, rve=0.0, assd_mm=0.0, hd95_mm=0.0,
                 surface_status="ok", spacing_mm=spacing)
    panel.update({f"nsd_tau{tolerance}_mm": 1.0 for tolerance in nsd_tolerances_mm})
    return panel


def write_mask(path, voxels):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    array[:voxels, 0, 0] = 1
    nib.save(nib.Nifti1Image(array, np.diag([0.5, 0.5, 1.5, 1.0])), str(path))


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw" / DATASET
        self.model = self.root / "results" / DATASET / MODEL
        for case in "abcd":
            write_mask(self.raw / "labelsTr" / f"{case}.nii.gz", 2)
        (self.raw / "splits_final.json").write_text(json.dumps(SPLITS))
        for fold, cases in enumerate(("ab", "cd")):
            for case in cases:
                write_mask(self.model / f"fold_{fold}" / "validation" / f"{case}.nii.gz", 1)

    def evaluate(self, *extra):
        arguments = ["--dataset", DATASET, "--folds", "0", "1", "--nnunet-raw", str(self.root / "raw"),
                     "--nnunet-preprocessed", str(self.root / "preprocessed"),
                     "--nnunet-results", str(self.root / "results"), *extra]
        with redirect_stdout(io.StringIO()):
            return evaluate_cv.main(arguments, calculate_all_metrics=fake_metrics)

    def test_complete_folds_write_the_same_columns_as_before(self):
        self.assertEqual(self.evaluate(), 0)
        frame = pd.read_csv(self.model / "cv_metrics" / "per_case.csv")
        self.assertEqual(list(frame.columns), COLUMNS)
        self.assertEqual(frame.case_id.tolist(), ["a", "b", "c", "d"])
        self.assertEqual(frame.fold.tolist(), [0, 0, 1, 1])
        np.testing.assert_allclose(frame.dsc, 2 / 3)
        self.assertEqual(set(frame.spacing_mm), {"0.5x0.5x1.5"})
        summary = pd.read_csv(self.model / "cv_metrics" / "summary.csv")
        self.assertEqual(set(summary.group), {"all", "fold_0", "fold_1"})

    def test_partly_validated_fold_is_an_error(self):
        (self.model / "fold_1" / "validation" / "d.nii.gz").unlink()
        with self.assertRaisesRegex(SystemExit, r"fold 1 validation is not complete.*missing \['d'\]"):
            self.evaluate()
        with self.assertRaisesRegex(SystemExit, "fold 1 validation is not complete"):
            self.evaluate("--allow-missing-folds")
        self.assertFalse((self.model / "cv_metrics").exists())

    def test_prediction_from_another_fold_is_an_error(self):
        write_mask(self.model / "fold_0" / "validation" / "c.nii.gz", 1)
        with self.assertRaisesRegex(SystemExit, r"fold 0 .*not in this fold \['c'\]"):
            self.evaluate()

    def test_fold_without_predictions_needs_allow_missing_folds(self):
        for path in (self.model / "fold_1" / "validation").iterdir():
            path.unlink()
        with self.assertRaisesRegex(SystemExit, r"No validation predictions for requested folds: \[1\]"):
            self.evaluate()
        self.assertEqual(self.evaluate("--allow-missing-folds"), 0)
        frame = pd.read_csv(self.model / "cv_metrics" / "per_case.csv")
        self.assertEqual(frame.case_id.tolist(), ["a", "b"])

    def test_splits_used_for_training_take_precedence_and_must_agree(self):
        preprocessed = self.root / "preprocessed" / DATASET
        preprocessed.mkdir(parents=True)
        (preprocessed / "splits_final.json").write_text(json.dumps(SPLITS))
        self.assertEqual(self.evaluate(), 0)
        (preprocessed / "splits_final.json").write_text(json.dumps(SPLITS[::-1]))
        with self.assertRaisesRegex(SystemExit, "splits_final.json differs"):
            self.evaluate()
        (self.raw / "splits_final.json").unlink()
        with self.assertRaisesRegex(SystemExit, "fold 0 validation is not complete"):
            self.evaluate()

    def test_fold_outside_the_splits_is_an_error(self):
        with self.assertRaisesRegex(SystemExit, "has 2 folds"):
            self.evaluate("--folds", "0", "5")

    def test_fastmonai_root_option_is_gone(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            evaluate_cv._parser().parse_args(["--dataset", DATASET, "--fastmonai-root", "."])

    @unittest.skipUnless(importlib.util.find_spec("fastMONAI"), "fastMONAI is not installed")
    def test_installed_fastmonai_panel_fills_every_column(self):
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            # SimpleITK's SWIG bindings warn this when fastMONAI first imports them.
            warnings.filterwarnings("ignore", r"builtin type \w+ has no __module__ attribute",
                                    DeprecationWarning)
            self.assertEqual(evaluate_cv.main(
                ["--dataset", DATASET, "--folds", "0", "1", "--nnunet-raw", str(self.root / "raw"),
                 "--nnunet-preprocessed", str(self.root / "preprocessed"),
                 "--nnunet-results", str(self.root / "results")]), 0)
        frame = pd.read_csv(self.model / "cv_metrics" / "per_case.csv")
        self.assertEqual(list(frame.columns), COLUMNS)
        np.testing.assert_allclose(frame.dsc, 2 / 3, atol=1e-6)
        np.testing.assert_allclose(frame.sensitivity, 0.5, atol=1e-6)
        self.assertEqual(set(frame.surface_status), {"ok"})


if __name__ == "__main__":
    unittest.main()
