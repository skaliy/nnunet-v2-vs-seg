"""Checks that the VS project still behaves as released.

They need this workstation's trained model, datasets and source images, so they
run only when asked for, from the repository root:

    VS_CHECKS=1 python -m unittest discover -s projects/vs/checks -p 'test_*.py' -v
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPOSITORY = Path(__file__).resolve().parents[3]
WORKSPACE = Path(os.environ.get("NNUNET_WORKSPACE", REPOSITORY / "nnUNet_data"))
SOURCE_ROOT = Path(os.environ.get(
    "VS_SOURCE_ROOT", REPOSITORY.parent / "fastMONAI/research/vestibular_schwannoma"))
DATASET = "Dataset003_VSSegmentationCurated"
RAW = WORKSPACE / "nnUNet_raw" / DATASET
MODEL = (WORKSPACE / "nnUNet_results" / DATASET
         / "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres")
CASES = 344
INDEX_SHA256 = "6ae5557a1b2bea2c4f30f5289a759d6db1b992881652458a31d5d02e82ea5cf7"
BUNDLE_SHA256 = "e91f17a3af1621a773a562ba12e426404322119234b91195fc8ef9382b0e79cc"
METRIC_TOLERANCE = 1e-6
PRESENTATION_COLUMNS = {"false_positive_component_count", "volume_mm3", "quartile_label"}

requires_workstation = unittest.skipUnless(
    os.environ.get("VS_CHECKS") == "1",
    "set VS_CHECKS=1 on the workstation that holds the VS model and data")


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=REPOSITORY, check=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_image(first, second):
    """Same grid and the same voxels."""
    a, b = sitk.ReadImage(str(first)), sitk.ReadImage(str(second))
    return (a.GetSize() == b.GetSize() and a.GetSpacing() == b.GetSpacing()
            and a.GetOrigin() == b.GetOrigin() and a.GetDirection() == b.GetDirection()
            and np.array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b)))


@requires_workstation
class VSUnchangedTests(unittest.TestCase):
    def test_staging_the_model_gives_the_registered_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            run("-m", "nnunet_inference.deployment", "--source", MODEL,
                "--output", Path(tmp) / "model", "--checkpoint", "final")
            self.assertEqual((Path(tmp) / "model" / "MODEL_VERSION.txt").read_text().strip(),
                             BUNDLE_SHA256)

    def test_prepare_reproduces_dataset003(self):
        index = SOURCE_ROOT / "data" / "ml_dataset.csv"
        self.assertEqual(file_sha256(index), INDEX_SHA256)
        # Inside the repository, so the images are hard links to the sources
        # like the ones in Dataset003, and nothing is copied.
        with tempfile.TemporaryDirectory(prefix=".vs-check-", dir=REPOSITORY) as tmp:
            run("training/prepare_dataset.py", "--data-csv", index, "--source-root", SOURCE_ROOT,
                "--dataset-id", 3, "--dataset-name", "VSSegmentationCurated",
                "--label-mode", "largest-component", "--nnunet-raw", tmp)
            built = Path(tmp) / DATASET
            for name in ("dataset.json", "splits_final.json", "component_counts.csv"):
                self.assertEqual((built / name).read_bytes(), (RAW / name).read_bytes(), name)
            for folder in ("imagesTr", "labelsTr"):
                names = sorted(path.name for path in (RAW / folder).iterdir())
                self.assertEqual(len(names), CASES)
                self.assertEqual(sorted(path.name for path in (built / folder).iterdir()), names)
            images = sorted(path.name for path in (RAW / "imagesTr").iterdir())
            self.assertEqual([name for name in images if not (
                os.path.samefile(built / "imagesTr" / name, RAW / "imagesTr" / name)
                or file_sha256(built / "imagesTr" / name) == file_sha256(RAW / "imagesTr" / name))], [])
            labels = sorted(path.name for path in (RAW / "labelsTr").iterdir())
            with ThreadPoolExecutor(max_workers=16) as pool:
                same = list(pool.map(
                    lambda name: same_image(built / "labelsTr" / name, RAW / "labelsTr" / name), labels))
            self.assertEqual([name for name, equal in zip(labels, same) if not equal], [])

    def test_evaluate_reproduces_the_stored_metrics(self):
        stored = pd.read_csv(MODEL / "cv_metrics" / "per_case.csv")
        with tempfile.TemporaryDirectory() as tmp:
            run("training/evaluate_cv.py", "--dataset", DATASET,
                "--nnunet-raw", WORKSPACE / "nnUNet_raw",
                "--nnunet-results", WORKSPACE / "nnUNet_results", "--output-dir", tmp)
            computed = pd.read_csv(Path(tmp) / "per_case.csv")
        self.assertEqual(len(computed), CASES)
        # The stored file has exactly three more columns, added later by presentation_summary.py.
        self.assertLessEqual(set(computed.columns), set(stored.columns))
        self.assertEqual(set(stored.columns) - set(computed.columns), PRESENTATION_COLUMNS)
        for column in computed.columns:
            if pd.api.types.is_numeric_dtype(computed[column]):
                np.testing.assert_allclose(
                    computed[column].to_numpy(dtype=float), stored[column].to_numpy(dtype=float),
                    rtol=0, atol=METRIC_TOLERANCE, equal_nan=True, err_msg=column)
            else:
                self.assertEqual(computed[column].tolist(), stored[column].tolist(), column)


if __name__ == "__main__":
    unittest.main()
