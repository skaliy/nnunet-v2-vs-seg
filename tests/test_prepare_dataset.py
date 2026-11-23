"""training/prepare_dataset.py must refuse inputs nnU-Net would misread."""
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import prepare_dataset  # noqa: E402

SHAPE = (4, 6, 8)   # (z, y, x)


def write_volume(path, array, *, spacing=(0.5, 0.5, 1.5), origin=(10.0, 20.0, 30.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = []
        rng = np.random.default_rng(0)
        for index in range(4):
            case = f"case{index}"
            mask = np.zeros(SHAPE, dtype=np.uint8)
            mask[1:3, 1:3, 1:3] = 1          # the lesion
            mask[0, 5, 7] = 1                # a stray voxel
            write_volume(self.root / "images" / f"{case}.nii.gz",
                         rng.integers(0, 100, SHAPE).astype(np.int16))
            write_volume(self.root / "masks" / f"{case}.nii.gz", mask)
            self.rows.append(dict(case_id=case, t1_img_path=f"images/{case}.nii.gz",
                                  t1_seg_path=f"masks/{case}.nii.gz", fold=index % 2 + 1))

    def index(self):
        path = self.root / "index.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def load(self):
        self.printed = io.StringIO()
        with redirect_stdout(self.printed):
            return prepare_dataset.load_cases(self.index(), self.root)

    def test_valid_index_builds_the_dataset(self):
        destination = self.root / "raw" / "Dataset900_Synthetic"
        _, components = prepare_dataset.build_dataset(
            cases=self.load(), destination=destination, dataset_name="Synthetic",
            label_mode="largest-component", workers=2, copy_images=False)
        self.assertEqual(json.loads((destination / "dataset.json").read_text()), dict(
            channel_names={"0": "T1"}, labels={"background": 0, "VS": 1}, numTraining=4,
            file_ending=".nii.gz", dataset_name="Synthetic"))
        self.assertEqual(json.loads((destination / "splits_final.json").read_text()), [
            dict(train=["case1", "case3"], val=["case0", "case2"]),
            dict(train=["case0", "case2"], val=["case1", "case3"])])
        self.assertEqual(sorted(p.name for p in (destination / "imagesTr").iterdir()),
                         [f"case{i}_0000.nii.gz" for i in range(4)])
        self.assertEqual(set(components.values()), {2})
        label = sitk.ReadImage(str(destination / "labelsTr" / "case0.nii.gz"))
        self.assertEqual(int(sitk.GetArrayFromImage(label).sum()), 8)   # stray voxel removed
        self.assertEqual(label.GetSpacing(), (0.5, 0.5, 1.5))

    def test_mask_on_another_grid_is_rejected_when_the_index_is_loaded(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        for case, changes, expected in (
                ("case0", dict(spacing=(0.5, 0.5, 1.0)),
                 r"case0: spacing \(0\.5, 0\.5, 1\.0\) in the mask, \(0\.5, 0\.5, 1\.5\) in the image"),
                # One slice further along z: three voxels of the smallest spacing.
                ("case1", dict(origin=(10.0, 20.0, 31.5)), r"case1: corner voxels are 3\.00 voxels apart"),
                ("case2", dict(array=np.zeros((4, 6, 7), dtype=np.uint8)),
                 r"case2: size \(7, 6, 4\) in the mask, \(8, 6, 4\) in the image")):
            write_volume(self.root / "masks" / f"{case}.nii.gz", changes.pop("array", mask), **changes)
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, expected):
                self.load()
        with self.assertRaisesRegex(ValueError, r"not on the same grid \(3 cases"):
            self.load()

    def test_header_rounding_is_noted_and_a_flipped_mask_is_rejected(self):
        image, mask = self.root / "images" / "case0.nii.gz", self.root / "masks" / "case0.nii.gz"
        self.assertEqual(prepare_dataset.grid_mismatch(image, mask), (None, 0.0))
        # A rotation of 0.01 rad moves the far corner by a small part of a voxel.
        rotated = sitk.ReadImage(str(mask))
        cos, sin = np.cos(0.01), np.sin(0.01)
        rotated.SetDirection((cos, -sin, 0, sin, cos, 0, 0, 0, 1))
        sitk.WriteImage(rotated, str(mask))
        error, shift = prepare_dataset.grid_mismatch(image, mask)
        self.assertIsNone(error)
        self.assertTrue(0.01 < shift < prepare_dataset.MAX_CORNER_SHIFT, shift)
        self.assertEqual(len(self.load()), 4)
        self.assertIn("Note: 1 masks have a slightly different origin or direction",
                      self.printed.getvalue())
        rotated.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))
        sitk.WriteImage(rotated, str(mask))
        with self.assertRaisesRegex(ValueError, "case0: corner voxels are"):
            self.load()

    def prepare(self, *arguments, environment=None):
        """Run the script's main() with nnU-Net's variables set only as given."""
        variables = {key: value for key, value in os.environ.items()
                     if key not in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}
        with mock.patch.dict(os.environ, variables | (environment or {}), clear=True), \
                redirect_stdout(io.StringIO()):
            return prepare_dataset.main([
                "--data-csv", str(self.index()), "--source-root", str(self.root),
                "--dataset-id", "900", "--dataset-name", "Synthetic",
                "--label-mode", "binary", "--workers", "1", *arguments])

    def test_dataset_id_in_use_in_the_raw_folder_is_refused(self):
        raw = self.root / "workspace" / "nnUNet_raw"
        (raw / "Dataset900_Other").mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit, f"in use.*{re.escape(str(raw / 'Dataset900_Other'))}"):
            self.prepare("--nnunet-raw", str(raw))
        self.assertEqual([path.name for path in raw.iterdir()], ["Dataset900_Other"])

    def test_dataset_id_in_use_only_in_the_results_next_to_raw_is_refused(self):
        workspace = self.root / "workspace"
        (workspace / "nnUNet_results" / "Dataset900_Synthetic").mkdir(parents=True)
        for arguments, environment in (
                (["--nnunet-raw", str(workspace / "nnUNet_raw")], None),
                ([], dict(nnUNet_raw=str(workspace / "nnUNet_raw")))):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                    SystemExit, "in use.*" + re.escape(
                        str(workspace / "nnUNet_results" / "Dataset900_Synthetic"))):
                self.prepare(*arguments, environment=environment)
        self.assertFalse((workspace / "nnUNet_raw").exists())

    def test_dataset_id_in_the_environment_workspace_is_refused_without_nnunet_raw(self):
        workspace = self.root / "workspace"
        (workspace / "nnUNet_preprocessed" / "Dataset900_Other").mkdir(parents=True)
        environment = dict(nnUNet_raw=str(self.root / "scratch" / "raw"),
                           nnUNet_preprocessed=str(workspace / "nnUNet_preprocessed"),
                           nnUNet_results=str(workspace / "nnUNet_results"))
        with self.assertRaisesRegex(SystemExit, "in use.*" + re.escape(
                str(workspace / "nnUNet_preprocessed" / "Dataset900_Other"))):
            self.prepare(environment=environment)
        self.assertFalse((self.root / "scratch").exists())

    def test_explicit_nnunet_raw_does_not_consult_the_environment_workspace(self):
        """The VS check builds Dataset003 into a scratch folder while the variables
        point at the real workspace."""
        workspace = self.root / "workspace"
        for folder in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            (workspace / folder / "Dataset900_Synthetic").mkdir(parents=True)
        scratch = self.root / "scratch"
        self.assertEqual(self.prepare("--nnunet-raw", str(scratch), environment={
            folder: str(workspace / folder)
            for folder in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}), 0)
        self.assertTrue((scratch / "Dataset900_Synthetic" / "dataset.json").is_file())

    def test_image_that_is_not_nii_gz_is_rejected(self):
        write_volume(self.root / "images" / "case0.nii", np.zeros(SHAPE, dtype=np.int16))
        self.rows[0]["t1_img_path"] = "images/case0.nii"
        with self.assertRaisesRegex(ValueError, r"Images must be \.nii\.gz files \(1 are not"):
            self.load()


if __name__ == "__main__":
    unittest.main()
