"""Unit tests for nnunet_inference.dicom_io (geometry I/O)."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.uid import UID

from nnunet_inference import dicom_io

DEFAULT_FIXTURE = Path(__file__).resolve().parent / "dicom_input"
FIXTURE = Path(os.environ.get("NNUNET_DICOM_FIXTURE", DEFAULT_FIXTURE))
HAS_DICOM_FIXTURE = FIXTURE.is_dir() and any(FIXTURE.iterdir())
requires_dicom_fixture = unittest.skipUnless(
    HAS_DICOM_FIXTURE,
    "set NNUNET_DICOM_FIXTURE to a local DICOM series to run this test",
)
EXPECTED_SHAPE = (176, 512, 448)          # (z, y, x)
EXPECTED_SPACING_ZYX = (1.1, 0.541015625, 0.541015625)
UID_CONTEXT = dicom_io.make_uid_context(
    model_name="ResEnc-L Dataset002 five-fold",
    model_version="a" * 64,
    checkpoint_name="checkpoint_final.pth",
    folds=(0, 1, 2, 3, 4),
    tta=False,
)


class TestUIDGeneration(unittest.TestCase):
    def test_uids_are_numeric_deterministic_and_bind_inference_identity(self):
        mask = dicom_io.make_derived_dicom_uid(
            UID_CONTEXT, "source-series", "segmentation"
        )
        repeated = dicom_io.make_derived_dicom_uid(
            UID_CONTEXT, "source-series", "segmentation"
        )
        probability = dicom_io.make_derived_dicom_uid(
            UID_CONTEXT, "source-series", "probability"
        )
        instance = dicom_io.make_derived_dicom_uid(
            UID_CONTEXT, "source-series", "segmentation",
            source_sop_uid="source-instance", slice_index=0,
        )
        tta_context = dicom_io.make_uid_context(
            "ResEnc-L Dataset002 five-fold", "a" * 64,
            "checkpoint_final.pth", (0, 1, 2, 3, 4), True,
        )
        changed_tta = dicom_io.make_derived_dicom_uid(
            tta_context, "source-series", "segmentation"
        )

        self.assertEqual(mask, repeated)
        self.assertEqual(len({mask, probability, instance, changed_tta}), 4)
        for value in (mask, probability, instance, changed_tta):
            self.assertTrue(UID(value).is_valid, value)
            self.assertLessEqual(len(value), 64)
            self.assertRegex(value, r"^2\.25\.[0-9]+$")


@requires_dicom_fixture
class TestInputBridge(unittest.TestCase):
    def test_read_series_shape_spacing_patient(self):
        series = dicom_io.read_series(FIXTURE)
        self.assertEqual(tuple(series.shape), EXPECTED_SHAPE)
        self.assertEqual(series.slices, 176)
        self.assertEqual(series.patientID, "VESTSC00001")
        for got, exp in zip(series.spacing, EXPECTED_SPACING_ZYX):
            self.assertAlmostEqual(float(got), exp, places=3)

    def test_series_to_nifti_names_and_roundtrips_geometry(self):
        series = dicom_io.read_series(FIXTURE)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "VS_0000.nii.gz"
            dicom_io.series_to_nifti(series, target)
            # exactly one nii.gz, named exactly as requested
            written = sorted(Path(tmp).glob("*.nii.gz"))
            self.assertEqual([p.name for p in written], ["VS_0000.nii.gz"])
            # SimpleITK reads it back as the same (z, y, x) grid + spacing
            img = sitk.ReadImage(str(target))
            arr = sitk.GetArrayFromImage(img)
            self.assertEqual(arr.shape, EXPECTED_SHAPE)
            self.assertEqual(img.GetSize(), (448, 512, 176))   # sitk (x, y, z)
            sx, sy, sz = img.GetSpacing()
            for got, exp in zip((sz, sy, sx), EXPECTED_SPACING_ZYX):
                self.assertAlmostEqual(got, exp, places=3)

    def test_series_to_nifti_rejects_bad_name(self):
        series = dicom_io.read_series(FIXTURE)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dicom_io.series_to_nifti(series, Path(tmp) / "VS.nii.gz")


@requires_dicom_fixture
class TestOutputBridge(unittest.TestCase):
    """Model-free geometry/values proof for the NIfTI -> DICOM path."""

    def _imagedata_marker_nifti(self, arr, tmp):
        """Write `arr` (imagedata (z, y, x) convention) to a NIfTI VIA imagedata,
        so imagedata-read recovers it exactly — the convention the write path
        uses. A SimpleITK-built NIfTI would be in a different axis order, so
        reading it with SimpleITK in the write path silently flips the output;
        this helper + the assertion below are the regression guard for that bug.
        """
        series = dicom_io.read_series(FIXTURE)
        series[:] = arr.astype(np.uint16)
        out = Path(tmp) / "marker_0000.nii.gz"
        dicom_io.series_to_nifti(series, out)
        return out

    def _sitk_float_nifti_on_grid(self, arr, tmp):
        """Write a float `arr` (z, y, x) as a NIfTI on the fixture's grid via
        SimpleITK — mirrors how prob_npz_to_nifti produces the probability NIfTI
        that the write path consumes."""
        series = dicom_io.read_series(FIXTURE)
        ref = Path(tmp) / "ref_0000.nii.gz"
        dicom_io.series_to_nifti(series, ref)
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(sitk.ReadImage(str(ref)))
        out = Path(tmp) / "prob.nii.gz"
        sitk.WriteImage(img, str(out))
        return out

    def test_write_mask_dicom_preserves_orientation_and_values(self):
        # asymmetric marker in a known (low-z, low-y, low-x) corner
        marker = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
        marker[0:15, 0:25, 0:35] = 1
        with tempfile.TemporaryDirectory() as tmp:
            nifti = self._imagedata_marker_nifti(marker, tmp)
            out_dir = Path(tmp) / "mask"
            dicom_io.write_mask_dicom(nifti, FIXTURE, out_dir, UID_CONTEXT)
            readback = np.asarray(dicom_io.read_series(out_dir))
            self.assertEqual(readback.shape, EXPECTED_SHAPE)
            self.assertTrue(set(np.unique(readback)).issubset({0, 1}))
            # orientation preserved end-to-end: marker lands in the same corner
            self.assertTrue(np.array_equal((readback > 0).astype(np.uint8), marker))

    def test_write_prob_dicom_scales_0_1_to_uint16(self):
        # three uniform value regions; verified by value (orientation-independent)
        prob = np.full(EXPECTED_SHAPE, 0.5, dtype=np.float32)
        prob[0:40] = 1.0
        prob[40:80] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            nifti = self._sitk_float_nifti_on_grid(prob, tmp)
            out_dir = Path(tmp) / "vote_map"
            dicom_io.write_prob_dicom(nifti, FIXTURE, out_dir, UID_CONTEXT)
            rb = np.asarray(dicom_io.read_series(out_dir))
            self.assertEqual(rb.dtype, np.uint16)
            self.assertEqual(int(rb.min()), 0)        # 0.0 -> 0
            self.assertEqual(int(rb.max()), 65535)    # 1.0 -> 65535
            vals = np.unique(rb)
            self.assertTrue(np.any(np.abs(vals - 32767) <= 1))  # 0.5 -> ~32767

    def test_save_series_pred_derives_unique_uids(self):
        mask = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
        mask[5, 5, 5] = 1
        with tempfile.TemporaryDirectory() as tmp:
            nifti = self._imagedata_marker_nifti(mask, tmp)
            out_dir = Path(tmp) / "mask"
            dicom_io.write_mask_dicom(nifti, FIXTURE, out_dir, UID_CONTEXT)
            sop_uids, series_uids = [], set()
            for f in sorted(Path(out_dir).iterdir()):
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                sop_uids.append(ds.SOPInstanceUID)
                series_uids.add(ds.SeriesInstanceUID)
            self.assertEqual(len(sop_uids), 176)
            self.assertEqual(len(set(sop_uids)), 176)        # unique per slice
            self.assertEqual(len(series_uids), 1)            # one series
            for path in sorted(Path(out_dir).iterdir()):
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                self.assertTrue(UID(ds.SeriesInstanceUID).is_valid)
                self.assertTrue(UID(ds.SOPInstanceUID).is_valid)
                self.assertEqual(
                    ds.file_meta.MediaStorageSOPInstanceUID,
                    ds.SOPInstanceUID,
                )

    def test_write_mask_dicom_stamps_software_version(self):
        # Both outputs overwrite SoftwareVersions, are explicitly marked as
        # derived, and carry readable identity outside their numeric UIDs.
        sw = ["UNITTEST-MODEL", "abc12345", "nnUNetv2 9.9.9"]
        mask = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
        mask[5, 5, 5] = 1
        prob = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            mask_nifti = self._imagedata_marker_nifti(mask, tmp)
            mask_dir = Path(tmp) / "mask"
            dicom_io.write_mask_dicom(mask_nifti, FIXTURE, mask_dir, UID_CONTEXT,
                                      software_versions=sw)
            ds = pydicom.dcmread(str(sorted(mask_dir.iterdir())[0]),
                                 stop_before_pixels=True)
            sv = ds.get("SoftwareVersions")
            sv_values = [sv] if isinstance(sv, str) else list(sv or [])
            self.assertEqual(sv_values, sw)             # overwritten, exact values
            image_type = list(ds.get("ImageType") or [])
            self.assertEqual(image_type[:2], ["DERIVED", "SECONDARY"])
            self.assertEqual(image_type[-1], "MASK")

            prob_nifti = self._sitk_float_nifti_on_grid(prob, tmp)
            vote_dir = Path(tmp) / "vote_map"
            dicom_io.write_prob_dicom(prob_nifti, FIXTURE, vote_dir, UID_CONTEXT,
                                      software_versions=sw)
            ds2 = pydicom.dcmread(str(sorted(vote_dir.iterdir())[0]),
                                  stop_before_pixels=True)
            sv2 = ds2.get("SoftwareVersions")
            sv2_values = [sv2] if isinstance(sv2, str) else list(sv2 or [])
            self.assertEqual(sv2_values, sw)
            image_type2 = list(ds2.get("ImageType") or [])
            self.assertEqual(image_type2[:2], ["DERIVED", "SECONDARY"])
            self.assertEqual(image_type2[-1], "PROBABILITY")
            self.assertIn("foreground probability", ds2.SeriesDescription)
            self.assertIn("stored uint16 value / 65535", ds2.DerivationDescription)
            self.assertNotIn("RescaleSlope", ds2)
            self.assertNotIn("RescaleIntercept", ds2)


class TestProbNpzToNifti(unittest.TestCase):
    def test_extracts_fg_channel_with_seg_geometry(self):
        shape = (4, 5, 6)   # (z, y, x)
        seg = np.zeros(shape, dtype=np.uint8)
        seg[1, 2, 3] = 1
        prob = np.zeros((2, *shape), dtype=np.float32)
        prob[0] = 0.3                       # background channel
        prob[1] = np.linspace(0, 1, int(np.prod(shape)),
                              dtype=np.float32).reshape(shape)  # foreground
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seg_img = sitk.GetImageFromArray(seg)
            seg_img.SetSpacing((0.6, 0.6, 1.1))      # sitk (x, y, z)
            seg_img.SetOrigin((10.0, 20.0, 30.0))
            seg_path = tmp / "VS.nii.gz"
            sitk.WriteImage(seg_img, str(seg_path))
            npz_path = tmp / "VS.npz"
            np.savez_compressed(str(npz_path), probabilities=prob)
            out_path = tmp / "VS_prob.nii.gz"

            dicom_io.prob_npz_to_nifti(npz_path, seg_path, out_path)

            out_img = sitk.ReadImage(str(out_path))
            out_arr = sitk.GetArrayFromImage(out_img)
            self.assertEqual(out_arr.shape, shape)
            self.assertTrue(np.allclose(out_arr, prob[1], atol=1e-6))
            self.assertGreaterEqual(out_arr.min(), 0.0)
            self.assertLessEqual(out_arr.max(), 1.0)
            # geometry copied from the seg image (compare against the seg as it
            # lives on disk: NIfTI persists spacing as float32, so the in-memory
            # seg_img doubles would differ from the round-tripped grid)
            seg_on_disk = sitk.ReadImage(str(seg_path))
            self.assertEqual(out_img.GetSpacing(), seg_on_disk.GetSpacing())
            self.assertEqual(out_img.GetOrigin(), seg_on_disk.GetOrigin())
            self.assertEqual(out_img.GetDirection(), seg_on_disk.GetDirection())


@requires_dicom_fixture
class TestOrientation(unittest.TestCase):
    """Reorientation helpers that make nnU-Net (which ignores the direction
    matrix) see every input in the training orientation."""

    def test_get_orientation_of_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            nifti = Path(tmp) / "VS_0000.nii.gz"
            dicom_io.series_to_nifti(dicom_io.read_series(FIXTURE), nifti)
            # the bundled sagittal MPRAGE fixture is stored PSR
            self.assertEqual(dicom_io.get_orientation(nifti), "PSR")

    def test_reorient_to_lps_and_back_is_lossless(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            native = tmp / "VS_0000.nii.gz"
            dicom_io.series_to_nifti(dicom_io.read_series(FIXTURE), native)
            native_code = dicom_io.get_orientation(native)
            lps = tmp / "lps.nii.gz"
            back = tmp / "back.nii.gz"
            dicom_io.reorient_nifti(native, lps, "LPS")
            self.assertEqual(dicom_io.get_orientation(lps), "LPS")
            dicom_io.reorient_nifti(lps, back, native_code)
            a = sitk.GetArrayFromImage(sitk.ReadImage(str(native)))
            b = sitk.GetArrayFromImage(sitk.ReadImage(str(back)))
            self.assertTrue(np.array_equal(a, b))   # round-trip recovers voxels

    def test_resample_to_reference_matches_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            native = tmp / "VS_0000.nii.gz"
            dicom_io.series_to_nifti(dicom_io.read_series(FIXTURE), native)
            lps = tmp / "lps.nii.gz"
            dicom_io.reorient_nifti(native, lps, "LPS")
            out = tmp / "res.nii.gz"
            dicom_io.resample_to_reference(lps, native, out, is_mask=True)
            ref_img = sitk.ReadImage(str(native))
            out_img = sitk.ReadImage(str(out))
            self.assertEqual(out_img.GetSize(), ref_img.GetSize())
            self.assertEqual(out_img.GetSpacing(), ref_img.GetSpacing())
            self.assertTrue(np.allclose(out_img.GetDirection(),
                                        ref_img.GetDirection(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
