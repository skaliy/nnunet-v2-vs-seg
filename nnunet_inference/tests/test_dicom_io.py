"""Unit tests for nnunet_inference.dicom_io (geometry I/O)."""
from contextlib import contextmanager, nullcontext, redirect_stdout
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pydicom
import SimpleITK as sitk
from imagedata.header import Header
from imagedata.series import Series
from pydicom.uid import UID

from nnunet_inference import dicom_io
from nnunet_inference.tests.synthetic import write_image

DEFAULT_FIXTURE = Path(__file__).resolve().parent / "dicom_input"
FIXTURE = Path(os.environ.get("NNUNET_DICOM_FIXTURE", DEFAULT_FIXTURE))
HAS_DICOM_FIXTURE = FIXTURE.is_dir() and any(FIXTURE.iterdir())
requires_dicom_fixture = unittest.skipUnless(
    HAS_DICOM_FIXTURE,
    "set NNUNET_DICOM_FIXTURE to a local DICOM series to run this test",
)


def fixture_expectations():
    """Shape (z, y, x), spacing (z, y, x) and PatientID, read from the fixture's
    own headers with pydicom, so that any single series can be the fixture."""
    datasets = [pydicom.dcmread(str(path), stop_before_pixels=True)
                for path in sorted(FIXTURE.iterdir()) if path.is_file()]
    first = datasets[0]
    row, column = np.reshape(np.asarray(first.ImageOrientationPatient, dtype=float), (2, 3))
    positions = sorted(float(np.dot(d.ImagePositionPatient, np.cross(row, column))) for d in datasets)
    slice_spacing = float(np.median(np.diff(positions))) if len(positions) > 1 else 1.0
    return ((len(datasets), int(first.Rows), int(first.Columns)),
            (slice_spacing, float(first.PixelSpacing[0]), float(first.PixelSpacing[1])),
            str(first.PatientID))


EXPECTED_SHAPE, EXPECTED_SPACING_ZYX, EXPECTED_PATIENT_ID = (
    fixture_expectations() if HAS_DICOM_FIXTURE else (None, None, None))
UID_CONTEXT = dicom_io.make_uid_context(
    model_name="Synthetic five-fold model",
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
            "Synthetic five-fold model", "a" * 64,
            "checkpoint_final.pth", (0, 1, 2, 3, 4), True,
        )
        changed_tta = dicom_io.make_derived_dicom_uid(
            tta_context, "source-series", "segmentation"
        )
        changed_axes = dicom_io.make_derived_dicom_uid(
            {**tta_context, "tta_mirror_axes": [0]}, "source-series", "segmentation"
        )

        self.assertEqual(mask, repeated)
        self.assertEqual(
            len({mask, probability, instance, changed_tta, changed_axes}), 5
        )
        for value in (mask, probability, instance, changed_tta, changed_axes):
            self.assertTrue(UID(value).is_valid, value)
            self.assertLessEqual(len(value), 64)
            self.assertRegex(value, r"^2\.25\.[0-9]+$")


@requires_dicom_fixture
class TestInputBridge(unittest.TestCase):
    def test_read_series_shape_spacing_patient(self):
        series = dicom_io.read_series(FIXTURE)
        self.assertEqual(tuple(series.shape), EXPECTED_SHAPE)
        self.assertEqual(series.slices, EXPECTED_SHAPE[0])
        self.assertEqual(series.patientID, EXPECTED_PATIENT_ID)
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
            self.assertEqual(img.GetSize(), EXPECTED_SHAPE[::-1])   # sitk (x, y, z)
            sx, sy, sz = img.GetSpacing()
            for got, exp in zip((sz, sy, sx), EXPECTED_SPACING_ZYX):
                self.assertAlmostEqual(got, exp, places=3)

    def test_series_to_nifti_rejects_bad_name(self):
        series = dicom_io.read_series(FIXTURE)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dicom_io.series_to_nifti(series, Path(tmp) / "VS.nii.gz")


class TestVoteMapValues(unittest.TestCase):
    def test_blue_is_exactly_the_mask(self):
        probability = np.array([0.0, 0.2, 0.49999, 0.5, 0.9, 0.0, 0.3, 0.6, 1.0])
        inside = np.array([False] * 5 + [True] * 4)
        stored = dicom_io.vote_map_values(probability, inside)
        self.assertEqual(stored.dtype, np.uint16)
        np.testing.assert_array_equal(stored >= dicom_io.VOTE_MAP_BLUE, inside)
        # Values already on the right side of the threshold are kept, and an
        # outside zero stays transparent.
        self.assertEqual(stored[:2].tolist(), [0, 13107])
        self.assertEqual(stored[-1], 65535)
        self.assertEqual(stored[5], dicom_io.VOTE_MAP_BLUE)
        self.assertEqual(stored[4], dicom_io.VOTE_MAP_BLUE - 1)
        # pr2mask draws blue from votemapagree * votemapmax up.
        self.assertGreaterEqual(dicom_io.VOTE_MAP_BLUE, 0.5 * dicom_io.VOTE_MAP_MAX)
        self.assertLess(dicom_io.VOTE_MAP_BLUE - 1, 0.5 * dicom_io.VOTE_MAP_MAX)


@contextmanager
def imagedata_397_header_copies():
    """Copy Series headers as imagedata 3.9.7, the image's version, does.

    In 3.9.7, Header.add_template carries SOPInstanceUIDs over only from a
    header with ``keep_uid``, which a Series read from disk does not have, and
    Series.SOPInstanceUIDs makes up new UIDs when none are set. 3.9.6 copies
    them always. Under 3.9.7 this behaves as imagedata itself does for these
    single-tag stacks, so the tests that use it mean the same in both versions.
    """
    add_template = Header.add_template

    def add_template_397(self, template):
        kept = getattr(self, "SOPInstanceUIDs", None)
        add_template(self, template)
        if template is not None and "keep_uid" not in template.__dict__:
            self.SOPInstanceUIDs = kept

    def sop_instance_uids_397(series):
        if series.header.SOPInstanceUIDs is None:
            series.header.SOPInstanceUIDs = {(0, index): series.header.new_uid()
                                             for index in range(series.slices)}
        return series.header.SOPInstanceUIDs

    with mock.patch.object(Header, "add_template", add_template_397), \
            mock.patch.object(Series, "SOPInstanceUIDs",
                              property(sop_instance_uids_397, Series.SOPInstanceUIDs.fset)):
        yield


class TestDerivedValues(unittest.TestCase):
    """The vote map and the mask store exactly the values we compute, as unsigned
    16-bit without rescale, whatever the source's pixel representation."""

    SOURCES = {
        "plain": {},
        "rescaled": dict(RescaleSlope=2.5, RescaleIntercept=0.0),
        "signed": dict(PixelRepresentation=1),
    }
    VERSIONS = ["Synthetic five-fold model", "aaaaaaaa", "nnUNetv2 test"]

    def _source(self, root, tags, probability, mask):
        """Write a four-slice source with `tags` and the prediction NIfTIs on its grid."""
        source = root / "source"
        for index in range(4):
            write_image(source / f"{index}.dcm", index, sop_uid=f"1.2.3.4.{index + 1}",
                        pixels=(np.arange(36).reshape(6, 6) * 7 + index) % 300, **tags)
        native = root / "native" / "VS_0000.nii.gz"
        dicom_io.series_to_nifti(dicom_io.read_series(source), native)
        reference = sitk.ReadImage(str(native))
        paths = {}
        for name, arr in (("prob", probability), ("mask", mask)):
            image = sitk.GetImageFromArray(arr)
            image.CopyInformation(reference)
            paths[name] = root / f"{name}.nii.gz"
            sitk.WriteImage(image, str(paths[name]))
        return source, paths

    def _derive(self, source, paths, out_root):
        """Write our vote map and mask of `source` under `out_root`."""
        dicom_io.write_prob_dicom(paths["prob"], source, out_root / "vote_map", UID_CONTEXT,
                                  mask_nifti=paths["mask"], software_versions=self.VERSIONS)
        dicom_io.write_mask_dicom(paths["mask"], source, out_root / "mask", UID_CONTEXT,
                                  software_versions=self.VERSIONS)

    def _write(self, root, tags, probability, mask):
        """Write a four-slice source with `tags`, then our vote map and mask."""
        source, paths = self._source(root, tags, probability, mask)
        self._derive(source, paths, root)
        # The write path reads both NIfTIs with imagedata; so do the expectations.
        expected_votes = dicom_io.vote_map_values(np.asarray(Series(str(paths["prob"]))),
                                                  np.asarray(Series(str(paths["mask"]))) > 0)
        expected_mask = (np.asarray(Series(str(paths["mask"]))) > 0).astype(np.uint16)
        return source, expected_votes, expected_mask

    @staticmethod
    def _position(dataset):
        return tuple(float(v) for v in dataset.ImagePositionPatient)

    def _expected_sop_uids(self, source, output_kind):
        """Derived SOP UID by slice position, read from the source files alone.

        Each comes from the SOP UID of the source slice at that position and the
        slice's index along the slice normal (imagedata's slice order)."""
        datasets = [pydicom.dcmread(str(p), stop_before_pixels=True)
                    for p in Path(source).iterdir()]
        row, column = np.reshape(np.asarray(datasets[0].ImageOrientationPatient, float), (2, 3))
        normal = np.cross(row, column)
        datasets.sort(key=lambda d: float(np.dot(np.asarray(d.ImagePositionPatient, float),
                                                 normal)))
        return {self._position(d): dicom_io.make_derived_dicom_uid(
                    UID_CONTEXT, str(d.SeriesInstanceUID), output_kind,
                    source_sop_uid=str(d.SOPInstanceUID), slice_index=index)
                for index, d in enumerate(datasets)}

    def _sop_uids(self, out_dir):
        """SOP UID by slice position of a written series."""
        datasets = [pydicom.dcmread(str(p), stop_before_pixels=True)
                    for p in Path(out_dir).iterdir()]
        for d in datasets:
            self.assertEqual(str(d.file_meta.MediaStorageSOPInstanceUID), str(d.SOPInstanceUID))
        return {self._position(d): str(d.SOPInstanceUID) for d in datasets}

    def _stored(self, source, out_dir, output_kind):
        """Check the written headers and return the stored values (z, y, x)."""
        datasets = [pydicom.dcmread(str(p)) for p in sorted(Path(out_dir).iterdir())]
        source_positions = {tuple(float(v) for v in pydicom.dcmread(str(p)).ImagePositionPatient)
                            for p in Path(source).iterdir()}
        self.assertEqual({tuple(float(v) for v in d.ImagePositionPatient) for d in datasets},
                         source_positions)
        # Deterministic and bound to the source instances.
        self.assertEqual(self._sop_uids(out_dir), self._expected_sop_uids(source, output_kind))
        for d in datasets:
            self.assertEqual(d.PixelRepresentation, 0)
            self.assertEqual(d.BitsAllocated, 16)
            self.assertEqual(float(d.get("RescaleSlope", 1)), 1.0)
            self.assertEqual(float(d.get("RescaleIntercept", 0)), 0.0)
            self.assertEqual(list(d.SoftwareVersions), self.VERSIONS)
            self.assertIn("model_version=" + "a" * 64, d.DerivationDescription)
            self.assertEqual(str(d.StudyInstanceUID), "1.2.3")
            self.assertEqual(str(d.FrameOfReferenceUID), "1.2.3.5")
            # Deterministic: derived from the source series UID, not a new one.
            self.assertEqual(str(d.SeriesInstanceUID), dicom_io.make_derived_dicom_uid(
                UID_CONTEXT, "1.2.3.4", output_kind))
        # Without a rescale, imagedata returns the stored values in its slice order.
        stored = np.asarray(dicom_io.read_series(out_dir))
        self.assertEqual(stored.dtype, np.uint16)
        return stored

    def test_vote_map_and_mask_store_the_computed_values(self):
        shape = (4, 6, 6)
        probability = np.zeros(shape, dtype=np.float32)
        probability[:, :2] = 0.8
        probability[:, 2:4] = 0.3
        probability[1, 4:, :3] = 0.62
        mask = (probability > 0.5).astype(np.uint8)
        mask[1, 4:, :3] = 0        # a confident voxel outside the mask
        mask[2, 5, 5] = 1          # an uncertain voxel inside it
        for label, tags in self.SOURCES.items():
            with self.subTest(source=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, expected_votes, expected_mask = self._write(root, tags, probability, mask)
                self.assertEqual(sorted(np.unique(expected_votes).tolist()),
                                 [0, 19660, 32767, 32768, 52428])
                votes = self._stored(source, root / "vote_map", "probability")
                np.testing.assert_array_equal(votes, expected_votes)
                stored_mask = self._stored(source, root / "mask", "segmentation")
                np.testing.assert_array_equal(stored_mask, expected_mask)
                self.assertEqual(sorted(np.unique(stored_mask).tolist()), [0, 1])

    def test_empty_mask_with_uncertain_probabilities_stays_below_blue(self):
        shape = (4, 6, 6)
        probability = np.zeros(shape, dtype=np.float32)
        probability[:, :3] = 0.45
        probability[:, 3:, :2] = 0.2
        mask = np.zeros(shape, dtype=np.uint8)
        for label, tags in self.SOURCES.items():
            with self.subTest(source=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, expected_votes, _ = self._write(root, tags, probability, mask)
                votes = self._stored(source, root / "vote_map", "probability")
                np.testing.assert_array_equal(votes, expected_votes)
                self.assertEqual(int(votes.max()), 29491)
                self.assertEqual(int(self._stored(source, root / "mask", "segmentation").max()), 0)

    def test_sop_uids_come_from_the_source_instances_and_repeat(self):
        # Converting the template to uint16 copies its header. imagedata 3.9.7
        # leaves the SOP UIDs out of that copy and then makes up new ones, so
        # each subtest also runs with 3.9.7's header copies.
        shape = (4, 6, 6)
        probability = np.zeros(shape, dtype=np.float32)
        probability[1:3, 2:4, 2:4] = 0.9
        mask = (probability > 0.5).astype(np.uint8)
        for label, tags in self.SOURCES.items():
            for copies in ("installed imagedata", "imagedata 3.9.7 header copies"):
                with self.subTest(source=label, copies=copies), \
                        tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    emulate = copies != "installed imagedata"
                    with imagedata_397_header_copies() if emulate else nullcontext():
                        source, paths = self._source(root, tags, probability, mask)
                        for run in ("first", "second"):
                            self._derive(source, paths, root / run)
                    for kind, name in (("probability", "vote_map"), ("segmentation", "mask")):
                        first = self._sop_uids(root / "first" / name)
                        self.assertEqual(first, self._expected_sop_uids(source, kind))
                        self.assertEqual(self._sop_uids(root / "second" / name), first)


AXIAL = (1, 0, 0, 0, 1, 0)
SAGITTAL = (0, 1, 0, 0, 0, -1)


def write_stack(directory, count, *, step=1.0, orientation=AXIAL, skip=(), **tags):
    """Write `count` slices `step` mm apart along the slice normal, leaving out
    the indices in `skip`; return (position, pixels, path) for each written slice."""
    row, column = np.asarray(orientation[:3], float), np.asarray(orientation[3:], float)
    normal = np.cross(row, column)
    rng = np.random.default_rng(count)
    slices = []
    for index in range(count):
        pixels = rng.integers(0, 1000, (4, 5))
        if index in skip:
            continue
        position = np.array([3.0, -4.0, 5.0]) + index * step * normal
        path = write_image(Path(directory) / f"{index:03d}.dcm", index, orientation=orientation,
                           position=[f"{v:.12g}" for v in position], pixels=pixels, **tags)
        slices.append((position, pixels, path))
    return slices


class TestSliceStack(unittest.TestCase):
    """The stack given to nnU-Net is complete, evenly spaced, and spaced as its
    slice positions are, whatever SliceThickness or SpacingBetweenSlices say."""

    def check(self, directory):
        series = dicom_io.read_series(directory)
        output = io.StringIO()
        with redirect_stdout(output):
            dicom_io.check_slice_stack(series, directory)
        return series, output.getvalue()

    def physical_mismatches(self, slices, series, orientation, tmp):
        """Source pixels not found at their physical position in the NIfTI."""
        nifti = Path(tmp) / "nifti" / "VS_0000.nii.gz"
        dicom_io.series_to_nifti(series, nifti)
        image = sitk.ReadImage(str(nifti))
        voxels = sitk.GetArrayFromImage(image)
        row, column = np.asarray(orientation[:3], float), np.asarray(orientation[3:], float)
        bad = 0
        for position, pixels, _ in slices:
            for r, c in np.ndindex(pixels.shape):
                point = position + c * row + r * column     # PixelSpacing (1, 1)
                index = image.TransformPhysicalPointToIndex(tuple(float(v) for v in point))
                inside = all(0 <= i < n for i, n in zip(index, image.GetSize()))
                bad += not (inside and voxels[index[::-1]] == pixels[r, c])
        return bad, image.GetSpacing()[2]

    def test_slice_spacing_comes_from_the_positions(self):
        # No SpacingBetweenSlices: imagedata takes SliceThickness as the spacing.
        for orientation, step, thickness in ((AXIAL, 0.5, 1.0), (SAGITTAL, -3.3, 3.0)):
            with self.subTest(step=step, thickness=thickness), \
                    tempfile.TemporaryDirectory() as tmp:
                slices = write_stack(Path(tmp) / "in", 9, step=step, orientation=orientation,
                                     SliceThickness=thickness)
                series, printed = self.check(Path(tmp) / "in")
                self.assertAlmostEqual(float(series.spacing[0]), abs(step), places=6)
                self.assertEqual(len(printed.strip().splitlines()), 1)
                self.assertIn(f"{thickness:g}", printed)
                self.assertIn(f"{abs(step):g}", printed)
                bad, spacing = self.physical_mismatches(slices, series, orientation, tmp)
                self.assertEqual(bad, 0)
                self.assertAlmostEqual(spacing, abs(step), places=5)

    def test_wrong_spacing_between_slices_is_corrected(self):
        for spacing_between_slices in (2.0, -1.0):
            with self.subTest(spacing_between_slices=spacing_between_slices), \
                    tempfile.TemporaryDirectory() as tmp:
                slices = write_stack(Path(tmp) / "in", 6, step=1.0, SliceThickness=1.0,
                                     SpacingBetweenSlices=spacing_between_slices)
                series, printed = self.check(Path(tmp) / "in")
                self.assertIn(f"{spacing_between_slices:g}", printed)
                bad, spacing = self.physical_mismatches(slices, series, AXIAL, tmp)
                self.assertEqual(bad, 0)
                self.assertAlmostEqual(spacing, 1.0, places=5)

    def test_regular_stack_is_left_exactly_as_read(self):
        # Like a Siemens series: thickness and position step differ by 1e-8.
        with tempfile.TemporaryDirectory() as tmp:
            write_stack(Path(tmp) / "in", 7, step=1.1000000347, orientation=SAGITTAL,
                        SliceThickness="1.1000000238")
            series = dicom_io.read_series(Path(tmp) / "in")
            before = np.array(series.spacing)
            matrix = series.transformationMatrix    # setting the spacing would discard it
            output = io.StringIO()
            with redirect_stdout(output):
                dicom_io.check_slice_stack(series, Path(tmp) / "in")
            self.assertEqual(np.array(series.spacing).tobytes(), before.tobytes())
            self.assertIs(series.header.transformationMatrix, matrix)
            self.assertEqual(output.getvalue(), "")

    def test_single_slice_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_stack(Path(tmp) / "in", 1, SliceThickness=3.0)
            series = dicom_io.read_series(Path(tmp) / "in")
            before = np.array(series.spacing)
            output = io.StringIO()
            with redirect_stdout(output):
                dicom_io.check_slice_stack(series, Path(tmp) / "in")
            self.assertEqual(np.array(series.spacing).tobytes(), before.tobytes())
            self.assertEqual(output.getvalue(), "")

    def test_missing_second_slice_is_refused(self):
        # imagedata accepts this stack: only a gap larger than the first one fails there.
        with tempfile.TemporaryDirectory() as tmp:
            write_stack(Path(tmp) / "in", 10, skip=(1,), SliceThickness=1.0)
            series = dicom_io.read_series(Path(tmp) / "in")
            self.assertEqual(series.slices, 9)
            with self.assertRaisesRegex(ValueError, r"not evenly spaced.* 1(\.0+)? .* 2(\.0+)? mm"):
                dicom_io.check_slice_stack(series, Path(tmp) / "in")

    def test_undecodable_last_slice_is_refused(self):
        # imagedata skips a file whose pixels it cannot decode.
        with tempfile.TemporaryDirectory() as tmp:
            slices = write_stack(Path(tmp) / "in", 10, SliceThickness=1.0)
            last = pydicom.dcmread(str(slices[-1][2]))
            last.PixelData = last.PixelData[:len(last.PixelData) // 2]
            last.save_as(str(slices[-1][2]))
            series = dicom_io.read_series(Path(tmp) / "in")
            self.assertEqual(series.slices, 9)
            with self.assertRaisesRegex(ValueError, r"\b10\b.*\b9\b"):
                dicom_io.check_slice_stack(series, Path(tmp) / "in")


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

    def _sitk_float_nifti_on_grid(self, arr, tmp, name="prob.nii.gz"):
        """Write a float `arr` (z, y, x) as a NIfTI on the fixture's grid via
        SimpleITK — mirrors how prob_npz_to_nifti produces the probability NIfTI
        that the write path consumes."""
        series = dicom_io.read_series(FIXTURE)
        ref = Path(tmp) / "ref_0000.nii.gz"
        dicom_io.series_to_nifti(series, ref)
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(sitk.ReadImage(str(ref)))
        out = Path(tmp) / name
        sitk.WriteImage(img, str(out))
        return out

    def test_write_mask_dicom_preserves_orientation_and_values(self):
        # asymmetric marker in a known (low-z, low-y, low-x) corner
        marker = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
        marker[tuple(slice(0, max(1, size // 4)) for size in EXPECTED_SHAPE)] = 1
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
        # uniform value regions; verified by value (orientation-independent)
        third = EXPECTED_SHAPE[0] // 3      # the fixture needs at least three slices
        prob = np.full(EXPECTED_SHAPE, 0.25, dtype=np.float32)
        prob[:third] = 1.0
        prob[third:2 * third] = 0.0
        mask = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
        mask[:third] = 1
        with tempfile.TemporaryDirectory() as tmp:
            nifti = self._sitk_float_nifti_on_grid(prob, tmp)
            mask_nifti = self._sitk_float_nifti_on_grid(mask, tmp, "mask.nii.gz")
            out_dir = Path(tmp) / "vote_map"
            dicom_io.write_prob_dicom(nifti, FIXTURE, out_dir, UID_CONTEXT,
                                      mask_nifti=mask_nifti)
            rb = np.asarray(dicom_io.read_series(out_dir))
            self.assertEqual(rb.dtype, np.uint16)
            values, counts = np.unique(rb, return_counts=True)
            # 1.0 -> 65535, 0.0 -> 0, 0.25 -> 16384
            self.assertEqual(values.tolist(), [0, 16384, 65535])
            self.assertEqual(int(counts[-1]), int(mask.sum()))

    def test_save_series_pred_derives_unique_uids(self):
        mask = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
        mask[tuple(size // 2 for size in EXPECTED_SHAPE)] = 1
        with tempfile.TemporaryDirectory() as tmp:
            nifti = self._imagedata_marker_nifti(mask, tmp)
            out_dir = Path(tmp) / "mask"
            dicom_io.write_mask_dicom(nifti, FIXTURE, out_dir, UID_CONTEXT)
            sop_uids, series_uids = [], set()
            for f in sorted(Path(out_dir).iterdir()):
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                sop_uids.append(ds.SOPInstanceUID)
                series_uids.add(ds.SeriesInstanceUID)
            self.assertEqual(len(sop_uids), EXPECTED_SHAPE[0])
            self.assertEqual(len(set(sop_uids)), EXPECTED_SHAPE[0])   # unique per slice
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
        mask[tuple(size // 2 for size in EXPECTED_SHAPE)] = 1
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
            tta_context = {**UID_CONTEXT, "tta": True, "tta_mirror_axes": [0, 1]}
            dicom_io.write_prob_dicom(prob_nifti, FIXTURE, vote_dir, tta_context,
                                      mask_nifti=mask_nifti, software_versions=sw)
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
            self.assertIn("tta=on (original + one flip along axes 0,1)", ds2.DerivationDescription)
            self.assertNotIn("RescaleSlope", ds2)
            self.assertNotIn("RescaleIntercept", ds2)


class TestTagPr2maskMask(unittest.TestCase):
    def write_series(self, directory, series_uid, *, pixels=None, **tags):
        for index in range(2):
            path = write_image(Path(directory) / f"{index}.dcm", index,
                               series_uid=series_uid,
                               sop_uid=f"{series_uid}.{index + 1}", pixels=pixels)
            dataset = pydicom.dcmread(str(path))
            for keyword, value in tags.items():
                setattr(dataset, keyword, value)
            dataset.save_as(str(path))

    def test_copies_nnunet_provenance_and_keeps_pr2mask_identity(self):
        mask_pixels = np.array([[0, 1, 0], [1, 0, 0]])
        with tempfile.TemporaryDirectory() as tmp:
            ours, labels = Path(tmp) / "mask", Path(tmp) / "labels"
            self.write_series(
                ours, "2.25.1",
                SoftwareVersions=["ResEnc-L", "abcd1234", "nnUNetv2 2.6.2"],
                SeriesDescription="nnU-Net ResEnc-L segmentation mask",
                DerivationDescription="nnU-Net segmentation mask; model_version=abc",
                ImageType=["DERIVED", "SECONDARY", "M", "ND", "MASK"])
            self.write_series(
                labels, "1.3.6.1.4.1.45037.1", pixels=mask_pixels,
                StudyID="1.2.3", WindowCenter="0.5", WindowWidth="1",
                SoftwareVersions="syngo MR A35",
                SeriesDescription="t1_mpr_ns_sag (mask)",
                ImageType=["DERIVED", "SECONDARY", "OTHER"])

            dicom_io.tag_pr2mask_mask(labels, ours)
            written = [pydicom.dcmread(str(path)) for path in sorted(labels.iterdir())]

        self.assertEqual(len(written), 2)
        for index, dataset in enumerate(written):
            self.assertEqual(list(dataset.SoftwareVersions),
                             ["ResEnc-L", "abcd1234", "nnUNetv2 2.6.2"])
            self.assertEqual(dataset.SeriesDescription, "nnU-Net ResEnc-L segmentation mask")
            self.assertIn("model_version=abc", dataset.DerivationDescription)
            self.assertEqual(list(dataset.ImageType),
                             ["DERIVED", "SECONDARY", "OTHER", "MASK"])
            self.assertEqual(str(dataset.StudyID), "1.2.3")
            self.assertEqual(float(dataset.WindowCenter), 0.5)
            self.assertEqual(float(dataset.WindowWidth), 1.0)
            self.assertEqual(str(dataset.SeriesInstanceUID), "1.3.6.1.4.1.45037.1")
            self.assertEqual(str(dataset.SOPInstanceUID), f"1.3.6.1.4.1.45037.1.{index + 1}")
            np.testing.assert_array_equal(dataset.pixel_array, mask_pixels)

    def test_missing_provenance_or_empty_series_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ours, labels, empty = Path(tmp) / "mask", Path(tmp) / "labels", Path(tmp) / "empty"
            empty.mkdir()
            self.write_series(ours, "2.25.1", SeriesDescription="nnU-Net mask")
            self.write_series(labels, "1.3.6.1.4.1.45037.1")
            with self.assertRaisesRegex(RuntimeError, "lacks provenance"):
                dicom_io.tag_pr2mask_mask(labels, ours)
            with self.assertRaisesRegex(RuntimeError, "pr2mask mask series is empty"):
                dicom_io.tag_pr2mask_mask(empty, ours)
            with self.assertRaisesRegex(RuntimeError, "nnU-Net mask series is empty"):
                dicom_io.tag_pr2mask_mask(labels, empty)


class TestProbNpzToNifti(unittest.TestCase):
    def test_extracts_fg_channel_with_seg_geometry(self):
        shape = (4, 5, 6)   # (z, y, x)
        seg = np.zeros(shape, dtype=np.uint8)
        seg[1, 2, 3] = 1
        prob = np.zeros((2, *shape), dtype=np.float32)
        prob[0] = 0.3                       # background channel
        prob[1] = np.linspace(0, 1, int(np.prod(shape)),
                              dtype=np.float32).reshape(shape)  # foreground
        prob[0] = 1 - prob[1]
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
            # One letter for each anatomical axis, in the fixture's storage order.
            code = dicom_io.get_orientation(nifti)
            self.assertEqual(sorted(code.translate(str.maketrans("RAI", "LPS"))),
                             ["L", "P", "S"], code)

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
