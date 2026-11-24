"""validate_output.py must compare the published series and REDCap rows with the input."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import numpy as np
from pydicom import dcmread

from nnunet_inference.integration import validate_output
from nnunet_inference.redcap_output import pr2mask_dictionary_rows
from nnunet_inference.tests.synthetic import write_image

BUNDLE = "e91f17a3af1621a773a562ba12e426404322119234b91195fc8ef9382b0e79cc"   # block 100
SERIES = {"fused": "1.2.20", "fused_vote_map": "1.2.21", "labels": "1.2.22"}
SLICES = 3


def write_derived(path, index, series_uid, *, mask, **arguments):
    """One slice of a derived series, as the container writes it."""
    write_image(path, index, series_uid=series_uid, sop_uid=f"{series_uid}.{index + 1}", **arguments)
    dataset = dcmread(path)
    dataset.StudyID = "1"
    if mask:
        dataset.ImageType = ["DERIVED", "SECONDARY", "MASK"]
        dataset.SeriesDescription = "nnU-Net VS mask"
        dataset.DerivationDescription = "model_version=test"
    dataset.save_as(path)


class ValidateOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "input"
        for index in range(SLICES):
            write_image(self.input / f"{index}.dcm", index, pixels=np.full((4, 4), 100 + index))

    def build(self, *, foreground=0, record_id="SYNTHETIC", event="test_arm_1", missing=(),
              **changes):
        """A complete output of a run on the input. `changes` maps an output name to a
        function of the slice index giving the write_image arguments that differ, and
        `missing` lists (output, slice index) pairs that are not written. It replaces
        the output of an earlier call."""
        output = self.root / "output"
        shutil.rmtree(output, ignore_errors=True)
        for name, series_uid in SERIES.items():
            for index in range(SLICES):
                if (name, index) in missing:
                    continue
                pixels = np.zeros((4, 4), dtype=np.uint16)
                if name == "labels" and index == 0:
                    pixels[0, :foreground] = 1
                arguments = dict(pixels=pixels) | changes.get(name, lambda index: {})(index)
                write_derived(output / name / f"{index}.dcm", index, series_uid,
                              mask=name == "labels", **arguments)
        (output / "reports").mkdir()
        if foreground:
            write_derived(output / "reports" / "0.dcm", 0, "1.2.23", mask=False,
                          pixels=np.zeros((4, 4)))
        redcap = output / "redcap" / SERIES["labels"]
        redcap.mkdir(parents=True)
        columns = [row["Variable / Field Name"] for row in pr2mask_dictionary_rows()]
        columns.append("model_version")
        values = dict.fromkeys(columns, "") | dict(model_version="nnunet_large 5-fold, test")
        if foreground:
            values["number_of_pixel"] = str(foreground)
        (redcap / "output.json").write_text(json.dumps([
            dict(record_id=record_id, redcap_event_name=event, redcap_repeat_instrument="pr2mask",
                 redcap_repeat_instance="101", field_name=column, value=values[column])
            for column in columns]))
        dictionary = io.StringIO()
        writer = csv.writer(dictionary)
        writer.writerow(["Variable / Field Name", "Form Name", "Field Type", "Field Label"])
        writer.writerows([column, "pr2mask", "text", column] for column in columns)
        with zipfile.ZipFile(redcap / "output_data_dictionary.zip", "w") as archive:
            archive.writestr("instrument.csv", dictionary.getvalue())
        (output / "runtime_resources.json").write_text(json.dumps(dict(bundle_sha256=BUNDLE)))
        return output

    def validate(self, output):
        printed = io.StringIO()
        with redirect_stdout(printed):
            validate_output.validate(self.input, output)
        return printed.getvalue()

    def test_correct_output_passes(self):
        for foreground in (0, 2):
            with self.subTest(foreground=foreground):
                self.assertIn("validation: OK", self.validate(self.build(foreground=foreground)))

    def test_shifted_mask_fails(self):
        output = self.build(labels=lambda index: dict(position=(50, 50, 50 + index)))
        with self.assertRaisesRegex(RuntimeError, r"^labels slice 1 .*ImagePositionPatient"):
            self.validate(output)

    def test_mask_with_other_rows_and_columns_fails(self):
        output = self.build(labels=lambda index: dict(pixels=np.zeros((8, 8))))
        with self.assertRaisesRegex(RuntimeError, r"^labels slice 1 .*Rows 8 != 4.*Columns 8 != 4"):
            self.validate(output)

    def test_mask_with_other_spacing_orientation_or_frame_of_reference_fails(self):
        for key, change in (
                ("PixelSpacing", dict(pixel_spacing=(2, 2))),
                ("ImageOrientationPatient", dict(orientation=(1, 0, 0, 0, 0, -1)))):
            with self.subTest(key=key):
                output = self.build(labels=lambda index: change)
                with self.assertRaisesRegex(RuntimeError, rf"^labels slice 1 .*{key}"):
                    self.validate(output)
        output = self.build()
        for path in (output / "labels").iterdir():
            dataset = dcmread(path)
            dataset.FrameOfReferenceUID = "9.9.9"
            dataset.save_as(path)
        with self.assertRaisesRegex(RuntimeError, r"^labels slice 1 .*FrameOfReferenceUID"):
            self.validate(output)

    def test_fused_series_with_a_missing_slice_fails(self):
        for name in ("fused", "fused_vote_map"):
            with self.subTest(name=name):
                output = self.build(missing={(name, 1)})
                with self.assertRaisesRegex(RuntimeError, rf"^{name} slice count 2 != input 3"):
                    self.validate(output)

    def test_slices_are_paired_by_position_not_by_file_order(self):
        output = self.build(foreground=2, labels=lambda index: dict(position=(0, 0, SLICES - 1 - index)))
        self.assertIn("validation: OK", self.validate(output))

    def test_redcap_rows_must_go_to_the_input_record_and_event(self):
        for arguments, field in ((dict(record_id="OTHER"), "record_id"),
                                 (dict(record_id=""), "record_id"),
                                 (dict(event="EventName:test_arm_1"), "redcap_event_name")):
            with self.subTest(**arguments):
                output = self.build(**arguments)
                with self.assertRaisesRegex(RuntimeError, rf"REDCap {field} differs from the input"):
                    self.validate(output)


if __name__ == "__main__":
    unittest.main()
