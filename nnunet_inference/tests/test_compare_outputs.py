"""The run comparison must ignore what differs between two runs of one image
and report a changed, missing or added pixel, tag, file or REDCap value in a candidate."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np
from pydicom import dcmread
from pydicom.dataelem import DataElement
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

from nnunet_inference.integration import compare_outputs
from nnunet_inference.tests.synthetic import write_image

ROW = "redcap/0#SYNTHETIC/test_arm_1/pr2mask/101"
FIELDS = ("number_of_pixel", "model_version", "mask_base64_gzip_compressed", "boundingbox")


def write_run(root, *, prefix, content_time, memory_peak, pixel=1, model_version="model, image:1",
              address="0x762804002f18", size="[3, 1, 1]", transfer_syntax=ExplicitVRLittleEndian,
              row_extra=None, dictionary="Variable / Field Name\nnumber_of_pixel\n", stray=None):
    """A small container output: three mask slices, REDCap rows and resources.
    pr2mask writes a memory address into the bounding box of a region."""
    root = Path(root)
    for index in range(3):
        pixels = np.zeros((2, 3), dtype=np.uint16)
        pixels[0, index] = pixel
        path = write_image(root / "labels" / "1.2.8" / f"{prefix}{2 - index}.dcm", index,
                           series_uid="1.2.8", pixels=pixels)
        dataset = dcmread(path)
        dataset.InstanceNumber = index + 1
        dataset.ContentTime = content_time
        dataset.file_meta.TransferSyntaxUID = transfer_syntax
        dataset.save_as(path)
    report = root / "redcap" / "1.2.9"
    report.mkdir(parents=True)
    bounding_box = f"ImageRegion ({address})\n  Dimension: 3\n  Index: [0, 0, 0]\n  Size: {size}\n"
    rows = [dict(record_id="SYNTHETIC", redcap_event_name="test_arm_1",
                 redcap_repeat_instrument="pr2mask", redcap_repeat_instance="101",
                 field_name=name, value=value) | (row_extra or {})
            for name, value in zip(FIELDS, ("3", model_version, "m" * 500, bounding_box))]
    (report / "output.json").write_text(json.dumps(rows))
    with zipfile.ZipFile(report / "output_data_dictionary.zip", "w") as archive:
        archive.writestr("instrument.csv", dictionary)
    if stray:
        (report / stray).write_text("not part of the release\n")
    (root / "runtime_resources.json").write_text(json.dumps(
        dict(tta=True, cgroup_memory_peak_bytes=memory_peak, bundle_sha256="a" * 64)))
    (root / "pacs_command.log").write_text(f"finished at {content_time}\n")
    return root


def run_command(argv):
    """Exit status, standard output and standard error of the command line."""
    output, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        status = compare_outputs.main(argv)
    return status, output.getvalue(), errors.getvalue()


class CompareOutputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        first = compare_outputs.snapshot(write_run(
            self.root / "first", prefix="a", content_time="101010", memory_peak=1))
        second = compare_outputs.snapshot(write_run(
            self.root / "second", prefix="b", content_time="111111", memory_peak=2,
            address="0x7f3a1c00be40"))
        self.baseline, self.dropped, self.in_both = compare_outputs.stable(first, second)

    def candidate(self, **changes):
        settings = dict(prefix="c", content_time="121212", memory_peak=3,
                        address="0x55d0c0ffee00") | changes
        folder = Path(tempfile.mkdtemp(prefix="candidate-", dir=self.root))
        return compare_outputs.snapshot(write_run(folder, **settings))

    def check(self, candidate):
        return compare_outputs.check(self.baseline, candidate, self.dropped, self.in_both)

    def test_stable_keeps_pixels_and_values_and_drops_what_varies(self):
        self.assertEqual(self.baseline["labels#files"], "3")
        self.assertEqual(len([k for k in self.baseline if k.endswith("#pixels")]), 3)
        self.assertIn("labels/0000#SOPInstanceUID", self.baseline)
        self.assertIn(f"{ROW}/number_of_pixel", self.baseline)
        self.assertEqual(self.baseline["redcap/0#rows"], "4")
        self.assertEqual(self.baseline["runtime_resources#tta"], "true")
        self.assertEqual(self.dropped, [
            "labels/0000#ContentTime", "labels/0000#path",
            "labels/0001#ContentTime", "labels/0001#path",
            "labels/0002#ContentTime", "labels/0002#path",
            "runtime_resources#cgroup_memory_peak_bytes"])
        self.assertEqual(self.in_both, self.dropped)

    def test_dicom_file_meta_is_compared(self):
        self.assertEqual(self.baseline["labels/0000#meta/TransferSyntaxUID"],
                         json.dumps([ExplicitVRLittleEndian]))
        different = self.check(self.candidate(transfer_syntax=ImplicitVRLittleEndian))
        self.assertIn("labels/0000#meta/TransferSyntaxUID", different)

    def test_every_file_is_listed_and_the_ones_compared_nowhere_else_are_hashed(self):
        self.assertEqual(self.baseline["#file/pacs_command.log"], "present")
        self.assertEqual(self.baseline["#file/runtime_resources.json"], "present")
        self.assertEqual(self.baseline["redcap/0#file/output.json"], "present")
        self.assertEqual(self.baseline["redcap/0#file/output_data_dictionary.zip"], "present")
        self.assertFalse([key for key in self.baseline if key.startswith("labels") and "#file/" in key])
        candidate = self.root / "candidate"
        write_run(candidate, prefix="c", content_time="121212", memory_peak=3)
        (candidate / "extra").mkdir()
        (candidate / "extra" / "notes.txt").write_text("stray\n")
        entries = compare_outputs.snapshot(candidate)
        self.assertRegex(entries["#file/extra/notes.txt"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(self.check(entries), ["#entries", "#file/extra/notes.txt"])

    def test_a_stray_file_in_the_redcap_folder_is_reported(self):
        self.assertEqual(self.check(self.candidate(stray="notes.txt")), ["redcap/0#file/notes.txt"])

    def test_an_extra_key_in_every_redcap_row_is_reported(self):
        self.assertEqual(self.check(self.candidate(row_extra={"comment": "x"})),
                         sorted(f"{ROW}/{field}[comment]" for field in FIELDS))

    def test_another_number_of_redcap_rows_is_reported(self):
        root = write_run(self.root / "candidate", prefix="c", content_time="121212", memory_peak=3)
        output = root / "redcap" / "1.2.9" / "output.json"
        rows = json.loads(output.read_text())
        output.write_text(json.dumps(rows + rows[:1]))
        self.assertEqual(self.check(compare_outputs.snapshot(root)), ["redcap/0#rows"])

    def test_a_changed_data_dictionary_member_is_reported(self):
        self.assertEqual(self.check(self.candidate(dictionary="Variable / Field Name\nother\n")),
                         ["redcap/0#dictionary/instrument.csv"])

    def test_values_that_differ_only_in_a_memory_address_are_compared(self):
        key = f"{ROW}/boundingbox"
        self.assertIn("ImageRegion (0x...)", self.baseline[key])
        self.assertEqual(self.check(self.candidate()), [])
        self.assertEqual(self.check(self.candidate(size="[3, 2, 1]")), [key])

    def test_a_candidate_must_have_the_unstable_entries_of_both_reference_runs(self):
        candidate = self.candidate()
        del candidate["labels/0000#ContentTime"]
        self.assertEqual(self.check(candidate), ["labels/0000#ContentTime"])

    def test_an_entry_of_only_one_reference_run_may_be_missing(self):
        first = compare_outputs.snapshot(self.root / "first") | {"runtime_resources#extra": '"1"'}
        baseline, dropped, in_both = compare_outputs.stable(
            first, compare_outputs.snapshot(self.root / "second"))
        self.assertIn("runtime_resources#extra", dropped)
        self.assertNotIn("runtime_resources#extra", in_both)
        self.assertEqual(compare_outputs.check(baseline, self.candidate(), dropped, in_both), [])

    def test_long_values_are_stored_as_a_hash(self):
        value = self.baseline[f"{ROW}/mask_base64_gzip_compressed"]
        self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")

    def test_an_unchanged_candidate_matches(self):
        self.assertEqual(self.check(self.candidate()), [])

    def test_entries_that_vary_are_additions_unless_they_are_unstable(self):
        self.assertEqual(compare_outputs.check(self.baseline, self.candidate()), self.dropped)

    def test_an_added_entry_is_reported(self):
        candidate = self.candidate()
        candidate[f"{ROW}/new_field"] = '"1"'
        self.assertEqual(self.check(candidate), [f"{ROW}/new_field"])

    def test_a_changed_pixel_and_redcap_value_are_reported(self):
        different = self.check(self.candidate(pixel=2, model_version="model, other:1"))
        self.assertEqual(different, [
            "labels/0000#pixels", "labels/0001#pixels", "labels/0002#pixels",
            f"{ROW}/model_version"])

    def test_a_missing_output_is_reported(self):
        candidate = self.candidate()
        del candidate["labels/0002#pixels"]
        self.assertEqual(self.check(candidate), ["labels/0002#pixels"])

    def test_snapshot_does_not_show_pydicom_warnings(self):
        path = write_image(self.root / "invalid" / "labels" / "1.2.8" / "a.dcm", 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = dcmread(path)
            dataset["FrameOfReferenceUID"] = DataElement(0x00200052, "UI", "1.2.abc")
            dataset.save_as(path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entries = compare_outputs.snapshot(self.root / "invalid")
        self.assertEqual([str(warning.message) for warning in caught], [])
        self.assertEqual(entries["labels/0000#FrameOfReferenceUID"], '["1.2.abc"]')

    def test_command_line_writes_files_and_fails_on_a_difference(self):
        paths = {name: self.root / f"{name}.json" for name in ("first", "second", "baseline", "candidate")}
        for name in ("first", "second"):
            self.assertEqual(run_command(["snapshot", str(self.root / name), str(paths[name])])[0], 0)
        self.assertEqual(run_command(
            ["stable", str(paths["first"]), str(paths["second"]), str(paths["baseline"])])[0], 0)
        baseline = json.loads(paths["baseline"].read_text())
        self.assertEqual(sorted(baseline), ["stable", "unstable", "unstable_in_both"])
        self.assertEqual(baseline["stable"], self.baseline)
        self.assertEqual(baseline["unstable"], self.dropped)
        self.assertEqual(baseline["unstable_in_both"], self.in_both)
        paths["candidate"].write_text(json.dumps(self.candidate(pixel=2)))
        status, _, errors = run_command(["check", str(paths["baseline"]), str(paths["candidate"])])
        self.assertEqual(status, 1)
        self.assertIn("Error: 3 entries differ", errors)
        self.assertIn("labels/0000#pixels", errors)
        candidate = compare_outputs.snapshot(self.root / "first")
        paths["candidate"].write_text(json.dumps(candidate))
        status, output, _ = run_command(["check", str(paths["baseline"]), str(paths["candidate"])])
        self.assertEqual(status, 0)
        self.assertIn("candidate matches the baseline", output)
        del candidate["labels/0001#ContentTime"]
        paths["candidate"].write_text(json.dumps(candidate))
        status, _, errors = run_command(["check", str(paths["baseline"]), str(paths["candidate"])])
        self.assertEqual(status, 1)
        self.assertIn("labels/0001#ContentTime", errors)

    def test_outputs_are_written_through_a_temporary_file(self):
        """A run killed while writing must not leave a file that looks finished."""
        snapshot = self.root / "first.json"
        snapshot.write_text(json.dumps(compare_outputs.snapshot(self.root / "first")))
        commands = {"snapshot": ["snapshot", str(self.root / "second"), str(self.root / "second.json")],
                    "stable": ["stable", str(snapshot), str(snapshot), str(self.root / "baseline.json")]}
        for name, argv in commands.items():
            with self.subTest(name), mock.patch("os.replace", side_effect=OSError("killed")):
                with self.assertRaises(OSError):
                    run_command(argv)
                self.assertEqual(sorted(path.name for path in self.root.iterdir() if path.is_file()),
                                 ["first.json"])

    def test_two_runs_that_disagree_on_the_mask_are_an_error(self):
        paths = {name: self.root / f"{name}.json" for name in ("first", "other", "baseline")}
        paths["first"].write_text(json.dumps(compare_outputs.snapshot(self.root / "first")))
        paths["other"].write_text(json.dumps(self.candidate(pixel=2)))
        status, _, errors = run_command(
            ["stable", str(paths["first"]), str(paths["other"]), str(paths["baseline"])])
        self.assertEqual(status, 1)
        self.assertIn("Error: the two runs disagree on the mask pixels", errors)

    def test_two_runs_that_disagree_on_other_pixels_give_a_warning_and_a_summary(self):
        first = compare_outputs.snapshot(self.root / "first")
        first |= {f"fused/000{index}#pixels": "sha256:" + "0" * 64 for index in range(2)}
        second = first | {f"fused/000{index}#pixels": "sha256:" + "1" * 64 for index in range(2)}
        paths = {name: self.root / f"{name}.json" for name in ("first", "second", "baseline")}
        paths["first"].write_text(json.dumps(first))
        paths["second"].write_text(json.dumps(second))
        status, _, errors = run_command(
            ["stable", str(paths["first"]), str(paths["second"]), str(paths["baseline"])])
        self.assertEqual(status, 0)
        self.assertEqual(errors, "Warning: the two runs disagree on 2 entries of 'fused pixels'; "
                                 "they are not compared\n")
        self.assertIn("fused/0000#pixels", json.loads(paths["baseline"].read_text())["unstable"])
        self.assertEqual(run_command(["uncompared", str(paths["baseline"])]),
                         (0, "fused pixels (2 entries)\n", ""))
        paths["second"].write_text(json.dumps(first))
        run_command(["stable", str(paths["first"]), str(paths["second"]), str(paths["baseline"])])
        self.assertEqual(run_command(["uncompared", str(paths["baseline"])]), (0, "", ""))


if __name__ == "__main__":
    unittest.main()
