"""The run comparison must ignore what differs between two runs of one image
and report a changed, missing or added pixel, tag or REDCap value in a candidate."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from pydicom import dcmread

from nnunet_inference.integration import compare_outputs
from nnunet_inference.tests.synthetic import write_image


def write_run(root, *, prefix, content_time, memory_peak, pixel=1, model_version="model, image:1"):
    """A small container output: three mask slices, REDCap rows and resources."""
    root = Path(root)
    for index in range(3):
        pixels = np.zeros((2, 3), dtype=np.uint16)
        pixels[0, index] = pixel
        path = write_image(root / "labels" / "1.2.8" / f"{prefix}{2 - index}.dcm", index,
                           series_uid="1.2.8", pixels=pixels)
        dataset = dcmread(path)
        dataset.InstanceNumber = index + 1
        dataset.ContentTime = content_time
        dataset.save_as(path)
    report = root / "redcap" / "1.2.9"
    report.mkdir(parents=True)
    rows = [dict(record_id="SYNTHETIC", redcap_event_name="test_arm_1",
                 redcap_repeat_instrument="pr2mask", redcap_repeat_instance="101",
                 field_name=name, value=value)
            for name, value in (("number_of_pixel", "3"), ("model_version", model_version),
                                ("mask_base64_gzip_compressed", "m" * 500))]
    (report / "output.json").write_text(json.dumps(rows))
    with zipfile.ZipFile(report / "output_data_dictionary.zip", "w") as archive:
        archive.writestr("instrument.csv", "Variable / Field Name\nnumber_of_pixel\n")
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
            self.root / "second", prefix="b", content_time="111111", memory_peak=2))
        self.baseline, self.dropped = compare_outputs.stable(first, second)

    def candidate(self, **changes):
        settings = dict(prefix="c", content_time="121212", memory_peak=3) | changes
        return compare_outputs.snapshot(write_run(self.root / "candidate", **settings))

    def test_stable_keeps_pixels_and_values_and_drops_what_varies(self):
        self.assertEqual(self.baseline["labels#files"], "3")
        self.assertEqual(len([k for k in self.baseline if k.endswith("#pixels")]), 3)
        self.assertIn("labels/0000#SOPInstanceUID", self.baseline)
        self.assertIn("redcap/0#SYNTHETIC/test_arm_1/pr2mask/101/number_of_pixel", self.baseline)
        self.assertEqual(self.baseline["runtime_resources#tta"], "true")
        self.assertEqual(self.dropped, [
            "labels/0000#ContentTime", "labels/0000#path",
            "labels/0001#ContentTime", "labels/0001#path",
            "labels/0002#ContentTime", "labels/0002#path",
            "runtime_resources#cgroup_memory_peak_bytes"])

    def test_long_values_are_stored_as_a_hash(self):
        value = self.baseline["redcap/0#SYNTHETIC/test_arm_1/pr2mask/101/mask_base64_gzip_compressed"]
        self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")

    def test_an_unchanged_candidate_matches(self):
        self.assertEqual(compare_outputs.check(self.baseline, self.candidate(), self.dropped), [])

    def test_entries_that_vary_are_additions_unless_they_are_unstable(self):
        self.assertEqual(compare_outputs.check(self.baseline, self.candidate()), self.dropped)

    def test_an_added_entry_is_reported(self):
        candidate = self.candidate()
        candidate["redcap/0#SYNTHETIC/test_arm_1/pr2mask/101/new_field"] = '"1"'
        self.assertEqual(compare_outputs.check(self.baseline, candidate, self.dropped),
                         ["redcap/0#SYNTHETIC/test_arm_1/pr2mask/101/new_field"])

    def test_a_changed_pixel_and_redcap_value_are_reported(self):
        different = compare_outputs.check(
            self.baseline, self.candidate(pixel=2, model_version="model, other:1"), self.dropped)
        self.assertEqual(different, [
            "labels/0000#pixels", "labels/0001#pixels", "labels/0002#pixels",
            "redcap/0#SYNTHETIC/test_arm_1/pr2mask/101/model_version"])

    def test_a_missing_output_is_reported(self):
        candidate = self.candidate()
        del candidate["labels/0002#pixels"]
        self.assertEqual(compare_outputs.check(self.baseline, candidate, self.dropped),
                         ["labels/0002#pixels"])

    def test_command_line_writes_files_and_fails_on_a_difference(self):
        paths = {name: self.root / f"{name}.json" for name in ("first", "second", "baseline", "candidate")}
        for name in ("first", "second"):
            self.assertEqual(run_command(["snapshot", str(self.root / name), str(paths[name])])[0], 0)
        self.assertEqual(run_command(
            ["stable", str(paths["first"]), str(paths["second"]), str(paths["baseline"])])[0], 0)
        baseline = json.loads(paths["baseline"].read_text())
        self.assertEqual(sorted(baseline), ["stable", "unstable"])
        self.assertEqual(baseline["stable"], self.baseline)
        self.assertEqual(baseline["unstable"], self.dropped)
        paths["candidate"].write_text(json.dumps(self.candidate(pixel=2)))
        status, _, errors = run_command(["check", str(paths["baseline"]), str(paths["candidate"])])
        self.assertEqual(status, 1)
        self.assertIn("Error: 3 entries differ", errors)
        self.assertIn("labels/0000#pixels", errors)
        paths["candidate"].write_text(json.dumps(compare_outputs.snapshot(self.root / "first")))
        status, output, _ = run_command(["check", str(paths["baseline"]), str(paths["candidate"])])
        self.assertEqual(status, 0)
        self.assertIn("candidate matches the baseline", output)

    def test_two_runs_that_disagree_on_the_mask_are_an_error(self):
        paths = {name: self.root / f"{name}.json" for name in ("first", "other", "baseline")}
        paths["first"].write_text(json.dumps(compare_outputs.snapshot(self.root / "first")))
        paths["other"].write_text(json.dumps(self.candidate(pixel=2)))
        status, _, errors = run_command(
            ["stable", str(paths["first"]), str(paths["other"]), str(paths["baseline"])])
        self.assertEqual(status, 1)
        self.assertIn("Error: the two runs disagree on the mask pixels", errors)

    def test_two_runs_that_disagree_on_other_pixels_give_a_warning(self):
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


if __name__ == "__main__":
    unittest.main()
