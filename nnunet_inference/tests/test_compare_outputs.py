"""The run comparison must ignore what differs between two runs of one image
and report a changed pixel, tag or REDCap value in a candidate."""
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
        self.assertEqual(compare_outputs.check(self.baseline, self.candidate()), [])

    def test_a_changed_pixel_and_redcap_value_are_reported(self):
        different = compare_outputs.check(
            self.baseline, self.candidate(pixel=2, model_version="model, other:1"))
        self.assertEqual(different, [
            "labels/0000#pixels", "labels/0001#pixels", "labels/0002#pixels",
            "redcap/0#SYNTHETIC/test_arm_1/pr2mask/101/model_version"])

    def test_a_missing_output_is_reported(self):
        candidate = self.candidate()
        del candidate["labels/0002#pixels"]
        self.assertEqual(compare_outputs.check(self.baseline, candidate), ["labels/0002#pixels"])

    def test_command_line_writes_files_and_fails_on_a_difference(self):
        paths = {name: self.root / f"{name}.json" for name in ("first", "second", "baseline", "candidate")}
        for name in ("first", "second"):
            self.assertEqual(compare_outputs.main(["snapshot", str(self.root / name), str(paths[name])]), 0)
        self.assertEqual(compare_outputs.main(
            ["stable", str(paths["first"]), str(paths["second"]), str(paths["baseline"])]), 0)
        paths["candidate"].write_text(json.dumps(self.candidate(pixel=2)))
        self.assertEqual(compare_outputs.main(["check", str(paths["baseline"]), str(paths["candidate"])]), 1)
        paths["candidate"].write_text(json.dumps(compare_outputs.snapshot(self.root / "first")))
        self.assertEqual(compare_outputs.main(["check", str(paths["baseline"]), str(paths["candidate"])]), 0)

    def test_two_runs_that_disagree_on_the_mask_are_an_error(self):
        paths = {name: self.root / f"{name}.json" for name in ("first", "other", "baseline")}
        paths["first"].write_text(json.dumps(compare_outputs.snapshot(self.root / "first")))
        paths["other"].write_text(json.dumps(self.candidate(pixel=2)))
        self.assertEqual(compare_outputs.main(
            ["stable", str(paths["first"]), str(paths["other"]), str(paths["baseline"])]), 1)


if __name__ == "__main__":
    unittest.main()
