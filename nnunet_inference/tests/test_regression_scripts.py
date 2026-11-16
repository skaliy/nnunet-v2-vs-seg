"""Exercise smoke-test.sh and release-regression.sh with stand-in programs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

INTEGRATION = Path(__file__).resolve().parents[1] / "integration"


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


def environment_without(*names):
    return {key: value for key, value in os.environ.items() if key not in names}


class SmokeTestScriptTests(unittest.TestCase):
    def docker_arguments(self, **environment):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "input").mkdir()
        log = root / "docker.log"
        executable(root / "docker", '#!/bin/bash\nprintf "%s\\n" "$@" > "$DOCKER_LOG"\n')
        executable(root / "python", "#!/bin/bash\nexit 0\n")
        env = dict(environment_without("TTA", "IMAGE_TAG", "CONTAINER_VERSION_OVERRIDE"),
                   PATH=f"{root}:{os.environ['PATH']}", DOCKER_LOG=str(log),
                   NNUNET_PYTHON=str(root / "python"), CPUSET="0-15", **environment)
        result = subprocess.run(["bash", str(INTEGRATION / "smoke-test.sh"), str(root / "input"),
                                 str(root / "output")], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return log.read_text().splitlines()

    def test_default_run_keeps_the_image_identity(self):
        arguments = self.docker_arguments(IMAGE_TAG="image:1")
        self.assertEqual(arguments[-1], "image:1")
        self.assertIn('ROR_CONT_OPTIONS={"tta":true}', arguments)
        self.assertFalse([a for a in arguments if a.startswith("CONTAINER_VERSION=")])

    def test_override_passes_the_reference_identity(self):
        arguments = self.docker_arguments(IMAGE_TAG="candidate:1", TTA="false",
                                          CONTAINER_VERSION_OVERRIDE="reference:1")
        self.assertIn('ROR_CONT_OPTIONS={"tta":false}', arguments)
        self.assertEqual(arguments[arguments.index("CONTAINER_VERSION=reference:1") - 1], "-e")
        self.assertEqual(arguments[-1], "candidate:1")


REFERENCE_ID = "0123456789ab" + "0" * 52
CANDIDATE_ID = "fedcba987654" + "0" * 52
REBUILT_ID = "abcdef012345" + "0" * 52
REFERENCE_VERSION = "nnunet-vsseg:20261112T095746Z"
REFERENCE_RUN = f"sha256:{REFERENCE_ID}"
CANDIDATE_RUN = f"sha256:{CANDIDATE_ID}"
ALL_RUNS = [f"{REFERENCE_RUN} true ", f"{REFERENCE_RUN} true ",
            f"{CANDIDATE_RUN} true {REFERENCE_VERSION}",
            f"{REFERENCE_RUN} false ", f"{REFERENCE_RUN} false ",
            f"{CANDIDATE_RUN} false {REFERENCE_VERSION}"]
OK = "Release regression: OK (candidate:1 matches reference:1)"


class ReleaseRegressionScriptTests(unittest.TestCase):
    """smoke-test.sh is replaced by a script that writes runtime_resources.json
    with the image, TTA setting and a value that differs in every run, and
    docker by a script that gives each image the ID set in the environment and
    the reference image a CONTAINER_VERSION. With STANDIN_PIXELS=<output> the
    stand-in also writes one DICOM file to that output, with another pixel
    value in every run; with STANDIN_FAILS=1 a candidate run fails halfway."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "input").mkdir()
        (self.root / "input" / "slice.dcm").write_bytes(b"synthetic series")
        (self.root / "bin").mkdir()
        for name in ("release-regression.sh", "compare_outputs.py"):
            shutil.copy2(INTEGRATION / name, self.root / name)
        executable(self.root / "smoke-test.sh", r'''#!/bin/bash
set -euo pipefail
mkdir -p "$2"
printf '%s %s %s\n' "${IMAGE_TAG}" "${TTA}" "${CONTAINER_VERSION_OVERRIDE:-}" >> "${RUN_LOG}"
value=same
if [ "${IMAGE_TAG}" = "sha256:${STANDIN_CANDIDATE_ID}" ]; then
    if [ -n "${STANDIN_FAILS:-}" ]; then echo partial > "$2/partial"; exit 1; fi
    value="${STANDIN_CANDIDATE_VALUE}"
fi
if [ -n "${STANDIN_PIXELS:-}" ]; then
    "${NNUNET_PYTHON}" -c 'import sys
from nnunet_inference.tests.synthetic import write_image
write_image(sys.argv[1], 0, pixels=[[int(sys.argv[2]), 0, 0], [0, 0, 0]])' \
        "$2/${STANDIN_PIXELS}/1.2.8/0.dcm" "$(wc -l < "${RUN_LOG}")"
fi
printf '{"tta": %s, "value": "%s", "run": "%s"}\n' "${TTA}" "${value}" "${RANDOM}${RANDOM}" \
    > "$2/runtime_resources.json"
''')
        executable(self.root / "bin" / "docker", r'''#!/bin/bash
[ "$1 $2 $4" = 'image inspect --format' ] || exit 2
case "$3" in
    reference:1|"sha256:${STANDIN_REFERENCE_ID}")
        id="${STANDIN_REFERENCE_ID}" version="${STANDIN_REFERENCE_VERSION}" ;;
    candidate:1|"sha256:${STANDIN_CANDIDATE_ID}")
        id="${STANDIN_CANDIDATE_ID}" version=nnunet-vsseg-check:1 ;;
    *) printf 'Error: No such image: %s\n' "$3" >&2; exit 1 ;;
esac
case "$5" in
    '{{.Id}}') printf 'sha256:%s\n' "${id}" ;;
    '{{range .Config.Env}}{{println .}}{{end}}')
        printf 'PATH=/usr/local/bin:/usr/bin\n'
        if [ -n "${version}" ]; then printf 'CONTAINER_VERSION=%s\n' "${version}"; fi
        printf 'VERSION=1\n\n' ;;
    *) exit 2 ;;
esac
''')
        self.log = self.root / "runs.log"
        self.work = self.root / "work"

    def run_regression(self, candidate_value="same", candidate="candidate:1",
                       candidate_id=CANDIDATE_ID, arguments=None, cwd=None, **environment):
        # The stand-ins' variables have their own prefix: the script's own
        # variables would overwrite inherited ones of the same name.
        env = dict(environment_without("STANDIN_FAILS", "STANDIN_PIXELS"),
                   PATH=f"{self.root / 'bin'}:{os.environ['PATH']}",
                   NNUNET_PYTHON=sys.executable, RUN_LOG=str(self.log),
                   STANDIN_CANDIDATE_VALUE=candidate_value, STANDIN_REFERENCE_ID=REFERENCE_ID,
                   STANDIN_CANDIDATE_ID=candidate_id, STANDIN_REFERENCE_VERSION=REFERENCE_VERSION,
                   PYTHONPATH=str(INTEGRATION.parents[1]),
                   # Without this, numpy's thread pools spend seconds of CPU per Python start.
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1") | environment
        if arguments is None:
            arguments = [str(self.root / "input"), str(self.work)]
        return subprocess.run(
            ["bash", str(self.root / "release-regression.sh"), "reference:1", candidate, *arguments],
            env=env, cwd=cwd, capture_output=True, text=True)

    def runs(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_matching_candidate_passes_after_two_reference_runs_per_setting(self):
        result = self.run_regression()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.runs(), ALL_RUNS)
        self.assertEqual(sorted(path.name for path in self.work.glob("*.json")), [
            "baseline-0123456789ab-tta-false.json", "baseline-0123456789ab-tta-true.json",
            "candidate-fedcba987654-tta-false.json", "candidate-fedcba987654-tta-true.json",
            "reference-0123456789ab-tta-false-1.json", "reference-0123456789ab-tta-false-2.json",
            "reference-0123456789ab-tta-true-1.json", "reference-0123456789ab-tta-true-2.json"])
        baseline = json.loads((self.work / "baseline-0123456789ab-tta-false.json").read_text())
        self.assertEqual(baseline["stable"]["runtime_resources#value"], '"same"')
        self.assertIn("runtime_resources#run", baseline["unstable"])
        self.assertEqual(result.stdout.splitlines()[-1], OK)
        self.assertRegex((self.work / "input-series.txt").read_text(),
                         rf"^[0-9a-f]{{64}} {self.root / 'input'}\n$")

    def test_differing_candidate_fails(self):
        result = self.run_regression("changed")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("runtime_resources#value", result.stderr)
        self.assertNotIn("Release regression: OK", result.stdout)

    def test_rebuilt_candidate_runs_again_and_reuses_the_reference_runs(self):
        self.assertEqual(self.run_regression().returncode, 0)
        result = self.run_regression("changed", candidate_id=REBUILT_ID)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("runtime_resources#value", result.stderr)
        self.assertEqual(self.runs(), ALL_RUNS + [f"sha256:{REBUILT_ID} true {REFERENCE_VERSION}",
                                                  f"sha256:{REBUILT_ID} false {REFERENCE_VERSION}"])
        self.assertTrue((self.work / "candidate-abcdef012345-tta-false.json").is_file())

    def test_unfinished_run_is_removed_and_run_again(self):
        unfinished = self.work / "reference-0123456789ab-tta-true-1"
        unfinished.mkdir(parents=True)
        (unfinished / "partial").write_text("interrupted")
        result = self.run_regression()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"unfinished run {unfinished}", result.stdout)
        self.assertFalse((unfinished / "partial").exists())
        self.assertTrue((unfinished / "runtime_resources.json").is_file())
        self.assertEqual(self.runs(), ALL_RUNS)

    def test_failed_candidate_run_fails_and_runs_again_next_time(self):
        result = self.run_regression(STANDIN_FAILS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Release regression: OK", result.stdout)
        self.assertEqual(self.runs(), ALL_RUNS[:3])
        result = self.run_regression()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Removing the unfinished run", result.stdout)
        self.assertEqual(self.runs(), ALL_RUNS[:3] + ALL_RUNS[2:])

    def test_reference_runs_that_disagree_on_the_mask_stop_before_the_candidate(self):
        result = self.run_regression(STANDIN_PIXELS="labels")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("Error: the two runs disagree on the mask pixels", result.stderr)
        self.assertNotIn("Release regression: OK", result.stdout)
        self.assertEqual(self.runs(), ALL_RUNS[:2])

    def test_pixels_the_reference_runs_disagree_on_are_named_next_to_the_ok_line(self):
        result = self.run_regression(STANDIN_PIXELS="fused")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-3:], [OK] + [
            f"Not compared with TTA {tta}, because the two release runs disagree: "
            "fused pixels (1 entry)" for tta in ("true", "false")])

    def test_relative_paths_are_resolved_from_the_callers_directory(self):
        caller = self.root / "caller"
        caller.mkdir()
        result = self.run_regression(arguments=["../input", "work"], cwd=caller)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((caller / "work" / "candidate-fedcba987654-tta-false.json").is_file())
        self.assertFalse(self.work.exists())
        self.assertTrue((caller / "work" / "input-series.txt").read_text().endswith(
            f" {self.root / 'input'}\n"))

    def test_a_work_directory_in_the_repository_must_be_ignored_by_git(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        result = self.run_regression()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(f"{self.work} is inside the repository {self.root}", result.stderr)
        self.assertEqual(self.runs(), [])
        self.assertFalse(self.work.exists())
        (self.root / ".gitignore").write_text("/work/\n")
        result = self.run_regression()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.runs(), ALL_RUNS)

    def test_a_work_directory_belongs_to_one_input_series(self):
        self.assertEqual(self.run_regression().returncode, 0)
        copy = shutil.copytree(self.root / "input", self.root / "copy")
        self.assertEqual(self.run_regression(arguments=[str(copy), str(self.work)]).returncode, 0)
        other = self.root / "other"
        other.mkdir()
        (other / "slice.dcm").write_bytes(b"another series")
        result = self.run_regression(arguments=[str(other), str(self.work)])
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(str(self.root / "input"), result.stderr)
        self.assertIn(str(other), result.stderr)
        self.assertEqual(self.runs(), ALL_RUNS)

    def test_a_work_directory_without_an_input_record_gets_one(self):
        """A work directory of the previous version of the script has none."""
        self.assertEqual(self.run_regression().returncode, 0)
        record = self.work / "input-series.txt"
        expected = record.read_text()
        record.unlink()
        result = self.run_regression()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record.read_text(), expected)
        self.assertEqual(self.runs(), ALL_RUNS)

    def test_reference_without_a_container_version_is_a_usage_error(self):
        result = self.run_regression(STANDIN_REFERENCE_VERSION="")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("image sha256:0123456789ab", result.stderr)
        self.assertIn("has no CONTAINER_VERSION", result.stderr)
        self.assertEqual(self.runs(), [])
        self.assertFalse(self.work.exists())

    def test_missing_image_is_a_usage_error(self):
        result = self.run_regression(candidate="missing:1")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("image missing:1 is not present locally", result.stderr)
        self.assertEqual(self.runs(), [])
        self.assertFalse(self.work.exists())

    def test_usage_error_without_arguments(self):
        result = subprocess.run(["bash", str(INTEGRATION / "release-regression.sh")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage", result.stderr)


if __name__ == "__main__":
    unittest.main()
