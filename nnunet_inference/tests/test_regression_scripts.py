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


class SmokeTestScriptTests(unittest.TestCase):
    def docker_arguments(self, **environment):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "input").mkdir()
        log = root / "docker.log"
        executable(root / "docker", '#!/bin/bash\nprintf "%s\\n" "$@" > "$DOCKER_LOG"\n')
        executable(root / "python", "#!/bin/bash\nexit 0\n")
        env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", DOCKER_LOG=str(log),
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
ALL_RUNS = ["reference:1 true ", "reference:1 true ", "candidate:1 true reference:1",
            "reference:1 false ", "reference:1 false ", "candidate:1 false reference:1"]


class ReleaseRegressionScriptTests(unittest.TestCase):
    """smoke-test.sh is replaced by a script that writes runtime_resources.json
    with the image, TTA setting and a value that differs in every run, and
    docker by a script that gives each image the ID set in the environment."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "input").mkdir()
        (self.root / "bin").mkdir()
        for name in ("release-regression.sh", "compare_outputs.py"):
            shutil.copy2(INTEGRATION / name, self.root / name)
        executable(self.root / "smoke-test.sh", r'''#!/bin/bash
set -euo pipefail
mkdir -p "$2"
printf '%s %s %s\n' "${IMAGE_TAG}" "${TTA}" "${CONTAINER_VERSION_OVERRIDE:-}" >> "${RUN_LOG}"
value=same
if [ "${IMAGE_TAG}" = candidate:1 ]; then value="${CANDIDATE_VALUE}"; fi
printf '{"tta": %s, "value": "%s", "run": "%s"}\n' "${TTA}" "${value}" "${RANDOM}${RANDOM}" \
    > "$2/runtime_resources.json"
''')
        executable(self.root / "bin" / "docker", r'''#!/bin/bash
[ "$1 $2 $4 $5" = 'image inspect --format {{.Id}}' ] || exit 2
case "$3" in
    reference:1) printf 'sha256:%s\n' "${REFERENCE_ID}" ;;
    candidate:1) printf 'sha256:%s\n' "${CANDIDATE_ID}" ;;
    *) printf 'Error: No such image: %s\n' "$3" >&2; exit 1 ;;
esac
''')
        self.log = self.root / "runs.log"
        self.work = self.root / "work"

    def run_regression(self, candidate_value="same", candidate="candidate:1",
                       candidate_id=CANDIDATE_ID):
        env = dict(os.environ, PATH=f"{self.root / 'bin'}:{os.environ['PATH']}",
                   NNUNET_PYTHON=sys.executable, RUN_LOG=str(self.log),
                   CANDIDATE_VALUE=candidate_value, REFERENCE_ID=REFERENCE_ID,
                   CANDIDATE_ID=candidate_id, PYTHONPATH=str(INTEGRATION.parents[1]))
        return subprocess.run(
            ["bash", str(self.root / "release-regression.sh"), "reference:1", candidate,
             str(self.root / "input"), str(self.work)], env=env, capture_output=True, text=True)

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
        self.assertIn("Release regression: OK", result.stdout)

    def test_differing_candidate_fails(self):
        result = self.run_regression("changed")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("runtime_resources#value", result.stderr)

    def test_rebuilt_candidate_runs_again_and_reuses_the_reference_runs(self):
        self.assertEqual(self.run_regression().returncode, 0)
        result = self.run_regression("changed", candidate_id=REBUILT_ID)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("runtime_resources#value", result.stderr)
        self.assertEqual(self.runs(), ALL_RUNS + ["candidate:1 true reference:1",
                                                  "candidate:1 false reference:1"])
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
