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


class ReleaseRegressionScriptTests(unittest.TestCase):
    """smoke-test.sh is replaced by a script that writes runtime_resources.json
    with the image, TTA setting and a value that differs in every run."""

    def run_regression(self, candidate_value):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "input").mkdir()
        for name in ("release-regression.sh", "compare_outputs.py"):
            shutil.copy2(INTEGRATION / name, root / name)
        executable(root / "smoke-test.sh", r'''#!/bin/bash
set -euo pipefail
mkdir -p "$2"
printf '%s %s %s\n' "${IMAGE_TAG}" "${TTA}" "${CONTAINER_VERSION_OVERRIDE:-}" >> "${RUN_LOG}"
value=same
if [ "${IMAGE_TAG}" = candidate:1 ]; then value="${CANDIDATE_VALUE}"; fi
printf '{"tta": %s, "value": "%s", "run": "%s"}\n' "${TTA}" "${value}" "${RANDOM}${RANDOM}" \
    > "$2/runtime_resources.json"
''')
        log = root / "runs.log"
        env = dict(os.environ, NNUNET_PYTHON=sys.executable, RUN_LOG=str(log),
                   CANDIDATE_VALUE=candidate_value,
                   PYTHONPATH=str(INTEGRATION.parents[1]))
        result = subprocess.run(
            ["bash", str(root / "release-regression.sh"), "reference:1", "candidate:1",
             str(root / "input"), str(root / "work")], env=env, capture_output=True, text=True)
        return result, log.read_text().splitlines(), root / "work"

    def test_matching_candidate_passes_after_two_reference_runs_per_setting(self):
        result, runs, work = self.run_regression("same")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(runs, ["reference:1 true ", "reference:1 true ", "candidate:1 true reference:1",
                                "reference:1 false ", "reference:1 false ", "candidate:1 false reference:1"])
        baseline = json.loads((work / "baseline-tta-false.json").read_text())
        self.assertEqual(baseline["stable"]["runtime_resources#value"], '"same"')
        self.assertIn("runtime_resources#run", baseline["unstable"])
        self.assertIn("Release regression: OK", result.stdout)

    def test_differing_candidate_fails(self):
        result, _, _ = self.run_regression("changed")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("runtime_resources#value", result.stderr)

    def test_usage_error_without_arguments(self):
        result = subprocess.run(["bash", str(INTEGRATION / "release-regression.sh")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage", result.stderr)


if __name__ == "__main__":
    unittest.main()
