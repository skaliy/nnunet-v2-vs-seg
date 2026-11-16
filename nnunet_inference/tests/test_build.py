"""Exercise build verification and staging cleanup without loading real weights."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class BuildTests(unittest.TestCase):
    def run_build(self, *, mode="--build", fail_verification=False, **environment):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        package = root / "nnunet_inference"
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source, package, ignore=shutil.ignore_patterns(
            "integration", "tests", "__pycache__"))
        integration = package / "integration"
        integration.mkdir()
        shutil.copy2(source / "integration/build.sh", integration / "build.sh")
        original = root / "trained-model"
        original.mkdir()
        (original / "checkpoint.pth").write_bytes(b"preserve trained weights")
        binaries = root / "bin"
        binaries.mkdir()
        scripts = {
            "python": r'''#!/bin/bash
set -euo pipefail
if [ "$1" = -c ]; then exit 0; fi
if [ "$2" = nnunet_inference.deployment ]; then
    if [[ " $* " == *" --source "* ]]; then
        while [ "$#" -gt 0 ]; do
            if [ "$1" = --output ]; then mkdir -p "$2"; exit 0; fi
            shift
        done
        exit 2
    fi
    printf '%s\n' verification >> "$BUILD_TEST_LOG"
    if [ "$BUILD_TEST_FAIL" = 1 ]; then exit 7; fi
elif [ "$2" = nnunet_inference.pacs ]; then
    printf '%s\n' help >> "$BUILD_TEST_LOG"
else
    exit 2
fi
''',
            "docker": r'''#!/bin/bash
set -euo pipefail
case "$1" in
    pull) printf '%s\n' "$2" >> "$BUILD_TEST_PULLS"; exit 0 ;;
    build) exit 0 ;;
    image) printf '%s\n' 'example/fiona@sha256:0000000000000000000000000000000000000000000000000000000000000000' ;;
    run) exec bash --noprofile --norc -c "${@: -1}" ;;
    *) exit 2 ;;
esac
''',
        }
        for name, script in scripts.items():
            path = binaries / name
            path.write_text(script)
            path.chmod(0o755)
        log = root / "commands.log"
        env = dict(os.environ, PATH=str(binaries) + ":" + os.environ["PATH"],
                   NNUNET_PYTHON=str(binaries / "python"), NNUNET_MODEL_SRC=str(original),
                   VERSION="cleanup-test", IMAGE_NAME="nnunet-build-test",
                   BUILD_TEST_LOG=str(log), BUILD_TEST_FAIL=str(int(fail_verification)),
                   BUILD_TEST_PULLS=str(root / "pulls.log"))
        env.pop("FIONA_BASE", None)
        env.update(environment)
        result = subprocess.run(["bash", str(integration / "build.sh"), mode],
                                env=env, capture_output=True, text=True)
        commands = log.read_text().splitlines() if log.exists() else []
        self.assertEqual((original / "checkpoint.pth").read_bytes(), b"preserve trained weights")
        self.assertFalse(list(integration.glob(".bundle-stage-*")))
        pulls = root / "pulls.log"
        self.pulled = pulls.read_text().splitlines() if pulls.exists() else []
        return result, integration, commands

    def test_failed_model_verification_stops_before_help_can_hide_it(self):
        result, _, commands = self.run_build(fail_verification=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(commands, ["verification"])

    def test_successful_build_removes_reproducible_copies(self):
        result, integration, commands = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, ["verification", "help"])
        self.assertFalse((integration / "model").exists())
        self.assertFalse((integration / "_pkg").exists())

    def test_base_image_defaults_to_latest_and_can_be_pinned(self):
        result, _, _ = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pulled, ["haukebartsch/fiona-component-python:latest"])
        pinned = "haukebartsch/fiona-component-python@sha256:" + "1" * 64
        result, _, _ = self.run_build(FIONA_BASE=pinned)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pulled, [pinned])

    def test_stage_only_retains_copies_without_running_docker(self):
        result, integration, commands = self.run_build(mode="--stage-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, [])
        self.assertTrue((integration / "model").is_dir())
        self.assertTrue((integration / "_pkg/nnunet_inference").is_dir())
