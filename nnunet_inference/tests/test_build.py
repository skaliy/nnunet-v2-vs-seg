"""Exercise build verification and staging cleanup without loading real weights."""
import gzip
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

VERSION = "cleanup-test"
DATED_TAG = f"nnunet-build-test:{VERSION}"
LATEST_TAG = "nnunet-build-test:latest"
PACKAGE_FILES = ["README.md", f"nnunet-build-test-{VERSION}.tar.gz", "ror_select_entry.txt"]

# Stand-ins. python stages an empty model, logs the in-image checks and writes
# release.py's two files; docker keeps the tags it was given in BUILD_TEST_IMAGES.
STANDIN_PYTHON = r'''#!/bin/bash
set -euo pipefail
if [ "$1" = -c ]; then exit 0; fi
if [ "$1" = release.py ]; then
    printf '%s\n' release >> "$BUILD_TEST_LOG"
    if [ "${BUILD_TEST_FAIL_RELEASE:-0}" = 1 ]; then exit 3; fi
    printf 'handoff\n' > "$2/README.md"
    printf 'entry for %s\n' "$3" > "$2/ror_select_entry.txt"
    exit 0
fi
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
'''
STANDIN_DOCKER = r'''#!/bin/bash
set -euo pipefail
printf '%s\n' "${*//$'\n'/ }" >> "$BUILD_TEST_DOCKER_LOG"   # one line per command
case "$1" in
    pull) printf '%s\n' "$2" >> "$BUILD_TEST_PULLS" ;;
    build)
        while [ "$#" -gt 0 ]; do
            if [ "$1" = -t ]; then printf '%s\n' "$2" >> "$BUILD_TEST_IMAGES"; fi
            shift
        done ;;
    tag) printf '%s\n' "$3" >> "$BUILD_TEST_IMAGES" ;;
    image)
        if [ "$2" = inspect ] && [ "${4:-}" = --format ]; then
            printf '%s\n' 'example/fiona@sha256:0000000000000000000000000000000000000000000000000000000000000000'
        elif [ "$2" = inspect ] && grep -qxF -- "$3" "$BUILD_TEST_IMAGES"; then
            printf '[{"Id": "sha256:1111"}]\n'
        else
            printf 'Error: No such image: %s\n' "$3" >&2; exit 1
        fi ;;
    run) exec bash --noprofile --norc -c "${@: -1}" ;;
    save)
        printf 'archive of %s\n' "${*:2}"
        if [ "${BUILD_TEST_FAIL_SAVE:-0}" = 1 ]; then exit 1; fi ;;
    *) exit 2 ;;
esac
'''


class BuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = root = Path(temporary.name)
        package = root / "nnunet_inference"
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source, package, ignore=shutil.ignore_patterns(
            "integration", "tests", "__pycache__"))
        self.integration = package / "integration"
        self.integration.mkdir()
        shutil.copy2(source / "integration/build.sh", self.integration / "build.sh")
        self.original = root / "trained-model"
        self.original.mkdir()
        (self.original / "checkpoint.pth").write_bytes(b"preserve trained weights")
        binaries = root / "bin"
        binaries.mkdir()
        for name, script in (("python", STANDIN_PYTHON), ("docker", STANDIN_DOCKER)):
            path = binaries / name
            path.write_text(script)
            path.chmod(0o755)
        self.images = root / "images.txt"
        self.images.touch()
        self.packages = self.integration / "release_packages"

    def run_build(self, *, mode="--build", fail_verification=False, **environment):
        logs = {name: self.root / f"{name}.log" for name in ("commands", "pulls", "docker")}
        for path in logs.values():
            path.unlink(missing_ok=True)
        env = dict(os.environ, PATH=str(self.root / "bin") + ":" + os.environ["PATH"],
                   NNUNET_PYTHON=str(self.root / "bin" / "python"),
                   NNUNET_MODEL_SRC=str(self.original), VERSION=VERSION,
                   IMAGE_NAME="nnunet-build-test", BUILD_TEST_LOG=str(logs["commands"]),
                   BUILD_TEST_FAIL=str(int(fail_verification)),
                   BUILD_TEST_PULLS=str(logs["pulls"]), BUILD_TEST_DOCKER_LOG=str(logs["docker"]),
                   BUILD_TEST_IMAGES=str(self.images))
        for name in ("FIONA_BASE", "BUILD_TEST_FAIL_SAVE", "BUILD_TEST_FAIL_RELEASE"):
            env.pop(name, None)
        env.update(environment)
        result = subprocess.run(["bash", str(self.integration / "build.sh"), mode],
                                env=env, capture_output=True, text=True)
        self.assertEqual((self.original / "checkpoint.pth").read_bytes(), b"preserve trained weights")
        self.assertFalse(list(self.integration.glob(".bundle-stage-*")))
        lines = {name: path.read_text().splitlines() if path.exists() else []
                 for name, path in logs.items()}
        self.pulled, self.docker = lines["pulls"], lines["docker"]
        return result, self.integration, lines["commands"]

    def tags(self):
        return self.images.read_text().splitlines()

    def remove_images(self):
        """What `docker image rm` of the build's tags does to the stand-in."""
        self.images.write_text("")

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
        result, _, _ = self.run_build(FIONA_BASE=pinned, VERSION="pinned-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pulled, [pinned])

    def test_stage_only_retains_copies_without_running_docker(self):
        result, integration, commands = self.run_build(mode="--stage-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, [])
        self.assertEqual(self.docker, [])
        self.assertTrue((integration / "model").is_dir())
        source = Path(__file__).resolve().parents[1]
        self.assertEqual(sorted(p.name for p in (integration / "_pkg/nnunet_inference").iterdir()),
                         sorted(p.name for p in source.iterdir()
                                if p.suffix in (".py", ".json", ".csv")))
        self.assertIn("pacs.py", {p.name for p in (integration / "_pkg/nnunet_inference").iterdir()})

    def test_existing_dated_tag_is_refused_before_anything_is_staged(self):
        for mode in ("--build", "--save"):
            with self.subTest(mode=mode):
                self.images.write_text(f"{DATED_TAG}\n")
                result, integration, commands = self.run_build(mode=mode)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(f"image {DATED_TAG} already exists; remove it or choose another VERSION",
                              result.stderr)
                self.assertEqual(self.docker, [f"image inspect {DATED_TAG}"])
                self.assertEqual(commands, [])
                self.assertFalse((integration / "model").exists())
                self.assertFalse(self.packages.exists())
                self.assertEqual(self.tags(), [DATED_TAG])

    def test_latest_moves_only_to_a_verified_image(self):
        result, _, _ = self.run_build(fail_verification=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(self.tags(), [DATED_TAG])
        self.assertFalse([line for line in self.docker if line.startswith("tag ")])
        self.remove_images()
        result, _, _ = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.tags(), [DATED_TAG, LATEST_TAG])
        build = next(line for line in self.docker if line.startswith("build "))
        self.assertIn(f"-t {DATED_TAG} ", build)
        self.assertNotIn(LATEST_TAG, build)
        self.assertEqual([line.split()[0] for line in self.docker[-2:]], ["run", "tag"])
        self.assertEqual(self.docker[-1], f"tag {DATED_TAG} {LATEST_TAG}")

    def test_successful_save_leaves_the_archive_readme_and_ror_entry(self):
        leftover = self.packages / f".{VERSION}.partial"
        leftover.mkdir(parents=True)
        (leftover / "from-a-killed-run").write_text("partial")
        result, _, commands = self.run_build(mode="--save")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, ["verification", "help", "release"])
        self.assertEqual([path.name for path in self.packages.iterdir()], [VERSION])
        package = self.packages / VERSION
        self.assertEqual(sorted(path.name for path in package.iterdir()), PACKAGE_FILES)
        with gzip.open(package / PACKAGE_FILES[1], "rt") as archive:
            self.assertEqual(archive.read(), f"archive of {DATED_TAG} {LATEST_TAG}\n")
        self.assertEqual((package / "ror_select_entry.txt").read_text(), f"entry for {DATED_TAG}\n")
        self.assertIn(f"Image and release archive verified: {package}", result.stdout)
        self.assertEqual([line.split()[0] for line in self.docker[-3:]], ["run", "tag", "save"])

    def test_failed_export_leaves_no_package_and_the_version_can_be_built_again(self):
        for failure, code in (("BUILD_TEST_FAIL_SAVE", 1), ("BUILD_TEST_FAIL_RELEASE", 3)):
            with self.subTest(failure=failure):
                version = failure.lower().replace("_", "-")
                result, _, _ = self.run_build(mode="--save", VERSION=version, **{failure: "1"})
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(list(self.packages.iterdir()), [])
                self.remove_images()
                result, _, _ = self.run_build(mode="--save", VERSION=version)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(sorted(path.name for path in (self.packages / version).iterdir()),
                                 ["README.md", f"nnunet-build-test-{version}.tar.gz",
                                  "ror_select_entry.txt"])
                shutil.rmtree(self.packages)

    def test_image_name_must_be_a_docker_repository_name(self):
        for name in ("registry.example.org/nnunet-vsseg", "nnunet-VSseg", "nnunet-",
                     "nnunet..vsseg"):
            with self.subTest(name=name):
                shutil.rmtree(self.packages, ignore_errors=True)
                result, _, _ = self.run_build(mode="--save", IMAGE_NAME=name)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("IMAGE_NAME must be", result.stderr)
                self.assertEqual(self.docker, [])
                self.assertFalse(self.packages.exists())
        result, _, _ = self.run_build(IMAGE_NAME="nnunet-vsseg-check.v2_a")
        self.assertEqual(result.returncode, 0, result.stderr)
