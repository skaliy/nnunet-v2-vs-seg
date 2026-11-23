from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from nnunet_inference import pacs
from nnunet_inference.redcap_output import MODEL_INSTANCES_PATH
from nnunet_inference.tests.test_redcap_output import fixture


class PacsTests(unittest.TestCase):
    def test_owned_directories_including_redcap_cannot_be_overwritten(self):
        for name in pacs.FINAL_OUTPUTS:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / name).mkdir()
                with self.assertRaisesRegex(RuntimeError, "Owned"):
                    pacs.prepare_output(tmp)

    def test_postprocessing_exports_all_five_products_with_distinct_identity(self):
        registry = json.loads(MODEL_INSTANCES_PATH.read_text())
        large = next(digest for digest, n in registry.items() if n == 100)
        deployment = dict(model_type="nnunet_large", model_name="Large", bundle_sha256=large,
                          members=[dict(member_id=f"fold_{n}") for n in range(5)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, *_ = fixture(root)
            for name in ("vote_map", "fused", "fused_vote_map", "reports"):
                (work / name).mkdir()
            output = root / "output"
            output.mkdir()
            with patch.object(pacs.subprocess, "run") as run, \
                    patch.object(pacs, "tag_pr2mask_mask") as tag_mask:
                pacs.postprocess(source, work, output, deployment, version="test",
                                 container="nnunet-vsseg:test",
                                 use_tta=True, pr2mask_dir=Path("/pr2mask"))
            tag_mask.assert_called_once_with(work / "labels" / "1.2.8", work / "mask")
            self.assertEqual(run.call_count, 3)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([command[command.index("-u") + 1] for command in commands],
                             ["test_nnunet_large_tta_report", "test_nnunet_large_tta_fused",
                              "test_nnunet_large_tta_votemap"])
            self.assertEqual(commands[0][commands[0].index("-i") + 1], "test_tta")
            self.assertTrue(all("-i" not in command for command in commands[1:]))
            titles = {command[command.index("-t") + 1] for command in commands}
            self.assertEqual(len(titles), 1)
            self.assertRegex(titles.pop(), r"^nnunet_l 5-model ensemble, [A-Z][a-z]{2}\d{6} $")
            self.assertEqual({p.name for p in output.iterdir()}, set(pacs.FINAL_OUTPUTS) | {pacs.LOG_NAME})
            rows = json.loads(next((output / "redcap").glob("*/output.json")).read_text())
            self.assertEqual({r["redcap_repeat_instance"] for r in rows}, {"101", "102", "103"})
            self.assertEqual({r["value"] for r in rows if r["field_name"] == "model_version"},
                             {"nnunet_large 5-fold, nnunet-vsseg:test"})
            self.assertEqual(len(list((output / "labels" / "1.2.8").iterdir())), 3)
            self.assertFalse((output / "vote_map").exists())
            self.assertFalse((output / "mask").exists())

    def test_postprocessing_requires_exactly_one_pr2mask_mask_series(self):
        deployment = dict(model_type="nnunet_large", model_name="Large", bundle_sha256="a" * 64,
                          members=[dict(member_id=f"fold_{n}") for n in range(5)])
        for extra in (None, "1.2.99"):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, work, *_ = fixture(root)
                if extra is None:
                    for path in (work / "labels" / "1.2.8").iterdir():
                        path.unlink()
                    (work / "labels" / "1.2.8").rmdir()
                else:
                    (work / "labels" / extra).mkdir()
                output = root / "output"
                output.mkdir()
                with patch.object(pacs.subprocess, "run"), \
                        patch.object(pacs, "tag_pr2mask_mask") as tag_mask, \
                        redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "exactly one mask series"):
                        pacs.postprocess(source, work, output, deployment, version="test",
                                         container="nnunet-vsseg:test",
                                         use_tta=True, pr2mask_dir=Path("/pr2mask"))
                tag_mask.assert_not_called()
                self.assertFalse(any(output.iterdir()))

    def test_failed_postprocessing_does_not_publish_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, *_ = fixture(root)
            output = root / "output"
            output.mkdir()
            deployment = dict(model_type="nnunet_large", model_name="Large", bundle_sha256="a" * 64,
                              members=[dict(member_id=f"fold_{n}") for n in range(5)])
            with patch.object(pacs.subprocess, "run", side_effect=RuntimeError("tool failure")), \
                    redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "tool failure"):
                    pacs.postprocess(source, work, output, deployment, version="test",
                                     container="nnunet-vsseg:test",
                                     use_tta=True, pr2mask_dir=Path("/pr2mask"))
            self.assertFalse(any(output.iterdir()))


    def stand_in_tools(self, root, script):
        """pr2mask stand-ins that run `script`."""
        tools = root / "pr2mask"
        tools.mkdir()
        for name in ("imageAndMask2Report", "imageAndMask2Fused"):
            (tools / name).write_text("#!/bin/sh\n" + script)
            (tools / name).chmod(0o755)
        return tools

    def run_postprocess(self, root, tools, source, work):
        output = root / "output"
        output.mkdir()
        deployment = dict(model_type="nnunet_large", model_name="Large",
                          bundle_sha256=next(iter(json.loads(MODEL_INSTANCES_PATH.read_text()))),
                          members=[dict(member_id=f"fold_{n}") for n in range(5)])
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                pacs.postprocess(source, work, output, deployment, version="test",
                                 container="nnunet-vsseg:test", use_tta=True, pr2mask_dir=tools)
        finally:
            self.stderr, self.output = stderr.getvalue(), output

    def test_failing_pr2mask_tool_leaves_its_last_40_log_lines_on_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, *_ = fixture(root)
            tools = self.stand_in_tools(root, 'for i in $(seq 1 60); do echo "line $i"; done\n'
                                              'echo "pr2mask: cannot read the mask series" >&2\n'
                                              'exit 3\n')
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_postprocess(root, tools, source, work)
            lines = self.stderr.splitlines()
            self.assertIn("pr2mask: cannot read the mask series", lines)
            self.assertIn("line 22", lines)
            self.assertNotIn("line 21", lines)
            self.assertFalse(any(self.output.iterdir()))

    def test_missing_pr2mask_products_leave_its_log_on_stderr(self):
        # pr2mask can print a message and exit 0 without writing its products.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, *_ = fixture(root)
            shutil.rmtree(work / "labels" / "1.2.8")
            tools = self.stand_in_tools(root, 'echo "readMaskImageSerie: no mask series"\n')
            with self.assertRaisesRegex(RuntimeError, "exactly one mask series"):
                self.run_postprocess(root, tools, source, work)
            self.assertIn("readMaskImageSerie: no mask series", self.stderr)
            self.assertFalse(any(self.output.iterdir()))

    def test_successful_pr2mask_run_writes_nothing_to_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, *_ = fixture(root)
            for name in ("vote_map", "fused", "fused_vote_map", "reports"):
                (work / name).mkdir()
            tools = self.stand_in_tools(root, 'echo "pr2mask: done"\n')
            with patch.object(pacs, "tag_pr2mask_mask"):
                self.run_postprocess(root, tools, source, work)
            self.assertEqual(self.stderr, "")
            self.assertIn("pr2mask: done", (self.output / pacs.LOG_NAME).read_text())

    def test_products_appear_only_when_all_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, output = Path(tmp) / "work", Path(tmp) / "output"
            output.mkdir()
            for name in pacs.FINAL_OUTPUTS:
                (work / name).mkdir(parents=True)
                (work / name / "file").write_text(name)
            log = work / pacs.LOG_NAME
            log.write_text("log")
            copytree, seen = pacs.shutil.copytree, []

            def recording_copytree(source, destination):
                seen.append(sorted(p.name for p in output.iterdir()))
                return copytree(source, destination)
            with patch.object(pacs.shutil, "copytree", side_effect=recording_copytree):
                pacs.publish(work, output, log)
            # While the products were copied, output held only the hidden folder.
            self.assertEqual(len(seen), len(pacs.FINAL_OUTPUTS))
            self.assertTrue(all(len(names) == 1 and names[0].startswith(".nnunet-publishing-")
                                for names in seen), seen)
            self.assertEqual({p.name for p in output.iterdir()}, set(pacs.FINAL_OUTPUTS) | {pacs.LOG_NAME})
            self.assertEqual((output / "redcap" / "file").read_text(), "redcap")
            self.assertEqual((output / "labels").stat().st_mode & 0o777, 0o777)
            self.assertEqual((output / pacs.LOG_NAME).stat().st_mode & 0o666, 0o666)

    def test_failure_while_publishing_leaves_the_output_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, output = Path(tmp) / "work", Path(tmp) / "output"
            output.mkdir()
            for name in pacs.FINAL_OUTPUTS:
                (work / name).mkdir(parents=True)
            log = work / pacs.LOG_NAME
            log.write_text("log")
            rename, calls = pacs.os.rename, []

            def failing_rename(source, destination):
                calls.append(destination)
                if len(calls) == 3:
                    raise OSError("disk failure")
                return rename(source, destination)
            with patch.object(pacs.os, "rename", side_effect=failing_rename):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    pacs.publish(work, output, log)
            self.assertFalse(any(output.iterdir()))
            self.assertTrue(all((work / name).is_dir() for name in pacs.FINAL_OUTPUTS))

    def test_model_version_names_model_folds_and_image(self):
        members = [dict(member_id=f"fold_{n}") for n in range(5)]
        self.assertEqual(pacs.model_version(dict(model_type="nnunet_large", members=members),
                                            "nnunet-vsseg:20261111T090000Z"),
                         "nnunet_large 5-fold, nnunet-vsseg:20261111T090000Z")
        self.assertEqual(pacs.model_version(dict(model_type="nnunet_large", members=members[:1]), "x:1"),
                         "nnunet_large single model, x:1")

    def test_run_requires_version_and_container(self):
        for version, container in [(None, "nnunet-vsseg:test"), ("test", None)]:
            with self.subTest(version=version, container=container), \
                    self.assertRaisesRegex(RuntimeError, "CONTAINER_VERSION"):
                pacs.run_pacs("/missing", "/missing", version=version, container=container)


@unittest.skipUnless(shutil.which("jq"), "entrypoint.sh validates ROR_CONT_OPTIONS with jq, "
                                         "which is not installed on this machine")
class EntrypointTests(unittest.TestCase):
    def test_boolean_options_default_and_rejections(self):
        entrypoint = Path(__file__).resolve().parents[1] / "integration/entrypoint.sh"
        for options, expected in [("", "--tta"), ("{}", "--tta"), ('{"tta":true}', "--tta"),
                                  ('{"tta":false}', "--no-tta"),
                                  ('{"model-type":"nnunet_large"}', "--tta"),
                                  ('{"model-type":"nnunet_large","tta":true}', "--tta"),
                                  ('""', None), ('{"tta":""}', None), ('{"--tta":""}', None),
                                  ('{"tta":"false"}', None), ('{"tta":null}', None),
                                  ('{"model-type":"large"}', None), ('{"model-type":"nnunet_medium"}', None),
                                  ('{"other":1}', None), (" ", None),
                                  ('{"tta":true};touch /tmp/invalid', None),
                                  # Exactly one JSON value: jq would judge only the last one.
                                  ('{"tta":false}{"tta":false}', None), ('{"tta":false} {}', None),
                                  ('{"tta":"no"} {}', None), ('{"model-type":"other"} {}', None),
                                  ("[] {}", None), ("{} {}", None)]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "python").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
                (root / "python").chmod(0o755)
                (root / "shell-env").write_text('conda() { return 0; }\nexport -f conda\n')
                (root / "deployment_manifest.json").write_text('{"model_type": "nnunet_large"}')
                env = dict(os.environ, PATH=str(root) + ":" + os.environ["PATH"],
                           BASH_ENV=str(root / "shell-env"), ROR_CONT_OPTIONS=options,
                           NNUNET_MODEL_DIR=str(root))
                result = subprocess.run(["bash", str(entrypoint)], env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if expected is None:
                    self.assertNotEqual(result.returncode, 0, options)
                    self.assertIn(f"invalid ROR_CONT_OPTIONS: {options}", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines(), ["-m", "nnunet_inference.pacs",
                                     "/data/input", "/output", expected,
                                     "--model-type", "nnunet_large"])
