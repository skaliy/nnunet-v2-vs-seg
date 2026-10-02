import json
import os
from pathlib import Path
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
        medium = next(digest for digest, n in registry.items() if n == 3)
        deployment = dict(model_type="nnunet_medium", model_name="Medium", bundle_sha256=medium,
                          members=[dict(member_id=f"fold_{n}") for n in range(5)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, *_ = fixture(root)
            for name in ("vote_map", "fused", "fused_vote_map", "reports"):
                (work / name).mkdir()
            output = root / "output"
            output.mkdir()
            with patch.object(pacs.subprocess, "run") as run:
                pacs.postprocess(source, work, output, deployment, version="test",
                                 use_tta=True, pr2mask_dir=Path("/pr2mask"))
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertIn("test_nn_m1_b" + medium[:32] + "_t1", command)
            self.assertEqual({p.name for p in output.iterdir()}, set(pacs.FINAL_OUTPUTS) | {pacs.LOG_NAME})
            rows = json.loads(next((output / "redcap").glob("*/output.json")).read_text())
            self.assertTrue(all(r["redcap_repeat_instance"] == "3" for r in rows))
            self.assertFalse((output / "vote_map").exists())

    def test_failed_postprocessing_does_not_publish_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work, *_ = fixture(root)
            output = root / "output"
            output.mkdir()
            deployment = dict(model_name="Medium", bundle_sha256="a" * 64)
            with patch.object(pacs.subprocess, "run", side_effect=RuntimeError("tool failure")):
                with self.assertRaisesRegex(RuntimeError, "tool failure"):
                    pacs.postprocess(source, work, output, deployment, version="test",
                                     use_tta=True, pr2mask_dir=Path("/pr2mask"))
            self.assertFalse(any(output.iterdir()))


class EntrypointTests(unittest.TestCase):
    def test_boolean_options_default_and_rejections(self):
        entrypoint = Path(__file__).resolve().parents[1] / "integration/entrypoint.sh"
        for options, expected in [("{}", "--tta"), ('{"tta":false}', "--no-tta"),
                                  ('{"model-type":"nnunet_medium","tta":true}', "--tta"),
                                  ('{"tta":"false"}', None), ('{"tta":null}', None),
                                  ('{"model-type":"large"}', None), ('{"other":1}', None),
                                  ('{"tta":true};touch /tmp/invalid', None)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "python").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
                (root / "python").chmod(0o755)
                (root / "shell-env").write_text('conda() { return 0; }\nexport -f conda\n')
                env = dict(os.environ, PATH=str(root) + ":" + os.environ["PATH"],
                           BASH_ENV=str(root / "shell-env"), ROR_CONT_OPTIONS=options)
                result = subprocess.run(["bash", str(entrypoint)], env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if expected is None:
                    self.assertNotEqual(result.returncode, 0, options)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines(), ["-m", "nnunet_inference.pacs",
                                     "/data/input", "/output", expected])
