"""Write reproducible handoff metadata; qualification is recorded separately."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(image):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]


def handoff_readme(image_tag, repeat_instance):
    version = image_tag.rsplit(":", 1)[-1]
    return f'''# nnU-Net PACS handoff — {version}

CPU-only Dataset003 ResEnc-M, using five fold-best models.
Image: `{image_tag}` (`latest` is also included).

## Run

Defaults: **TTA on**, five parallel folds, **three CPU threads per fold**.
Disable TTA with `ROR_CONT_OPTIONS='{{"tta":false}}'`.

```bash
docker run --rm \\
  -v /absolute/path/dicom-series:/data/input:ro \\
  -v /absolute/path/empty-output:/output \\
  {image_tag}
```

Use a fresh output folder. Outputs: `labels` (the mask), `fused`, `fused_vote_map`,
`reports` and `redcap`; reports may be empty for an empty mask.

## REDCap

Export includes the recoverable mask, geometry, measurements and provenance,
using **repeat instance {repeat_instance}**. Install the emitted `output_data_dictionary.zip`;
the importer must update the supplied repeat instance.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("image_tag")
    parser.add_argument("base_image")
    args = parser.parse_args()
    integration = Path(__file__).resolve().parent
    project = integration.parent.parent
    release = args.release_dir
    image, base = inspect(args.image_tag), inspect(args.base_image)
    manifest = json.loads((integration / "model/deployment_manifest.json").read_text())
    registry = json.loads((integration / "_pkg/nnunet_inference/redcap_model_instances.json").read_text())
    fingerprints = {}
    for root in (integration / "_pkg",):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in str(path):
                fingerprints[str(path.relative_to(integration))] = sha256(path)
    for name in ("entrypoint.sh", "requirements.yml", ".ror/virt/Dockerfile"):
        fingerprints[name] = sha256(integration / name)
    state = dict(image_tag=args.image_tag, image_id=image["Id"],
                 base_id=base["Id"], base_repo_digests=base.get("RepoDigests", []),
                 source_commit=subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
                 source_has_changes=bool(subprocess.check_output(["git", "-C", str(project), "status", "--porcelain"], text=True).strip()),
                 packaged_files_sha256=fingerprints, model=manifest,
                 redcap_repeat_instance=registry[manifest["bundle_sha256"]],
                 tta_default=True, cpu_threads=16, memory_limit_gib=64,
                 qualification="Image integrity checked; see qualification.json for completed inference checks.")
    (release / "release.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    (release / "README.md").write_text(handoff_readme(
        args.image_tag, state["redcap_repeat_instance"]))


if __name__ == "__main__":
    main()
