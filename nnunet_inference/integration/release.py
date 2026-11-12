"""Write the handoff README and ror select entry next to a release archive."""
import argparse
import json
from pathlib import Path
import subprocess


def inspect(image):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]


def latest_tag(image_tag):
    return image_tag.rsplit(":", 1)[0] + ":latest"


def ror_select_entry(image_tag):
    return f'''"ENTRY_NAME": {{
  "select": "<VS CE-T1 series selector>",
  "ROR_CONT_OPTIONS": "{{}}",
  "docker_image": "{image_tag}"
}},
'''


def handoff_readme(image_tag, image_id, base_ref, source, instance_base, manifest):
    name, version = image_tag.rsplit(":", 1)
    latest = latest_tag(image_tag)
    folds = len(manifest["members"])
    return f'''# {name} {version}

{manifest["model_name"]}, each fold's `{manifest["checkpoint"]}`. CPU only, TTA on.

## Load

```bash
docker load -i {name}-{version}.tar.gz
docker image inspect {latest} --format '{{{{.Id}}}}'
```

The image ID should be `{image_id}`.

The archive tags the image `{image_tag}` and `{latest}`.
The PACS entry uses `latest` (`ror_select_entry.txt`), so no config change is needed.

## Run

`"ROR_CONT_OPTIONS": "{{}}"` runs the defaults (TTA on); `"{{\\"tta\\":false}}"` turns TTA off.
Needs at least 15 CPU threads and 64 GiB memory.

## REDCap

Each predicted region goes to its own repeat instance of the `pr2mask` instrument
({instance_base + 1}, {instance_base + 2}, ...). An empty mask gives instance
{instance_base + 1} with blank measurements. `model_version` is filled in every instance, here
`{manifest["model_type"]} {folds}-fold, {image_tag}`.

Built from nnunet-v2-vs-seg commit {source} (Fiona base {base_ref.rsplit(":", 1)[-1][:8]}).
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("image_tag")
    parser.add_argument("base_image")
    args = parser.parse_args()
    integration = Path(__file__).resolve().parent
    project = integration.parent.parent
    manifest = json.loads((integration / "model/deployment_manifest.json").read_text())
    registry = json.loads((integration / "_pkg/nnunet_inference/redcap_model_instances.json").read_text())
    git = ["git", "-C", str(project)]
    source = subprocess.check_output([*git, "rev-parse", "--short=7", "HEAD"], text=True).strip()
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        source += " with uncommitted changes"
    (args.release_dir / "README.md").write_text(handoff_readme(
        args.image_tag, inspect(args.image_tag)["Id"], args.base_image, source,
        registry[manifest["bundle_sha256"]], manifest))
    (args.release_dir / "ror_select_entry.txt").write_text(ror_select_entry(latest_tag(args.image_tag)))


if __name__ == "__main__":
    main()
