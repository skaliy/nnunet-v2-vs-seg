"""Write the handoff README and ror select entry next to a release archive."""
import argparse
import json
from pathlib import Path
import subprocess

PACKAGE = "nnunet_inference"
PACKAGE_SUFFIXES = {".py", ".json", ".csv"}  # the files build.sh copies from it


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
Needs at least {folds * manifest["cpu"]["threads_per_fold"]} CPU threads and {manifest["cpu"]["memory_limit_gib"]} GiB memory.

## REDCap

Each predicted region goes to its own repeat instance of the `pr2mask` instrument
({instance_base + 1}, {instance_base + 2}, ...). An empty mask gives instance
{instance_base + 1} with blank measurements. `model_version` is filled in every instance, here
`{manifest["model_type"]} {folds}-fold, {image_tag}`.

Built from nnunet-v2-vs-seg commit {source} (Fiona base {base_ref.rsplit(":", 1)[-1][:8]}).
'''


def source_description(project):
    """The commit the image was built from, noting changes that the image may contain.

    build.sh copies the package's top-level .py, .json and .csv files as they are
    on disk, so such a file that Git does not track, ignored or not, is a change.
    """
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(project), *arguments], text=True)

    source = git("rev-parse", "--short=7", "HEAD").strip()
    changed = git("status", "--porcelain", "--untracked-files=no").strip()
    untracked = map(Path, git("ls-files", "--others", "-z", "--", PACKAGE).split("\0"))
    copied = [path for path in untracked
              if path.parent == Path(PACKAGE) and path.suffix in PACKAGE_SUFFIXES]
    if changed or copied:
        source += " with uncommitted changes"
    return source


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
    source = source_description(project)
    (args.release_dir / "README.md").write_text(handoff_readme(
        args.image_tag, inspect(args.image_tag)["Id"], args.base_image, source,
        registry[manifest["bundle_sha256"]], manifest))
    (args.release_dir / "ror_select_entry.txt").write_text(ror_select_entry(latest_tag(args.image_tag)))


if __name__ == "__main__":
    main()
