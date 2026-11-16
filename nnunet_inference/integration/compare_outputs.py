#!/usr/bin/env python3
"""Compare the outputs of Research PACS container runs.

    compare_outputs.py snapshot OUTPUT_DIR SNAPSHOT.json
    compare_outputs.py stable FIRST.json SECOND.json BASELINE.json
    compare_outputs.py check BASELINE.json CANDIDATE.json

A snapshot is a flat {key: value} map of everything one run wrote. ``stable``
keeps the entries on which two runs of the same image agree, which drops
timestamps and memory figures. ``check`` requires a candidate run to match
every entry of that baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from pydicom import dcmread

DICOM_OUTPUTS = ("labels", "fused", "fused_vote_map", "reports")
LONG_VALUE = 200


def _digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _text(value):
    """A JSON string for one value; long values are replaced by their hash."""
    text = json.dumps(value, sort_keys=True, default=str)
    return _digest(text.encode()) if len(text) > LONG_VALUE else text


def _slice_order(item):
    path, dataset = item
    position = tuple(float(v) for v in dataset.get("ImagePositionPatient", ()))
    return int(dataset.get("InstanceNumber", 0) or 0), position, str(path)


def snapshot_dicom(directory, name):
    """Entries of one DICOM output. Files are keyed by slice order, because
    file names and SOP Instance UIDs need not be the same in two runs."""
    directory = Path(directory)
    files = [(path.relative_to(directory), dcmread(path))
             for path in sorted(directory.rglob("*")) if path.is_file()]
    entries = {f"{name}#files": str(len(files))}
    for index, (path, dataset) in enumerate(sorted(files, key=_slice_order)):
        key = f"{name}/{index:04d}"
        entries[f"{key}#path"] = str(path)
        if "PixelData" in dataset:
            entries[f"{key}#pixels"] = _digest(dataset.PixelData)
            del dataset.PixelData
        values = dataset.to_json_dict()
        for element in dataset:
            value = values[f"{element.tag.group:04X}{element.tag.element:04X}"]
            entries[f"{key}#{element.keyword or element.tag}"] = _text(
                value.get("Value", value.get("InlineBinary", "")))
    return entries


def snapshot_redcap(directory):
    entries = {}
    reports = sorted(path.parent for path in Path(directory).glob("*/output.json"))
    entries["redcap#reports"] = str(len(reports))
    for index, report in enumerate(reports):
        key = f"redcap/{index}"
        entries[f"{key}#directory"] = report.name
        for row in json.loads((report / "output.json").read_text()):
            destination = "/".join(str(row[k]) for k in (
                "record_id", "redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance"))
            entries[f"{key}#{destination}/{row['field_name']}"] = _text(row["value"])
        dictionary = report / "output_data_dictionary.zip"
        if dictionary.is_file():
            with zipfile.ZipFile(dictionary) as archive:
                for member in sorted(archive.namelist()):
                    entries[f"{key}#dictionary/{member}"] = _digest(archive.read(member))
    return entries


def snapshot(output_dir):
    output_dir = Path(output_dir)
    entries = {"#entries": _text(sorted(path.name for path in output_dir.iterdir()))}
    for name in DICOM_OUTPUTS:
        if (output_dir / name).is_dir():
            entries.update(snapshot_dicom(output_dir / name, name))
    if (output_dir / "redcap").is_dir():
        entries.update(snapshot_redcap(output_dir / "redcap"))
    resources = output_dir / "runtime_resources.json"
    if resources.is_file():
        for key, value in json.loads(resources.read_text()).items():
            entries[f"runtime_resources#{key}"] = _text(value)
    return entries


def _group(key):
    """'labels/0003#pixels' -> 'labels pixels'; other tags are counted together."""
    name, _, field = key.partition("#")
    name = name.split("/")[0] or "run"
    return f"{name} {'pixels' if field == 'pixels' else 'other'}"


def _counts(keys):
    counts = {}
    for key in keys:
        counts[_group(key)] = counts.get(_group(key), 0) + 1
    return counts


def stable(first, second):
    """Entries with the same value in both runs, and the keys that differ."""
    kept = {key: value for key, value in first.items() if second.get(key) == value}
    dropped = sorted((set(first) | set(second)) - set(kept))
    return kept, dropped


def check(baseline, candidate):
    """Keys of the baseline that the candidate lacks or has with another value."""
    return sorted(key for key, value in baseline.items() if candidate.get(key) != value)


def _load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, arguments in (("snapshot", ("output_dir", "snapshot")),
                            ("stable", ("first", "second", "baseline")),
                            ("check", ("baseline", "candidate"))):
        command = commands.add_parser(name)
        for argument in arguments:
            command.add_argument(argument, type=Path)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        entries = snapshot(args.output_dir)
        args.snapshot.write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")
        print(f"{len(entries)} entries -> {args.snapshot}")
        return 0
    if args.command == "stable":
        kept, dropped = stable(_load(args.first), _load(args.second))
        args.baseline.write_text(json.dumps(kept, indent=1, sort_keys=True) + "\n")
        print(f"stable: {_counts(kept)}")
        print(f"not stable between the two runs: {_counts(dropped)}")
        if any(_group(key) == "labels pixels" for key in dropped):
            print("Error: the two runs disagree on the mask pixels", file=sys.stderr)
            return 1
        return 0
    baseline = _load(args.baseline)
    different = check(baseline, _load(args.candidate))
    print(f"compared: {_counts(baseline)}")
    if different:
        print(f"Error: {len(different)} entries differ: {_counts(different)}", file=sys.stderr)
        for key in different[:20]:
            print(f"  {key}", file=sys.stderr)
        return 1
    print("candidate matches the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
