#!/usr/bin/env python3
"""Compare the outputs of Research PACS container runs.

    compare_outputs.py snapshot OUTPUT_DIR SNAPSHOT.json
    compare_outputs.py stable FIRST.json SECOND.json BASELINE.json
    compare_outputs.py check BASELINE.json CANDIDATE.json
    compare_outputs.py uncompared BASELINE.json

A snapshot is a flat {key: value} map of everything one run wrote: every DICOM
element including the file meta, every column of every REDCap row, every
member of the data dictionary, and every other file by path (hashed unless
another entry covers it). ``stable`` compares two reference runs of the same
image. It writes the entries on which they agree as stable, and lists as
unstable the keys whose values differ or that only one run had, such as
timestamps and memory figures. ``check`` requires a candidate run to match
every stable entry, to have every unstable entry that both reference runs had
(with any value), and to have no entry that the reference runs did not have.
``uncompared`` names the pixel groups on which the two reference runs disagree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import warnings
import zipfile

from pydicom import dcmread

DICOM_OUTPUTS = ("labels", "fused", "fused_vote_map", "reports")
REDCAP_DESTINATION = ("record_id", "redcap_event_name", "redcap_repeat_instrument",
                      "redcap_repeat_instance", "field_name")
LONG_VALUE = 200
# pr2mask prints objects with their memory address, as in "ImageRegion (0x7628...)".
MEMORY_ADDRESS = re.compile(r"\(0x[0-9a-fA-F]+\)")
LISTED = "present"


def _digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _text(value):
    """A JSON string for one value, without memory addresses; long values are
    replaced by their hash."""
    text = MEMORY_ADDRESS.sub("(0x...)", json.dumps(value, sort_keys=True, default=str))
    return _digest(text.encode()) if len(text) > LONG_VALUE else text


def _slice_order(item):
    path, dataset = item
    position = tuple(float(v) for v in dataset.get("ImagePositionPatient", ()))
    return int(dataset.get("InstanceNumber", 0) or 0), position, str(path)


def _elements(dataset, prefix):
    """One entry per element of a dataset or its file meta."""
    values = dataset.to_json_dict()
    entries = {}
    for element in dataset:
        value = values[f"{element.tag.group:04X}{element.tag.element:04X}"]
        entries[f"{prefix}{element.keyword or element.tag}"] = _text(
            value.get("Value", value.get("InlineBinary", "")))
    return entries


def snapshot_dicom(directory, name):
    """Entries of one DICOM output. Files are keyed by slice order, because
    file names and SOP Instance UIDs need not be the same in two runs.
    pydicom's warnings about invalid values are not shown."""
    directory = Path(directory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        files = [(path.relative_to(directory), dcmread(path))
                 for path in sorted(directory.rglob("*")) if path.is_file()]
        entries = {f"{name}#files": str(len(files))}
        for index, (path, dataset) in enumerate(sorted(files, key=_slice_order)):
            key = f"{name}/{index:04d}"
            entries[f"{key}#path"] = str(path)
            if "PixelData" in dataset:
                entries[f"{key}#pixels"] = _digest(dataset.PixelData)
                del dataset.PixelData
            entries.update(_elements(dataset, f"{key}#"))
            entries.update(_elements(dataset.file_meta, f"{key}#meta/"))
    return entries


def _reports(directory):
    """The REDCap report folders, in the order of their snapshot keys."""
    return sorted(path.parent for path in Path(directory).glob("*/output.json"))


def snapshot_redcap(directory):
    """Entries of the REDCap reports: the number of rows, every column of every
    row and every member of the data dictionary."""
    entries = {}
    reports = _reports(directory)
    entries["redcap#reports"] = str(len(reports))
    for index, report in enumerate(reports):
        key = f"redcap/{index}"
        entries[f"{key}#directory"] = report.name
        rows = json.loads((report / "output.json").read_text())
        entries[f"{key}#rows"] = str(len(rows))
        for row in rows:
            destination = "/".join(str(row.get(column)) for column in REDCAP_DESTINATION)
            for column, value in row.items():
                if column not in REDCAP_DESTINATION:
                    suffix = "" if column == "value" else f"[{column}]"
                    entries[f"{key}#{destination}{suffix}"] = _text(value)
        dictionary = report / "output_data_dictionary.zip"
        if dictionary.is_file():
            with zipfile.ZipFile(dictionary) as archive:
                for member in sorted(archive.namelist()):
                    entries[f"{key}#dictionary/{member}"] = _digest(archive.read(member))
    return entries


def snapshot_files(output_dir):
    """One entry per file outside the DICOM outputs, keyed by its path (inside
    a REDCap report folder by the report's index). Files that other entries
    cover, and pacs_command.log with its timings, are listed; the others are
    hashed."""
    output_dir = Path(output_dir)
    reports = {report: f"redcap/{index}" for index, report in enumerate(_reports(output_dir / "redcap"))}
    covered = {output_dir / "runtime_resources.json", output_dir / "pacs_command.log"}
    covered |= {report / name for report in reports
                for name in ("output.json", "output_data_dictionary.zip")}
    entries = {}
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir)
        if not path.is_file() or (len(relative.parts) > 1 and relative.parts[0] in DICOM_OUTPUTS):
            continue
        report = next((report for report in reports if path.is_relative_to(report)), None)
        key = (f"{reports[report]}#file/{path.relative_to(report).as_posix()}" if report
               else f"#file/{relative.as_posix()}")
        entries[key] = LISTED if path in covered else _digest(path.read_bytes())
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
    entries.update(snapshot_files(output_dir))
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


def _pixel_counts(keys):
    return {group: count for group, count in _counts(keys).items() if group.endswith(" pixels")}


def stable(first, second):
    """Entries with the same value in both runs, the keys that differ, and
    those of the differing keys that both runs have."""
    kept = {key: value for key, value in first.items() if second.get(key) == value}
    dropped = sorted((set(first) | set(second)) - set(kept))
    in_both = sorted((set(first) & set(second)) - set(kept))
    return kept, dropped, in_both


def check(baseline, candidate, unstable=(), unstable_in_both=()):
    """Keys that differ: those of the baseline that the candidate lacks or has
    with another value, those of the candidate that are neither in the
    baseline nor unstable, and the unstable keys of both reference runs that
    the candidate lacks."""
    unstable = set(unstable)
    changed = {key for key, value in baseline.items() if candidate.get(key) != value}
    added = {key for key in candidate if key not in baseline and key not in unstable}
    missing = {key for key in unstable_in_both if key not in candidate}
    return sorted(changed | added | missing)


def _load(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    """Write JSON through a temporary file in the same folder, so that a killed
    run leaves no truncated file that looks finished."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=1, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, arguments in (("snapshot", ("output_dir", "snapshot")),
                            ("stable", ("first", "second", "baseline")),
                            ("check", ("baseline", "candidate")),
                            ("uncompared", ("baseline",))):
        command = commands.add_parser(name)
        for argument in arguments:
            command.add_argument(argument, type=Path)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        entries = snapshot(args.output_dir)
        _write(args.snapshot, entries)
        print(f"{len(entries)} entries -> {args.snapshot}")
        return 0
    if args.command == "stable":
        kept, dropped, in_both = stable(_load(args.first), _load(args.second))
        _write(args.baseline, dict(stable=kept, unstable=dropped, unstable_in_both=in_both))
        print(f"stable: {_counts(kept)}")
        print(f"not stable between the two runs: {_counts(dropped)}")
        pixels = _pixel_counts(dropped)
        for group, count in pixels.items():
            if group != "labels pixels":
                print(f"Warning: the two runs disagree on {count} entries of '{group}'; "
                      "they are not compared", file=sys.stderr)
        if "labels pixels" in pixels:
            print("Error: the two runs disagree on the mask pixels", file=sys.stderr)
            return 1
        return 0
    baseline = _load(args.baseline)
    if args.command == "uncompared":
        pixels = _pixel_counts(baseline["unstable"])
        if pixels:
            print(", ".join(f"{group} ({count} {'entry' if count == 1 else 'entries'})"
                            for group, count in pixels.items()))
        return 0
    different = check(baseline["stable"], _load(args.candidate), baseline["unstable"],
                      baseline["unstable_in_both"])
    print(f"compared: {_counts(baseline['stable'])}")
    if different:
        print(f"Error: {len(different)} entries differ: {_counts(different)}", file=sys.stderr)
        for key in different[:20]:
            print(f"  {key}", file=sys.stderr)
        return 1
    print("candidate matches the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
