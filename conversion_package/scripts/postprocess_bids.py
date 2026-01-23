from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

def _scrub_sys_path() -> None:
    py_path = os.environ.get("PYTHONPATH")
    if not py_path:
        return

    import site

    protected = {os.path.abspath(p) for p in site.getsitepackages()}
    entries = {os.path.abspath(p) for p in py_path.split(os.pathsep) if p}
    to_remove = entries - protected
    if to_remove:
        sys.path = [p for p in sys.path if os.path.abspath(p) not in to_remove]


_scrub_sys_path()

import numpy as np
import pyxdf


@dataclass(frozen=True)
class XdfStreams:
    eeg: dict[str, Any]
    pupil: dict[str, Any]
    markers: list[dict[str, Any]]


PUPIL_LABS_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "confidence": "Pupil Labs gaze confidence (0-1).",
    "norm_pos_x": "Gaze x-position in normalized coordinates.",
    "norm_pos_y": "Gaze y-position in normalized coordinates.",
    "gaze_point_3d_x": "3D gaze point x-coordinate.",
    "gaze_point_3d_y": "3D gaze point y-coordinate.",
    "gaze_point_3d_z": "3D gaze point z-coordinate.",
    "eye_center0_3d_x": "Eye center (eye 0) 3D x-coordinate.",
    "eye_center0_3d_y": "Eye center (eye 0) 3D y-coordinate.",
    "eye_center0_3d_z": "Eye center (eye 0) 3D z-coordinate.",
    "eye_center1_3d_x": "Eye center (eye 1) 3D x-coordinate.",
    "eye_center1_3d_y": "Eye center (eye 1) 3D y-coordinate.",
    "eye_center1_3d_z": "Eye center (eye 1) 3D z-coordinate.",
    "gaze_normal0_x": "Gaze normal vector (eye 0) x-component.",
    "gaze_normal0_y": "Gaze normal vector (eye 0) y-component.",
    "gaze_normal0_z": "Gaze normal vector (eye 0) z-component.",
    "gaze_normal1_x": "Gaze normal vector (eye 1) x-component.",
    "gaze_normal1_y": "Gaze normal vector (eye 1) y-component.",
    "gaze_normal1_z": "Gaze normal vector (eye 1) z-component.",
    "diameter0_2d": "Pupil diameter estimate (eye 0) in 2D.",
    "diameter1_2d": "Pupil diameter estimate (eye 1) in 2D.",
    "diameter0_3d": "Pupil diameter estimate (eye 0) in 3D.",
    "diameter1_3d": "Pupil diameter estimate (eye 1) in 3D."
}
PUPIL_LABS_DROP_LABELS = {
    "gaze_point_3d_x",
    "gaze_point_3d_y",
    "gaze_point_3d_z",
}


def _parse_pupil_channel_units(pupil_stream: dict[str, Any]) -> dict[str, str]:
    desc = pupil_stream.get("info", {}).get("desc", [None])[0]
    if not desc:
        return {}
    try:
        channels = desc["channels"][0]["channel"]
        units = {c["label"][0]: c.get("unit", ["n/a"])[0] for c in channels}
        return units
    except Exception:
        return {}


def _bids_task_from_root(bids_root: Path) -> str:
    name = bids_root.name.lower()
    if "nback" in name:
        return "nback"
    raise ValueError(f"Could not infer task from BIDS root name: {bids_root}")


def _resolve_bids_roots(
    repo_root: Path,
    bids_root: str | None,
    bids_roots: list[str] | None,
) -> list[Path]:
    if bids_root:
        roots = [bids_root]
    elif bids_roots:
        roots = bids_roots
    else:
        candidates = [
            name
            for name in ("bids_nback",)
            if (repo_root / name).exists()
        ]
        if len(candidates) == 1:
            roots = candidates
        elif len(candidates) > 1:
            raise ValueError(
                "Multiple BIDS roots found. Pass --bids-root or --bids-roots."
            )
        else:
            raise FileNotFoundError("No BIDS roots found. Pass --bids-root or --bids-roots.")
    return [(repo_root / p).resolve() for p in roots]


def _subject_from_xdf_filename(xdf_path: Path) -> str:
    match = re.match(r"^(?P<id>\d{3})_", xdf_path.stem)
    if not match:
        raise ValueError(f"Unexpected XDF filename (expected 'NNN_*.xdf'): {xdf_path.name}")
    return f"sub-{match.group('id')}"


def _estimate_sampling_frequency(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return float("nan")
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return float("nan")
    return 1.0 / median_dt


def _parse_pupil_channel_labels(pupil_stream: dict[str, Any]) -> list[str]:
    desc = pupil_stream.get("info", {}).get("desc", [None])[0]
    if not desc:
        return [f"ch{idx + 1:02d}" for idx in range(pupil_stream["time_series"].shape[1])]
    try:
        channels = desc["channels"][0]["channel"]
        labels = [c["label"][0] for c in channels]
        if len(labels) != pupil_stream["time_series"].shape[1]:
            raise ValueError("Label count does not match data channel count.")
        return labels
    except Exception:
        return [f"ch{idx + 1:02d}" for idx in range(pupil_stream["time_series"].shape[1])]


def _load_xdf_streams(xdf_path: Path) -> XdfStreams:
    try:
        streams, _ = pyxdf.load_xdf(
            str(xdf_path),
            select_streams=[{"name": "EEGStream"}, {"name": "pupil_capture"}],
        )
    except TypeError:
        streams, _ = pyxdf.load_xdf(str(xdf_path))

    eeg = None
    pupil = None
    markers: list[dict[str, Any]] = []
    for s in streams:
        name = s["info"]["name"][0]
        stype = s["info"].get("type", [""])[0]
        if name == "EEGStream":
            eeg = s
        elif name == "pupil_capture":
            pupil = s
        elif stype == "Markers":
            markers.append(s)
    if eeg is None:
        raise ValueError(f"EEGStream not found in {xdf_path}")
    if pupil is None:
        raise ValueError(f"pupil_capture not found in {xdf_path}")
    return XdfStreams(eeg=eeg, pupil=pupil, markers=markers)


def _write_physio_tsv(
    path: Path,
    header: list[str],
    rows: Iterable[list[Any]],
    *,
    write_header: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as text_f:
        writer = csv.writer(text_f, delimiter="\t", lineterminator="\n")
        if write_header:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _ensure_bidsignore(bids_root: Path) -> None:
    ignore_path = bids_root / ".bidsignore"
    lines = [
        "/sub-*/ecg",
        "/sub-*/pupil",
    ]
    ignore_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_value(value: float, decimals: int = 9) -> str:
    if not np.isfinite(value):
        raise ValueError("Non-finite value encountered in physio export.")
    return f"{value:.{decimals}f}"


def _remove_legacy_physio_files(bids_root: Path) -> None:
    patterns = [
        "sub-*/eeg/*_recording-*_physio.tsv",
        "sub-*/eeg/*_recording-*_physio.tsv.gz",
        "sub-*/eeg/*_recording-*_physio.json",
        "sub-*/ecg/*_recording-*_physio.tsv.gz",
        "sub-*/eyetrack/*_recording-eyetrack_physio.tsv",
        "sub-*/eyetrack/*_recording-eyetrack_physio.tsv.gz",
        "sub-*/eyetrack/*_recording-eyetrack_physio.json",
        "sub-*/eyetrack/*_eyetrack.tsv",
        "sub-*/eyetrack/*_eyetrack.tsv.gz",
        "sub-*/eyetrack/*_eyetrack.json",
        "sub-*/pupil/*_pupil.tsv",
        "sub-*/pupil/*_pupil.tsv.gz",
        "sub-*/pupil/*_eyetrack.json",
    ]
    removed = 0
    for pattern in patterns:
        for path in bids_root.glob(pattern):
            path.unlink()
            removed += 1
    for pupil_dir in bids_root.glob("sub-*/pupil"):
        if pupil_dir.exists() and not any(pupil_dir.iterdir()):
            pupil_dir.rmdir()
    for eyetrack_dir in bids_root.glob("sub-*/eyetrack"):
        if eyetrack_dir.exists() and not any(eyetrack_dir.iterdir()):
            eyetrack_dir.rmdir()
    if removed:
        print(f"Removed {removed} legacy physio file(s) from {bids_root}")


def add_eyetrack_from_xdf(*, bids_root: Path, overwrite: bool) -> None:
    task = _bids_task_from_root(bids_root)
    xdf_dir = bids_root / "sourcedata" / "xdf"
    xdf_paths = sorted(xdf_dir.glob("*.xdf"))
    if not xdf_paths:
        raise FileNotFoundError(f"No XDF files found in {xdf_dir}")

    for xdf_path in xdf_paths:
        subject = _subject_from_xdf_filename(xdf_path)
        streams = _load_xdf_streams(xdf_path)
        eeg_start = float(streams.eeg["time_stamps"][0])

        pupil_ts = streams.pupil["time_stamps"].astype(np.float64)
        pupil_data = streams.pupil["time_series"].astype(np.float64)
        labels = _parse_pupil_channel_labels(streams.pupil)
        units_by_label = _parse_pupil_channel_units(streams.pupil)
        pupil_desc = streams.pupil.get("info", {}).get("desc", [None])[0] or {}
        relay_version = None
        if isinstance(pupil_desc, dict):
            relay_version = pupil_desc.get("pupil_lsl_relay_version", [None])[0]
        software_versions = f"Pupil LSL Relay {relay_version}" if relay_version else "n/a"

        rel_time = pupil_ts - eeg_start
        valid_rows = np.isfinite(rel_time) & np.isfinite(pupil_ts)
        if not np.any(valid_rows):
            raise ValueError(f"No finite pupil timestamps available in {xdf_path}")
        rel_time = rel_time[valid_rows]
        pupil_ts = pupil_ts[valid_rows]
        pupil_data = pupil_data[valid_rows, :]
        rel_time_from_start = pupil_ts - float(pupil_ts[0])

        sampling_frequency = _estimate_sampling_frequency(pupil_ts)
        if not math.isfinite(sampling_frequency):
            sampling_frequency = 0.0

        out_dir = bids_root / subject / "pupil"
        out_tsv = out_dir / f"{subject}_task-{task}_pupil.tsv"
        out_json = out_dir / f"{subject}_task-{task}_eyetrack.json"
        if out_tsv.exists() and not overwrite:
            print(f"Pupil outputs already exist for {subject}; skipping.")
            continue

        label_index = {label: idx for idx, label in enumerate(labels)}
        finite_by_label = np.isfinite(pupil_data).all(axis=0)
        keep_indices = [
            idx for idx, keep in enumerate(finite_by_label)
            if keep and labels[idx] not in PUPIL_LABS_DROP_LABELS
        ]
        kept_labels = [labels[idx] for idx in keep_indices]
        kept_units_by_label = {label: units_by_label.get(label) for label in kept_labels}
        pupil_data_kept = pupil_data[:, keep_indices] if keep_indices else np.empty((pupil_data.shape[0], 0))

        norm_x_idx = label_index.get("norm_pos_x")
        norm_y_idx = label_index.get("norm_pos_y")
        diameter_labels = ["diameter0_2d", "diameter1_2d", "diameter0_3d", "diameter1_3d"]
        diameter_indices = [label_index[label] for label in diameter_labels if label in label_index]
        pupil_size_unit = next(
            (units_by_label.get(label) for label in diameter_labels if units_by_label.get(label)),
            None,
        )

        x_coord = None
        if norm_x_idx is not None and finite_by_label[norm_x_idx]:
            x_coord = np.clip(pupil_data[:, norm_x_idx], 0.0, 1.0)
        y_coord = None
        if norm_y_idx is not None and finite_by_label[norm_y_idx]:
            y_coord = np.clip(pupil_data[:, norm_y_idx], 0.0, 1.0)

        pupil_size = None
        if diameter_indices:
            pupil_size_values: list[float] = []
            for row_idx in range(pupil_data.shape[0]):
                values = [
                    float(pupil_data[row_idx, di])
                    for di in diameter_indices
                    if np.isfinite(pupil_data[row_idx, di])
                ]
                pupil_size_values.append(float(np.mean(values)) if values else float("nan"))
            pupil_size_array = np.array(pupil_size_values, dtype=np.float64)
            pupil_size_array = np.clip(pupil_size_array, 0.0, None)
            if np.isfinite(pupil_size_array).all():
                pupil_size = pupil_size_array

        columns: list[tuple[str, np.ndarray, int]] = [
            ("time", rel_time_from_start, 6),
            ("timestamp", pupil_ts, 9),
        ]
        if x_coord is not None:
            columns.append(("x_coordinate", x_coord, 9))
        if y_coord is not None:
            columns.append(("y_coordinate", y_coord, 9))
        if pupil_size is not None:
            columns.append(("pupil_size", pupil_size, 9))
        for idx, label in enumerate(kept_labels):
            columns.append((label, pupil_data_kept[:, idx], 9))

        def rows() -> Iterable[list[Any]]:
            for row_idx in range(pupil_data.shape[0]):
                yield [
                    _format_value(values[row_idx], decimals)
                    for _, values, decimals in columns
                ]

        header = [name for name, _, _ in columns]
        _write_physio_tsv(out_tsv, header, rows())

        column_metadata: dict[str, Any] = {
            "time": {
                "Description": "Time (s) relative to eye-tracking stream start; add StartTime to align with EEG.",
                "Format": "number",
                "Units": "s",
            },
            "timestamp": {
                "Description": "Original LSL timestamp from the pupil stream.",
                "Format": "number",
                "Units": "s",
            },
        }
        if x_coord is not None:
            column_metadata["x_coordinate"] = {
                "Description": "Gaze x-position in normalized coordinates (0-1) derived from norm_pos_x (clipped to [0, 1]).",
                "Format": "number",
                "Units": "n/a",
            }
        if y_coord is not None:
            column_metadata["y_coordinate"] = {
                "Description": "Gaze y-position in normalized coordinates (0-1) derived from norm_pos_y (clipped to [0, 1]).",
                "Format": "number",
                "Units": "n/a",
            }
        if pupil_size is not None:
            column_metadata["pupil_size"] = {
                "Description": "Mean pupil diameter from available diameter channels (negative values clipped to 0).",
                "Format": "number",
                "Units": pupil_size_unit if pupil_size_unit in ("mm", "s") else "n/a",
            }

        for label in kept_labels:
            entry: dict[str, Any] = {
                "Description": PUPIL_LABS_COLUMN_DESCRIPTIONS.get(label, "Pupil Labs stream channel."),
                "Format": "number",
            }
            unit = kept_units_by_label.get(label)
            if unit in ("mm", "s"):
                entry["Units"] = unit
            column_metadata[label] = entry

        sidecar: dict[str, Any] = {
            "TaskName": task,
            "RecordingType": "continuous",
            "PhysioType": "eyetrack",
            "Manufacturer": "Pupil Labs",
            "ManufacturersModelName": "Pupil Core",
            "SoftwareVersions": software_versions,
            "DeviceSerialNumber": "n/a",
            "SamplingFrequency": float(sampling_frequency),
            "StartTime": float(rel_time[0]) if rel_time.size else 0.0,
            "Columns": header,
            "SourceStreamName": "pupil_capture",
            **column_metadata,
        }
        out_json.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote pupil outputs for {subject}.")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        return list(reader.fieldnames), rows


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_marker_stream_column(*, bids_root: Path, overwrite: bool) -> None:
    _bids_task_from_root(bids_root)
    stream_for_trial = "n-backMarkers"
    dropout_stream = "UoHDataOffsetStream"

    for events_path in bids_root.glob("sub-*/eeg/*_events.tsv"):
        header, rows = _read_tsv(events_path)
        if "marker_stream" in header and not overwrite:
            continue

        if "marker_stream" not in header:
            header = [*header, "marker_stream"]

        for row in rows:
            trial_type = row.get("trial_type", "")
            if trial_type == "dropped_samples":
                row["marker_stream"] = dropout_stream
            else:
                row["marker_stream"] = stream_for_trial

        _write_tsv(events_path, header, rows)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


README_REFERENCE_ENTRIES = [
    "Clewett CJ, Langley P, Bateson AD et al (2016) Non-invasive, home-based electroencephalography hypoglycaemia warning system for personal monitoring using skin surface electrodes: a single-case feasibility study. Healthc Technol Lett 3:2-5. https://doi.org/10.1049/htl.2015.0037",
    "Bateson AD, Asghar AUR (2021) Development and evaluation of a smartphone-based electroencephalography (EEG) system. IEEE Access. https://doi.org/10.1109/ACCESS.2021.3079992",
]


def ensure_readme_references(*, bids_root: Path) -> None:
    readme_path = bids_root / "README"
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")
    missing = [entry for entry in README_REFERENCE_ENTRIES if entry not in text]
    if not missing:
        return
    updated = text.rstrip()
    if "References" not in text:
        updated += "\n\nReferences\n----------"
    updated += "\n\n" + "\n\n".join(missing) + "\n"
    readme_path.write_text(updated, encoding="utf-8")


def _load_metadata_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Sidecar metadata config must be a JSON object.")
    return data


def _get_eeg_defaults(metadata: dict[str, Any], task: str) -> dict[str, Any]:
    eeg_section = metadata.get("eeg", {})
    if not isinstance(eeg_section, dict):
        return {}
    defaults = eeg_section.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    task_overrides = eeg_section.get("tasks", {}).get(task, {})
    if not isinstance(task_overrides, dict):
        task_overrides = {}
    return {**defaults, **task_overrides}


def _get_ecg_defaults(metadata: dict[str, Any]) -> dict[str, Any]:
    physio_section = metadata.get("physio", {})
    if not isinstance(physio_section, dict):
        return {}
    ecg_defaults = physio_section.get("ecg", {})
    if not isinstance(ecg_defaults, dict):
        return {}
    return ecg_defaults


def apply_dataset_description_defaults(*, bids_root: Path, defaults: dict[str, Any]) -> None:
    if not defaults:
        return
    if not isinstance(defaults, dict):
        raise ValueError("dataset_description defaults must be a JSON object.")
    desc_path = bids_root / "dataset_description.json"
    data = _load_json(desc_path) if desc_path.exists() else {}
    data.update(defaults)
    _write_json(desc_path, data)


def apply_eeg_sidecar_defaults(*, bids_root: Path, metadata: dict[str, Any]) -> None:
    task = _bids_task_from_root(bids_root)
    defaults = _get_eeg_defaults(metadata, task)
    if not defaults:
        return

    drop_keys = []
    if "HeadCircumference" not in defaults:
        drop_keys.append("HeadCircumference")

    eeg_jsons = list(bids_root.glob(f"sub-*/eeg/*_task-{task}_eeg.json"))
    task_sidecar = bids_root / f"task-{task}_eeg.json"
    data = _load_json(task_sidecar) if task_sidecar.exists() else {}
    data.pop("RecordingDuration", None)
    if "HardwareFilters" not in defaults:
        data.pop("HardwareFilters", None)
    for key in drop_keys:
        data.pop(key, None)
    data.update(defaults)
    if "TaskName" not in data:
        data["TaskName"] = task
    _write_json(task_sidecar, data)

    inherited_keys = set(defaults.keys())
    for eeg_json in eeg_jsons:
        data = _load_json(eeg_json)
        for key in drop_keys:
            data.pop(key, None)
        for key in inherited_keys:
            data.pop(key, None)
        _write_json(eeg_json, data)


def _collect_unique_column_values(
    bids_root: Path, column: str, *, limit: int | None = None
) -> set[str]:
    values: set[str] = set()
    for events_path in bids_root.glob("sub-*/eeg/*_events.tsv"):
        header, rows = _read_tsv(events_path)
        if column not in header:
            continue
        for row in rows:
            v = row.get(column, "")
            if v and v != "n/a":
                values.add(v)
                if limit is not None and len(values) >= limit:
                    return values
    return values


def update_events_json_definitions(*, bids_root: Path) -> None:
    task = _bids_task_from_root(bids_root)
    events_json = bids_root / f"task-{task}_events.json"
    data = _load_json(events_json) if events_json.exists() else {}
    if "StimulusPresentation" not in data:
        data["StimulusPresentation"] = {}

    trial_types = sorted(_collect_unique_column_values(bids_root, "trial_type"))
    marker_streams = sorted(_collect_unique_column_values(bids_root, "marker_stream"))

    trial_type_levels: dict[str, str] = {}
    nback_re = re.compile(r"^\d+-back$")
    for v in trial_types:
        if v == "dropped_samples":
            trial_type_levels[v] = "Dropout annotation from acquisition (dropped samples)."
        elif nback_re.match(v):
            level = v.split("-", 1)[0]
            trial_type_levels[v] = f"N-back trial with nback_level={level}."
        else:
            trial_type_levels[v] = "Task marker."

    marker_stream_levels: dict[str, str] = {}
    for v in marker_streams:
        if v == "UoHDataOffsetStream":
            marker_stream_levels[v] = "Marker stream reporting dropped sample counts (e.g., 'Dropped N samples')."
        elif v == "n-backMarkers":
            marker_stream_levels[v] = "Task marker stream for the n-back paradigm."
        else:
            marker_stream_levels[v] = "Marker stream."

    def ensure_column(col: str, description: str) -> dict[str, Any]:
        existing = data.get(col, {})
        if not isinstance(existing, dict):
            existing = {}
        if "Description" not in existing:
            existing["Description"] = description
        data[col] = existing
        return existing

    ensure_column("marker_stream", "LSL marker stream name that generated this event.")["Levels"] = marker_stream_levels
    ensure_column("trial_type", "Event type / condition label.")["Levels"] = trial_type_levels
    timestamp_meta = ensure_column("timestamp_lsl", "Original LSL timestamp for this marker.")
    if "Units" not in timestamp_meta:
        timestamp_meta["Units"] = "s"
    ensure_column("marker", "Raw marker string emitted on the LSL marker stream.")
    ensure_column("nback_level", "N-back level (1-4) when applicable.")
    ensure_column("key_press", "Whether a response key was pressed on this trial.")["Levels"] = {
        "true": "Key pressed.",
        "false": "No key press.",
        "True": "Key pressed.",
        "False": "No key press.",
    }
    ensure_column("matched", "Whether the stimulus matched the N-back target on this trial.")["Levels"] = {
        "true": "Stimulus matched.",
        "false": "No match.",
        "True": "Stimulus matched.",
        "False": "No match.",
    }
    dropped_meta = ensure_column("dropped_samples", "Number of dropped samples reported by the dropout stream.")
    if "Units" not in dropped_meta:
        dropped_meta["Units"] = "samples"

    ensure_column("response_accuracy", "1=correct, 0=incorrect (when applicable).")["Levels"] = {
        "1": "Correct response.",
        "0": "Incorrect response.",
    }
    ensure_column("istutorial", "Whether this trial was marked as tutorial (when applicable).")["Levels"] = {
        "True": "Tutorial.",
        "False": "Non-tutorial.",
        "true": "Tutorial.",
        "false": "Non-tutorial.",
    }
    ensure_column("outcome", "Correct/Wrong (when applicable).")["Levels"] = {
        "Correct": "Correct.",
        "Wrong": "Wrong.",
    }

    _write_json(events_json, data)


def _load_ecg_map(ecg_map_path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(ecg_map_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid ECG map format (expected object at top-level).")
    out: dict[str, dict[str, str]] = {}
    for task in ("nback",):
        mapping = data.get(task, {})
        if not isinstance(mapping, dict):
            mapping = {}
        out[task] = {k: str(v) for k, v in mapping.items()}
    return out


def _infer_ecg_channel_from_brainvision(bids_root: Path, subject: str, task: str) -> str:
    try:
        import mne
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mne is required for ECG inference when not provided in the map.") from e

    vhdr = bids_root / subject / "eeg" / f"{subject}_task-{task}_eeg.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    candidate_names = [name for name in raw.ch_names if name in ("D1", "D3")]
    if len(candidate_names) != 2:
        raise ValueError(f"Expected D1 and D3 to exist in {vhdr}")

    raw.pick(candidate_names)
    raw.load_data()
    data = raw.get_data()

    def score(sig: np.ndarray) -> float:
        sig = sig.astype(np.float64)
        sig = sig[np.isfinite(sig)]
        if sig.size < 10:
            return float("-inf")
        sig = sig - float(np.median(sig))
        p99 = float(np.percentile(sig, 99))
        p1 = float(np.percentile(sig, 1))
        return (p99 - p1) / (float(np.std(sig)) + 1e-12)

    scores = {raw.ch_names[idx]: score(data[idx, :]) for idx in range(data.shape[0])}
    return max(scores, key=scores.get)


def fix_ecg_channel_types(*, bids_root: Path, ecg_map_path: Path, overwrite: bool) -> None:
    task = _bids_task_from_root(bids_root)
    mapping = _load_ecg_map(ecg_map_path).get(task, {})

    for channels_path in bids_root.glob("sub-*/eeg/*_channels.tsv"):
        subject = channels_path.parts[-3]
        expected_prefix = f"{subject}_task-{task}_"
        if not channels_path.name.startswith(expected_prefix):
            continue

        header, rows = _read_tsv(channels_path)
        if "name" not in header or "type" not in header:
            raise ValueError(f"Unexpected channels.tsv format: {channels_path}")

        ecg_channel = mapping.get(subject)
        if ecg_channel not in ("D1", "D3"):
            ecg_channel = _infer_ecg_channel_from_brainvision(bids_root, subject, task)

        updated = False
        for row in rows:
            name = row.get("name", "")
            if name not in ("D1", "D2", "D3", "D4", "D5"):
                continue
            if name == ecg_channel:
                if row.get("type") != "ECG" or row.get("description") != "ElectroCardioGram":
                    row["type"] = "ECG"
                    row["description"] = "ElectroCardioGram"
                    updated = True
            else:
                if row.get("type") != "MISC":
                    row["type"] = "MISC"
                    row["description"] = "Auxiliary channel (non-ECG)."
                    updated = True

        if updated or overwrite:
            _write_tsv(channels_path, header, rows)

        eeg_json = channels_path.with_name(channels_path.name.replace("_channels.tsv", "_eeg.json"))
        if eeg_json.exists():
            eeg_meta = _load_json(eeg_json)
            types = [r.get("type", "") for r in rows]
            eeg_meta["EEGChannelCount"] = sum(t == "EEG" for t in types)
            eeg_meta["ECGChannelCount"] = sum(t == "ECG" for t in types)
            misc_count = sum(t == "MISC" for t in types)
            eeg_meta.pop("MiscChannelCount", None)
            eeg_meta["MISCChannelCount"] = misc_count
            eeg_meta["EOGChannelCount"] = sum(t == "EOG" for t in types)
            eeg_meta["EMGChannelCount"] = sum(t == "EMG" for t in types)
            eeg_meta["TriggerChannelCount"] = sum(t in ("TRIG", "STIM") for t in types)
            _write_json(eeg_json, eeg_meta)


def add_ecg_physio_from_eeg(
    *,
    bids_root: Path,
    ecg_map_path: Path,
    overwrite: bool,
    metadata: dict[str, Any],
) -> None:
    try:
        import mne
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mne is required for ECG physio export.") from e

    task = _bids_task_from_root(bids_root)
    mapping = _load_ecg_map(ecg_map_path).get(task, {})
    ecg_defaults = _get_ecg_defaults(metadata)

    for vhdr_path in bids_root.glob(f"sub-*/eeg/*_task-{task}_eeg.vhdr"):
        subject = vhdr_path.parts[-3]
        ecg_channel = mapping.get(subject)
        if ecg_channel not in ("D1", "D3"):
            ecg_channel = _infer_ecg_channel_from_brainvision(bids_root, subject, task)

        subject_dir = vhdr_path.parent.parent
        out_dir = subject_dir / "ecg"
        out_tsv = out_dir / f"{subject}_task-{task}_recording-ecg_physio.tsv"
        out_json = out_dir / f"{subject}_task-{task}_recording-ecg_physio.json"
        if out_tsv.exists() and not overwrite:
            continue

        raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
        if ecg_channel not in raw.ch_names:
            raise ValueError(f"ECG channel {ecg_channel} not found in {vhdr_path}")
        raw.pick([ecg_channel])
        raw.load_data()

        data = raw.get_data()[0]
        times = raw.times
        sampling_frequency = float(raw.info["sfreq"])

        header = ["time", "cardiac"]

        def rows() -> Iterable[list[Any]]:
            for idx, value in enumerate(data):
                yield [
                    _format_value(times[idx], 6),
                    _format_value(float(value), 9),
                ]

        _write_physio_tsv(out_tsv, header, rows(), write_header=False)

        sidecar: dict[str, Any] = {
            "TaskName": task,
            "RecordingType": "continuous",
            "SamplingFrequency": sampling_frequency,
            "StartTime": float(times[0]) if times.size else 0.0,
            "Columns": header,
            "SourceChannels": [ecg_channel],
            "SourceStreamName": "EEGStream",
            "time": {
                "Description": "Time (s) relative to EEG recording start.",
                "Format": "number",
                "Units": "s",
            },
            "cardiac": {
                "Description": "Electrocardiogram derived from the EEG auxiliary channel.",
                "Format": "number",
                "Units": "V",
            },
        }
        if ecg_defaults:
            sidecar.update(ecg_defaults)
        out_json.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess BIDS outputs from XDF (markers, ECG typing, pupil, ECG physio)."
    )
    parser.add_argument("--bids-root", help="Single BIDS dataset root to process.")
    parser.add_argument(
        "--bids-roots",
        nargs="*",
        default=None,
        help="One or more BIDS dataset roots to process.",
    )
    parser.add_argument(
        "--ecg-map",
        default=str(Path("config") / "ecg_channel_map.json"),
        help="Path to JSON file mapping subject->ECG channel for each task.",
    )
    parser.add_argument(
        "--metadata-config",
        default=str(Path("config") / "sidecar_defaults.json"),
        help="Path to JSON file with EEG/ECG sidecar defaults.",
    )
    parser.add_argument(
        "--dataset-description-config",
        default=str(Path("config") / "dataset_description_defaults.json"),
        help="Path to JSON file with dataset_description.json defaults.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing derived outputs.")
    parser.add_argument(
        "--skip-physio",
        "--skip-eyetrack",
        action="store_true",
        help="Skip eyetrack and ECG physio extraction.",
    )
    parser.add_argument("--skip-events", action="store_true", help="Skip events.tsv/.json updates.")
    parser.add_argument("--skip-ecg", action="store_true", help="Skip ECG channel typing fixes.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    ecg_map_path = (repo_root / args.ecg_map).resolve()
    metadata_path = (repo_root / args.metadata_config).resolve()
    dataset_description_path = (repo_root / args.dataset_description_config).resolve()
    if not dataset_description_path.exists():
        fallback = (repo_root / "conversion_package" / "config" / "dataset_description_defaults.json").resolve()
        if fallback.exists():
            dataset_description_path = fallback
    metadata = _load_metadata_config(metadata_path)
    dataset_description_defaults = _load_metadata_config(dataset_description_path)
    bids_roots = _resolve_bids_roots(repo_root, args.bids_root, args.bids_roots)

    for bids_root in bids_roots:
        if not bids_root.exists():
            raise FileNotFoundError(bids_root)

        if not args.skip_events:
            add_marker_stream_column(bids_root=bids_root, overwrite=args.overwrite)
            update_events_json_definitions(bids_root=bids_root)

        if not args.skip_ecg:
            fix_ecg_channel_types(bids_root=bids_root, ecg_map_path=ecg_map_path, overwrite=args.overwrite)

        apply_dataset_description_defaults(bids_root=bids_root, defaults=dataset_description_defaults)
        apply_eeg_sidecar_defaults(bids_root=bids_root, metadata=metadata)
        ensure_readme_references(bids_root=bids_root)

        if not args.skip_physio:
            _ensure_bidsignore(bids_root)
            _remove_legacy_physio_files(bids_root)
            add_ecg_physio_from_eeg(
                bids_root=bids_root,
                ecg_map_path=ecg_map_path,
                overwrite=args.overwrite,
                metadata=metadata,
            )
            add_eyetrack_from_xdf(bids_root=bids_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
