from __future__ import annotations

import csv
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

NBACK_TRIAL_DURATION = 0.0

EVENT_FIELDS_BASE = [
    "onset",
    "duration",
    "trial_type",
    "timestamp_lsl",
    "marker",
    "response_accuracy",
    "nback_level",
    "key_press",
    "matched",
    "outcome",
    "istutorial",
]
EVENT_FIELDS_ALL = EVENT_FIELDS_BASE + ["dropped_samples", "marker_stream"]

_NBACK_RE = re.compile(
    r"^Steps:(?P<steps>\d+)\s+KeyPress:(?P<keypress>True|False)\s+Matched:(?P<matched>True|False)$",
    re.IGNORECASE,
)
_DROPOUT_RE = re.compile(r"^Dropped\s+(?P<count>\d+)\s+samples$", re.IGNORECASE)


@dataclass(frozen=True)
class XdfStreams:
    eeg: dict[str, Any]
    marker: dict[str, Any]
    dropout: dict[str, Any] | None


@dataclass(frozen=True)
class MarkerEntry:
    timestamp: float
    marker: str


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


def load_xdf_streams(xdf_path: Path, marker_name: str) -> XdfStreams:
    _scrub_sys_path()
    import pyxdf

    try:
        streams, _ = pyxdf.load_xdf(
            str(xdf_path),
            select_streams=[
                {"name": "EEGStream"},
                {"name": marker_name},
                {"name": "UoHDataOffsetStream"},
            ],
        )
    except TypeError:
        streams, _ = pyxdf.load_xdf(str(xdf_path))

    eeg = None
    marker = None
    dropout = None
    for stream in streams:
        name = stream["info"]["name"][0]
        if name == "EEGStream":
            eeg = stream
        elif name == marker_name:
            marker = stream
        elif name == "UoHDataOffsetStream":
            dropout = stream

    if eeg is None:
        raise ValueError(f"EEGStream not found in {xdf_path}")
    if marker is None:
        raise ValueError(f"{marker_name} not found in {xdf_path}")
    return XdfStreams(eeg=eeg, marker=marker, dropout=dropout)


def iter_marker_entries(stream: dict[str, Any]) -> list[MarkerEntry]:
    markers: list[MarkerEntry] = []
    series = stream["time_series"]
    timestamps = stream["time_stamps"]
    for idx in range(len(timestamps)):
        raw = series[idx]
        if isinstance(raw, (list, tuple)) and raw:
            marker = str(raw[0])
        else:
            marker = str(raw)
        markers.append(MarkerEntry(timestamp=float(timestamps[idx]), marker=marker))
    return markers


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def subject_from_xdf_filename(xdf_path: Path) -> str:
    match = re.match(r"^(?P<id>\d{3})_", xdf_path.stem)
    if not match:
        raise ValueError(f"Unexpected XDF filename (expected 'NNN_*.xdf'): {xdf_path.name}")
    return f"sub-{match.group('id')}"


def _sanitize_trial_type(marker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", marker.strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "marker"


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _format_timestamp(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _event_template() -> dict[str, Any]:
    return {field: None for field in EVENT_FIELDS_ALL}


def format_event(event: dict[str, Any], missing_value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for field in EVENT_FIELDS_ALL:
        value = event.get(field)
        if value is None:
            out[field] = missing_value
        elif field in ("onset", "duration", "timestamp_lsl"):
            out[field] = _format_seconds(float(value))
        else:
            out[field] = str(value)
    return out


def _get_stream_id(stream: dict[str, Any]) -> str:
    info = stream.get("info", {})
    raw = info.get("stream_id", "")
    if isinstance(raw, (list, tuple)):
        return str(raw[0]) if raw else ""
    return str(raw)


def _get_nominal_srate(eeg_stream: dict[str, Any], fallback: float) -> float:
    raw = eeg_stream["info"].get("nominal_srate", [fallback])[0]
    try:
        srate = float(raw)
    except (TypeError, ValueError):
        srate = fallback
    if not math.isfinite(srate) or srate <= 0:
        srate = fallback
    return srate


def build_dropout_events(
    entries: list[MarkerEntry],
    *,
    base_time: float,
    srate: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry in entries:
        match = _DROPOUT_RE.match(entry.marker.strip())
        if not match:
            continue
        dropped = int(match.group("count"))
        duration = dropped / srate if srate else 0.0
        event = _event_template()
        event["onset"] = entry.timestamp - base_time
        event["duration"] = duration
        event["trial_type"] = "dropped_samples"
        event["timestamp_lsl"] = entry.timestamp
        event["marker"] = entry.marker
        event["dropped_samples"] = dropped
        event["marker_stream"] = "UoHDataOffsetStream"
        events.append(event)
    return events


def _parse_nback_marker(marker: str) -> tuple[str, bool, bool] | None:
    match = _NBACK_RE.match(marker)
    if not match:
        return None
    steps = match.group("steps")
    key_press = match.group("keypress").lower() == "true"
    matched = match.group("matched").lower() == "true"
    return steps, key_press, matched


def build_nback_events(
    entries: list[MarkerEntry],
    *,
    base_time: float,
    marker_stream: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    tutorial_state: bool | None = None
    for entry in entries:
        marker = entry.marker.strip()
        event = _event_template()
        event["onset"] = entry.timestamp - base_time
        event["timestamp_lsl"] = entry.timestamp
        event["marker"] = marker
        event["marker_stream"] = marker_stream

        marker_lower = marker.lower()
        marker_tutorial: bool | None
        if marker_lower.startswith("started tutorial"):
            tutorial_state = True
            marker_tutorial = True
        elif marker_lower.startswith("finished tutorial"):
            marker_tutorial = True
            tutorial_state = None
        elif marker_lower.startswith("started n-back") or marker_lower.startswith("started nback"):
            tutorial_state = False
            marker_tutorial = False
        elif marker_lower.startswith("finished n-back") or marker_lower.startswith("finished nback"):
            marker_tutorial = False
            tutorial_state = None
        else:
            marker_tutorial = tutorial_state

        if marker_tutorial is not None:
            event["istutorial"] = "true" if marker_tutorial else "false"

        parsed = _parse_nback_marker(marker)
        if parsed:
            steps, key_press, matched = parsed
            accuracy = key_press == matched
            event["trial_type"] = f"{steps}-back"
            event["duration"] = NBACK_TRIAL_DURATION
            event["nback_level"] = steps
            event["key_press"] = "true" if key_press else "false"
            event["matched"] = "true" if matched else "false"
            event["response_accuracy"] = "1" if accuracy else "0"
            event["outcome"] = "Correct" if accuracy else "Wrong"
        else:
            event["trial_type"] = _sanitize_trial_type(marker)
            event["duration"] = 0.0
        events.append(event)
    return events


def count_nback_markers(
    entries: Iterable[MarkerEntry], *, file_label: str
) -> list[dict[str, Any]]:
    counts: dict[int, dict[str, int]] = {}
    for entry in entries:
        parsed = _parse_nback_marker(entry.marker.strip())
        if not parsed:
            continue
        steps, key_press, matched = parsed
        steps_val = int(steps)
        if steps_val not in counts:
            counts[steps_val] = {"correct": 0, "wrong": 0}
        if key_press == matched:
            counts[steps_val]["correct"] += 1
        else:
            counts[steps_val]["wrong"] += 1

    rows: list[dict[str, Any]] = []
    for steps_val in sorted(counts):
        correct = counts[steps_val]["correct"]
        wrong = counts[steps_val]["wrong"]
        total = correct + wrong
        accuracy = correct / total if total else 0.0
        rows.append(
            {
                "file": file_label,
                "steps": steps_val,
                "correct": correct,
                "wrong": wrong,
                "total": total,
                "accuracy": f"{accuracy:.6f}",
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tsv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, median, std


def write_nback_summary_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["steps"]), []).append(row)

    out_rows: list[dict[str, Any]] = []
    for steps_val in sorted(grouped):
        group = grouped[steps_val]
        correct_vals = [float(r["correct"]) for r in group]
        wrong_vals = [float(r["wrong"]) for r in group]
        total_vals = [float(r["total"]) for r in group]
        accuracy_vals = [float(r["accuracy"]) for r in group]

        correct_mean, correct_median, correct_std = _summary_stats(correct_vals)
        wrong_mean, wrong_median, wrong_std = _summary_stats(wrong_vals)
        total_mean, total_median, total_std = _summary_stats(total_vals)
        accuracy_mean, accuracy_median, accuracy_std = _summary_stats(accuracy_vals)

        out_rows.append(
            {
                "steps": steps_val,
                "n_participants": len(group),
                "correct_mean": f"{correct_mean:.6f}",
                "correct_median": f"{correct_median:.6f}",
                "correct_std": f"{correct_std:.6f}",
                "wrong_mean": f"{wrong_mean:.6f}",
                "wrong_median": f"{wrong_median:.6f}",
                "wrong_std": f"{wrong_std:.6f}",
                "total_mean": f"{total_mean:.6f}",
                "total_median": f"{total_median:.6f}",
                "total_std": f"{total_std:.6f}",
                "accuracy_mean": f"{accuracy_mean:.6f}",
                "accuracy_median": f"{accuracy_median:.6f}",
                "accuracy_std": f"{accuracy_std:.6f}",
            }
        )

    header = [
        "steps",
        "n_participants",
        "correct_mean",
        "correct_median",
        "correct_std",
        "wrong_mean",
        "wrong_median",
        "wrong_std",
        "total_mean",
        "total_median",
        "total_std",
        "accuracy_mean",
        "accuracy_median",
        "accuracy_std",
    ]
    write_csv(path, header, out_rows)


def _write_marker_csv(
    *, path: Path, file_label: str, marker_stream: dict[str, Any], entries: list[MarkerEntry]
) -> None:
    rows: list[dict[str, Any]] = []
    stream_id = _get_stream_id(marker_stream)
    stream_name = marker_stream["info"]["name"][0]
    for entry in entries:
        rows.append(
            {
                "file": file_label,
                "stream_id": stream_id,
                "stream_name": stream_name,
                "timestamp": _format_timestamp(entry.timestamp, 7),
                "marker": entry.marker,
            }
        )
    write_csv(path, ["file", "stream_id", "stream_name", "timestamp", "marker"], rows)


def extract_nback(
    *,
    xdf_dir: Path,
    markers_dir: Path,
    bids_stub_root: Path | None,
    bids_root: Path | None,
    overwrite: bool,
    include_dropouts: bool,
    dropout_srate: float | None,
) -> None:
    xdf_paths = sorted(xdf_dir.glob("*.xdf"))
    if not xdf_paths:
        raise FileNotFoundError(f"No XDF files found in {xdf_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    all_counts: list[dict[str, Any]] = []
    for xdf_path in xdf_paths:
        file_label = relative_path(xdf_path, repo_root)
        participant = xdf_path.stem
        subject = subject_from_xdf_filename(xdf_path)

        streams = load_xdf_streams(xdf_path, marker_name="n-backMarkers")
        entries = iter_marker_entries(streams.marker)

        markers_path = markers_dir / f"{participant}_markers.csv"
        if overwrite or not markers_path.exists():
            _write_marker_csv(
                path=markers_path, file_label=file_label, marker_stream=streams.marker, entries=entries
            )

        counts_rows = count_nback_markers(entries, file_label=file_label)
        counts_path = markers_dir / f"{participant}_counts_by_steps.csv"
        if overwrite or not counts_path.exists():
            write_csv(
                counts_path,
                ["file", "steps", "correct", "wrong", "total", "accuracy"],
                counts_rows,
            )
        all_counts.extend(counts_rows)

        if bids_stub_root is not None:
            if not entries:
                raise ValueError(f"No n-back markers in {xdf_path}")
            base_time = min(e.timestamp for e in entries)
            events = build_nback_events(entries, base_time=base_time, marker_stream="n-backMarkers")
            events.sort(key=lambda e: float(e["onset"]))
            formatted = [format_event(e, missing_value="") for e in events]
            out_path = bids_stub_root / subject / "eeg" / f"{subject}_task-nback_events.tsv"
            if overwrite or not out_path.exists():
                write_tsv(out_path, EVENT_FIELDS_BASE, formatted)

        if bids_root is not None:
            eeg_start = float(streams.eeg["time_stamps"][0])
            events = build_nback_events(entries, base_time=eeg_start, marker_stream="n-backMarkers")
            if include_dropouts and streams.dropout is not None:
                dropout_entries = iter_marker_entries(streams.dropout)
                srate = dropout_srate or _get_nominal_srate(streams.eeg, 250.0)
                events.extend(build_dropout_events(dropout_entries, base_time=eeg_start, srate=srate))
            events.sort(key=lambda e: float(e["onset"]))
            formatted = [format_event(e, missing_value="n/a") for e in events]
            out_path = bids_root / subject / "eeg" / f"{subject}_task-nback_events.tsv"
            if overwrite or not out_path.exists():
                write_tsv(out_path, EVENT_FIELDS_ALL, formatted)

    counts_all_path = markers_dir / "counts_by_steps_all_participants.csv"
    if overwrite or not counts_all_path.exists():
        write_csv(
            counts_all_path,
            ["file", "steps", "correct", "wrong", "total", "accuracy"],
            all_counts,
        )

    summary_path = markers_dir / "counts_by_steps_summary_stats.csv"
    if overwrite or not summary_path.exists():
        write_nback_summary_stats(summary_path, all_counts)
