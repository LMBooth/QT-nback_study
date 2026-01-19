from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any


def _scrub_sys_path() -> None:
    py_path = os.environ.get("PYTHONPATH")
    if not py_path:
        return

    import site
    protected = {os.path.abspath(p) for p in site.getsitepackages()}

    entries = {os.path.abspath(p) for p in py_path.split(os.pathsep) if p}
    to_remove = entries - protected  # don't remove actual site-packages

    if to_remove:
        sys.path = [p for p in sys.path if os.path.abspath(p) not in to_remove]



_scrub_sys_path()

import numpy as np
import pyxdf


def _load_mapping(map_path: Path) -> list[tuple[str, str]]:
    with map_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {map_path}")
    if "raw_xdf" in rows[0] and "bids_xdf" in rows[0]:
        return [(r["raw_xdf"], r["bids_xdf"]) for r in rows]
    raise ValueError(
        f"{map_path} must contain 'raw_xdf' and 'bids_xdf' columns (see reports/participant_id_map_*.tsv)."
    )


def _subject_from_xdf_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_", 1)
    if not parts or len(parts[0]) != 3 or not parts[0].isdigit():
        raise ValueError(f"Unexpected XDF filename (expected NNN_*.xdf): {name}")
    return parts[0]


def _ensure_suffix(name: str) -> str:
    return name if name.lower().endswith(".xdf") else f"{name}.xdf"


def _parse_channel_labels(stream: dict[str, Any]) -> list[str]:
    desc = stream.get("info", {}).get("desc", [None])[0]
    if not desc:
        return [f"ch{idx + 1:02d}" for idx in range(stream["time_series"].shape[1])]
    try:
        channels = desc["channels"][0]["channel"]
        labels = [c["label"][0] for c in channels]
        if len(labels) != stream["time_series"].shape[1]:
            raise ValueError("Label count does not match data channel count.")
        return labels
    except Exception:
        return [f"ch{idx + 1:02d}" for idx in range(stream["time_series"].shape[1])]


def _estimate_sampling_frequency(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return float("nan")
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    median_dt = float(np.median(diffs))
    return 1.0 / median_dt if median_dt > 0 else float("nan")


def _load_eeg_stream(xdf_path: Path, stream_name: str) -> dict[str, Any]:
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path), select_streams=[{"name": stream_name}])
    except TypeError:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    for stream in streams:
        if stream["info"]["name"][0] == stream_name:
            return stream
    raise ValueError(f"{stream_name} not found in {xdf_path}")


def _build_raw(stream: dict[str, Any], line_freq: float) -> "mne.io.RawArray":
    import mne

    labels = _parse_channel_labels(stream)
    data = np.asarray(stream["time_series"], dtype=np.float64).T
    timestamps = np.asarray(stream["time_stamps"], dtype=np.float64)
    sfreq = float(stream["info"].get("nominal_srate", [0.0])[0])
    if not math.isfinite(sfreq) or sfreq <= 0:
        sfreq = _estimate_sampling_frequency(timestamps)
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid sampling frequency for {stream['info']['name'][0]}")

    ch_types = ["misc" if name.startswith("D") else "eeg" for name in labels]
    info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=ch_types)
    info["line_freq"] = line_freq
    return mne.io.RawArray(data, info)


def _write_bids(raw: "mne.io.Raw", *, bids_root: Path, subject: str, task: str, overwrite: bool) -> None:
    try:
        import mne_bids
    except Exception as exc:
        raise RuntimeError("mne-bids is required for BIDS export.") from exc

    bids_path = mne_bids.BIDSPath(subject=subject, task=task, root=bids_root, suffix="eeg")
    mne_bids.write_raw_bids(
        raw,
        bids_path,
        overwrite=overwrite,
        format="BrainVision",
        allow_preload=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EEGStream from XDF to BIDS (BrainVision).")
    parser.add_argument("--xdf-dir", required=True, help="Directory containing raw XDF files.")
    parser.add_argument("--bids-root", required=True, help="BIDS dataset root (e.g., bids_nback).")
    parser.add_argument("--task", required=True, help="BIDS task label (e.g., nback).")
    parser.add_argument("--map", dest="map_path", help="Mapping TSV with raw_xdf and bids_xdf.")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Use the XDF filename prefix (NNN_*.xdf) as the BIDS subject ID.",
    )
    parser.add_argument("--stream-name", default="EEGStream", help="XDF stream name to extract.")
    parser.add_argument("--line-freq", type=float, default=50.0, help="Line frequency (Hz).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS outputs.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    xdf_dir = (repo_root / args.xdf_dir).resolve()
    bids_root = (repo_root / args.bids_root).resolve()
    if args.identity and args.map_path:
        raise ValueError("Specify either --identity or --map, not both.")
    if not args.identity and not args.map_path:
        raise ValueError("Either --identity or --map is required.")
    map_path = (repo_root / args.map_path).resolve() if args.map_path else None

    if args.identity:
        mapping = [(p.name, p.name) for p in sorted(xdf_dir.glob("*.xdf"))]
    else:
        mapping = _load_mapping(map_path)

    for raw_name, bids_name in mapping:
        raw_path = xdf_dir / _ensure_suffix(raw_name)
        bids_stem = _subject_from_xdf_name(bids_name)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        stream = _load_eeg_stream(raw_path, args.stream_name)
        raw = _build_raw(stream, line_freq=args.line_freq)
        _write_bids(raw, bids_root=bids_root, subject=bids_stem, task=args.task, overwrite=args.overwrite)
        print(f"Wrote BIDS EEG for sub-{bids_stem} from {raw_path.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
