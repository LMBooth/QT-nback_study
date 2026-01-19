from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def _load_mapping(map_path: Path) -> list[tuple[str, str]]:
    with map_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {map_path}")

    if "raw_xdf" in rows[0] and "bids_xdf" in rows[0]:
        mapping = [(r["raw_xdf"], r["bids_xdf"]) for r in rows]
    else:
        raise ValueError(
            f"{map_path} must contain 'raw_xdf' and 'bids_xdf' columns (see reports/participant_id_map_*.tsv)."
        )
    return mapping


def _subject_from_xdf_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_", 1)
    if not parts or len(parts[0]) != 3 or not parts[0].isdigit():
        raise ValueError(f"Unexpected XDF filename (expected NNN_*.xdf): {name}")
    return f"sub-{parts[0]}"


def _write_report(path: Path, mapping: list[tuple[str, str]]) -> None:
    rows: list[dict[str, str]] = []
    for raw_name, bids_name in mapping:
        subject = _subject_from_xdf_name(bids_name)
        original_id = subject.replace("sub-", "")
        rows.append(
            {
                "bids_subject": subject,
                "bids_xdf": _ensure_suffix(bids_name),
                "raw_xdf": _ensure_suffix(raw_name),
                "original_participant": original_id,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bids_subject", "bids_xdf", "raw_xdf", "original_participant"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _ensure_suffix(name: str) -> str:
    return name if name.lower().endswith(".xdf") else f"{name}.xdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy/rename raw XDF files into BIDS sourcedata/xdf.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw XDF files.")
    parser.add_argument("--bids-root", required=True, help="BIDS dataset root (e.g., bids_nback).")
    parser.add_argument(
        "--map",
        dest="map_path",
        help="Mapping TSV with raw_xdf and bids_xdf columns.",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Copy XDF files using the same filename (no remapping).",
    )
    parser.add_argument(
        "--report",
        dest="report_path",
        help="Optional path to write a mapping report TSV.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    raw_dir = (repo_root / args.raw_dir).resolve()
    bids_root = (repo_root / args.bids_root).resolve()
    if args.identity and args.map_path:
        raise ValueError("Specify either --identity or --map, not both.")
    if not args.identity and not args.map_path:
        raise ValueError("Either --identity or --map is required.")
    map_path = (repo_root / args.map_path).resolve() if args.map_path else None

    sourcedata = bids_root / "sourcedata" / "xdf"
    sourcedata.mkdir(parents=True, exist_ok=True)

    if args.identity:
        mapping = [(p.name, p.name) for p in sorted(raw_dir.glob("*.xdf"))]
    else:
        mapping = _load_mapping(map_path)
    copied = 0
    for raw_name, bids_name in mapping:
        src = raw_dir / _ensure_suffix(raw_name)
        dst = sourcedata / _ensure_suffix(bids_name)
        if dst.exists() and not args.overwrite:
            continue
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        copied += 1
        print(f"Copied {src.name} -> {dst.name}")
    print(f"Total copied: {copied}")

    if args.report_path:
        report_path = (repo_root / args.report_path).resolve()
        _write_report(report_path, mapping)
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
