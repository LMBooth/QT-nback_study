from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

def _expected_columns(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    columns = data.get("Columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Missing Columns in {json_path}")
    return [str(c) for c in columns]


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


def _rewrite_without_header(tsv_path: Path, columns: list[str]) -> bool:
    header = "\t".join(columns)
    if tsv_path.suffix == ".gz":
        with gzip.open(tsv_path, "rt", encoding="utf-8", newline="") as f:
            first_line = f.readline().rstrip("\r\n")
            compare_line = first_line.lstrip("\ufeff")
            if compare_line != header:
                return False
            remainder = f.read()

        tmp_path = tsv_path.with_suffix(tsv_path.suffix + ".tmp")
        with tmp_path.open("wb") as raw_f:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, mtime=0) as gz_f:
                with io.TextIOWrapper(gz_f, encoding="utf-8", newline="") as text_f:
                    if remainder:
                        text_f.write(remainder)
        tmp_path.replace(tsv_path)
        return True

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        first_line = f.readline().rstrip("\r\n")
        compare_line = first_line.lstrip("\ufeff")
        if compare_line != header:
            return False
        remainder = f.read()

    tmp_path = tsv_path.with_suffix(tsv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as text_f:
        if remainder:
            text_f.write(remainder)
    tmp_path.replace(tsv_path)
    return True


def _sidecar_path(tsv_path: Path) -> Path:
    name = tsv_path.name
    if name.endswith(".tsv.gz"):
        base = name[:-7]
    elif name.endswith(".tsv"):
        base = name[:-4]
    else:
        base = tsv_path.stem

    if base.endswith("_pupil"):
        base = base[:-len("_pupil")] + "_eyetrack"

    return tsv_path.with_name(base + ".json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure physio TSV files do not include header rows."
    )
    parser.add_argument("--bids-root", help="Single BIDS dataset root to process.")
    parser.add_argument(
        "--bids-roots",
        nargs="*",
        default=None,
        help="BIDS dataset roots to process.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    updated = 0
    bids_roots = _resolve_bids_roots(repo_root, args.bids_root, args.bids_roots)
    patterns = [
        "sub-*/ecg/*_recording-*_physio.tsv",
        "sub-*/ecg/*_recording-*_physio.tsv.gz",
        "sub-*/eeg/*_recording-*_physio.tsv",
        "sub-*/eeg/*_recording-*_physio.tsv.gz",
        "sub-*/eyetrack/*_recording-eyetrack_physio.tsv",
        "sub-*/eyetrack/*_recording-eyetrack_physio.tsv.gz",
        "sub-*/pupil/*_pupil.tsv",
        "sub-*/pupil/*_pupil.tsv.gz",
    ]
    for bids_root in bids_roots:
        for pattern in patterns:
            for tsv_path in bids_root.glob(pattern):
                json_path = _sidecar_path(tsv_path)
                columns = _expected_columns(json_path)
                if _rewrite_without_header(tsv_path, columns):
                    updated += 1
                    print(f"Removed header: {tsv_path}")
    print(f"Total updated: {updated}")


if __name__ == "__main__":  # pragma: no cover
    main()
