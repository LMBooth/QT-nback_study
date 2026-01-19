from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _task_from_bids_root(bids_root: Path) -> str:
    name = bids_root.name.lower()
    if "nback" in name:
        return "nback"
    raise ValueError(f"Could not infer task from BIDS root: {bids_root}")


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


def _find_ecg_channel(channels_path: Path) -> str | None:
    with channels_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("type") == "ECG" and row.get("name") in ("D1", "D3"):
                return row["name"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Update ECG channel map from BIDS channels.tsv files.")
    parser.add_argument("--bids-root", help="Single BIDS dataset root to process.")
    parser.add_argument(
        "--bids-roots",
        nargs="*",
        default=None,
        help="BIDS dataset roots to process.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("config") / "ecg_channel_map.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    out_path = (repo_root / args.out).resolve()
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass --overwrite or choose a different --out path."
        )

    note = (
        "Per-subject ECG channel location. Only one of D1 or D3 contains ECG; "
        "remaining D* channels are auxiliary/unused and should be typed MISC in *_channels.tsv."
    )
    data: dict[str, dict[str, str] | str] = {"note": note}

    bids_roots = _resolve_bids_roots(repo_root, args.bids_root, args.bids_roots)
    for bids_root in bids_roots:
        task = _task_from_bids_root(bids_root)
        mapping: dict[str, str] = {}

        for channels_path in sorted(bids_root.glob("sub-*/eeg/*_channels.tsv")):
            subject = channels_path.parts[-3]
            ecg_channel = _find_ecg_channel(channels_path)
            if ecg_channel:
                mapping[subject] = ecg_channel

        data[task] = dict(sorted(mapping.items()))

    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
