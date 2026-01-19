from __future__ import annotations

import argparse
from pathlib import Path

from marker_extraction import extract_nback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract n-back marker streams from XDF into CSVs and events TSVs."
    )
    parser.add_argument(
        "--xdf-dir",
        default=str(Path("nback_Data") / "sourcedata"),
        help="Directory containing n-back XDF files.",
    )
    parser.add_argument(
        "--markers-dir",
        default="markers_nback",
        help="Output directory for marker CSVs and count summaries.",
    )
    parser.add_argument(
        "--bids-stub-root",
        default="bids_stub",
        help="BIDS stub root for task-only events (onset relative to first marker).",
    )
    parser.add_argument(
        "--skip-bids-stub",
        action="store_true",
        help="Do not write stub events.tsv files.",
    )
    parser.add_argument(
        "--bids-root",
        default=None,
        help="Optional BIDS root for EEG-aligned events.tsv (includes dropouts).",
    )
    parser.add_argument(
        "--skip-dropouts",
        action="store_true",
        help="Skip UoHDataOffsetStream dropouts when writing BIDS events.tsv.",
    )
    parser.add_argument(
        "--dropout-srate",
        type=float,
        default=None,
        help="Override sampling rate (Hz) for dropout duration calculation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    bids_stub_root = None if args.skip_bids_stub else Path(args.bids_stub_root)
    bids_root = Path(args.bids_root) if args.bids_root else None

    extract_nback(
        xdf_dir=Path(args.xdf_dir),
        markers_dir=Path(args.markers_dir),
        bids_stub_root=bids_stub_root,
        bids_root=bids_root,
        overwrite=args.overwrite,
        include_dropouts=not args.skip_dropouts,
        dropout_srate=args.dropout_srate,
    )


if __name__ == "__main__":
    main()
