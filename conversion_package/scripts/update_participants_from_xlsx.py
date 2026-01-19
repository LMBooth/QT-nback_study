from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _load_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            xml = zf.read("xl/sharedStrings.xml")
            root = ET.fromstring(xml)
            ns = {"t": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("t:si", ns):
                texts = [t.text or "" for t in si.findall(".//t:t", ns)]
                shared.append("".join(texts))

        sheet = zf.read("xl/worksheets/sheet1.xml")
        root = ET.fromstring(sheet)
        ns = {"t": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        rows: list[list[str]] = []
        max_col = 0
        for row in root.findall(".//t:sheetData/t:row", ns):
            row_cells: dict[int, str] = {}
            for cell in row.findall("t:c", ns):
                ref = cell.get("r")
                if not ref:
                    continue
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = 0
                for ch in col_letters:
                    col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                cell_type = cell.get("t")
                v = cell.find("t:v", ns)
                val = v.text if v is not None else ""
                if cell_type == "s":
                    try:
                        val = shared[int(val)]
                    except Exception:
                        pass
                row_cells[col_idx] = val
                max_col = max(max_col, col_idx)
            row_list = [""] * max_col
            for col_idx, val in row_cells.items():
                row_list[col_idx - 1] = val
            rows.append(row_list)
    return rows


def _parse_participant_info(path: Path) -> dict[int, dict[str, str]]:
    rows = _load_xlsx_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    header = [h.strip().lower() for h in rows[0]]
    col_idx: dict[str, int] = {}
    for idx, name in enumerate(header):
        if name in ("participant no.", "participant_no", "participant", "participant no"):
            col_idx["participant"] = idx
        elif name == "age":
            col_idx["age"] = idx
        elif name == "sex":
            col_idx["sex"] = idx
        elif name in ("handidness", "handedness"):
            col_idx["handedness"] = idx

    missing = {"participant", "age", "sex", "handedness"} - set(col_idx)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    info: dict[int, dict[str, str]] = {}
    for row in rows[1:]:
        if not row or not row[col_idx["participant"]].strip():
            continue
        try:
            participant = int(float(row[col_idx["participant"]]))
        except ValueError:
            continue

        age_raw = row[col_idx["age"]].strip()
        sex_raw = row[col_idx["sex"]].strip().upper()
        hand_raw = row[col_idx["handedness"]].strip().upper()

        age = "n/a"
        if age_raw:
            try:
                age = f"{float(age_raw):.0f}"
            except ValueError:
                age = age_raw

        sex = sex_raw or "n/a"

        handedness = hand_raw or "n/a"

        info[participant] = {
            "age": age,
            "sex": sex,
            "handedness": handedness,
        }

    return info


def _build_size_map(folder: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in folder.glob("*.xdf"):
        out[path.stem] = path.stat().st_size
    return out


def _map_bids_to_raw(bids_xdf: Path, raw_xdf: Path) -> dict[str, str]:
    raw_sizes = _build_size_map(raw_xdf)
    bids_sizes = _build_size_map(bids_xdf)

    size_to_raw: dict[int, list[str]] = {}
    for stem, size in raw_sizes.items():
        size_to_raw.setdefault(size, []).append(stem)

    mapping: dict[str, str] = {}
    for bids_stem, size in bids_sizes.items():
        candidates = size_to_raw.get(size, [])
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one raw match for {bids_stem} (size={size}); found {candidates}"
            )
        mapping[bids_stem] = candidates[0]
    return mapping


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


def _list_raw_ids(xdf_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in xdf_dir.glob("*.xdf"):
        stem = path.stem
        parts = stem.split("_", 1)
        if not parts or len(parts[0]) != 3 or not parts[0].isdigit():
            raise ValueError(f"Unexpected XDF filename (expected NNN_*.xdf): {path.name}")
        ids.append(int(parts[0]))
    return sorted(ids)


def _subject_from_id(participant_id: int) -> str:
    return f"sub-{participant_id:03d}"


def _id_from_subject(subject: str) -> int:
    if not subject.startswith("sub-"):
        raise ValueError(f"Unexpected participant_id format: {subject}")
    return int(subject.replace("sub-", ""))


def _read_participants_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _write_participants_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _update_participants_json(path: Path) -> None:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}

    data.setdefault("participant_id", {"Description": "Unique participant identifier."})
    data["age"] = {"Description": "Age (years)."}
    data["sex"] = {
        "Description": "Biological sex.",
        "Levels": {"M": "Male", "F": "Female", "O": "Other", "n/a": "Not available"},
    }
    data["handedness"] = {
        "Description": "Handedness.",
        "Levels": {"R": "Right", "L": "Left", "A": "Ambidextrous", "n/a": "Not available"},
    }
    data.pop("height", None)

    data["analysis_included"] = {
        "Description": "Whether this participant was included in the authors' ML analysis.",
        "Levels": {"true": "Included.", "false": "Excluded."},
    }
    data["data_quality_overall"] = {
        "Description": "Overall data-quality assessment for the released dataset.",
        "Levels": {"pass": "Passed release QC.", "fail": "Excluded from release QC."},
    }

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _summarize(rows: list[dict[str, str]]) -> dict[str, str]:
    ages = []
    sex_counts: dict[str, int] = {}
    for row in rows:
        analysis_included = str(row.get("analysis_included", "true")).lower() == "true"
        if not analysis_included:
            continue
        age_raw = row.get("age", "")
        try:
            ages.append(float(age_raw))
        except ValueError:
            pass
        sex = row.get("sex", "n/a")
        sex_counts[sex] = sex_counts.get(sex, 0) + 1
    mean_age = sum(ages) / len(ages) if ages else 0.0
    return {
        "mean_age": f"{mean_age:.2f}" if ages else "n/a",
        "sex_counts": ", ".join(f"{k}:{v}" for k, v in sorted(sex_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update BIDS participants.tsv/json from participant_info.xlsx.")
    parser.add_argument("--xlsx", default="participant_info.xlsx", help="Path to participant info XLSX.")
    parser.add_argument("--bids-root", help="Single BIDS dataset root to update.")
    parser.add_argument(
        "--bids-roots",
        nargs="*",
        default=None,
        help="BIDS dataset roots to update.",
    )
    parser.add_argument(
        "--raw-nback",
        default="nback_Data/sourcedata",
        help="Raw n-back XDF directory (for ID mapping).",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to write participant ID mapping reports.",
    )
    parser.add_argument(
        "--id-mode",
        choices=["mapping", "identity"],
        default="mapping",
        help="Use XDF size-based mapping or identity subject IDs (NNN_*.xdf -> sub-NNN).",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include participants without XDF data and mark analysis_included=false.",
    )
    parser.add_argument(
        "--skip-copy-xlsx",
        action="store_true",
        help="Skip copying the XLSX file into each BIDS sourcedata folder.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    xlsx_path = (repo_root / args.xlsx).resolve()
    info = _parse_participant_info(xlsx_path)

    raw_dirs = {
        "bids_nback": (repo_root / args.raw_nback).resolve(),
    }
    report_dir = (repo_root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.include_excluded and args.id_mode != "identity":
        raise ValueError("--include-excluded requires --id-mode identity")

    bids_roots = _resolve_bids_roots(repo_root, args.bids_root, args.bids_roots)
    for bids_root in bids_roots:
        if not bids_root.exists():
            raise FileNotFoundError(bids_root)

        raw_dir = raw_dirs.get(bids_root.name)
        if raw_dir is None or not raw_dir.exists():
            raise FileNotFoundError(f"Raw XDF directory not found for {bids_root}: {raw_dir}")

        task = _task_from_bids_root(bids_root)
        bids_xdf = bids_root / "sourcedata" / "xdf"

        mapping: dict[str, int] = {}
        report_rows: list[dict[str, str]] = []

        if args.id_mode == "identity":
            raw_ids = _list_raw_ids(raw_dir)
            for participant_id in raw_ids:
                subject = _subject_from_id(participant_id)
                mapping[subject] = participant_id
                report_rows.append(
                    {
                        "bids_subject": subject,
                        "bids_xdf": f"{participant_id:03d}_{task}.xdf",
                        "raw_xdf": f"{participant_id:03d}_{task}.xdf",
                        "original_participant": f"{participant_id:03d}",
                    }
                )
        else:
            bids_to_raw = _map_bids_to_raw(bids_xdf, raw_dir)
            for bids_stem, raw_stem in bids_to_raw.items():
                subject = _subject_from_id(int(bids_stem.split("_", 1)[0]))
                original_id = int(raw_stem.split("_", 1)[0])
                mapping[subject] = original_id
                report_rows.append(
                    {
                        "bids_subject": subject,
                        "bids_xdf": bids_stem + ".xdf",
                        "raw_xdf": raw_stem + ".xdf",
                        "original_participant": f"{original_id:03d}",
                    }
                )

        if args.include_excluded:
            subjects = [_subject_from_id(pid) for pid in sorted(info)]
        else:
            subjects = sorted(mapping, key=_id_from_subject)

        updated_rows: list[dict[str, Any]] = []
        for subject in subjects:
            if subject in mapping:
                original_id = mapping[subject]
                analysis_included = "true"
                data_quality = "pass"
            else:
                original_id = _id_from_subject(subject)
                analysis_included = "false"
                data_quality = "fail"

            demographics = info.get(original_id)
            if demographics is None:
                raise ValueError(f"Missing demographics for participant {original_id:03d}")

            updated_rows.append(
                {
                    "participant_id": subject,
                    **demographics,
                    "analysis_included": analysis_included,
                    "data_quality_overall": data_quality,
                }
            )

        fieldnames = [
            "participant_id",
            "age",
            "sex",
            "handedness",
            "analysis_included",
            "data_quality_overall",
        ]
        participants_tsv = bids_root / "participants.tsv"
        _write_participants_tsv(participants_tsv, fieldnames, updated_rows)

        participants_json = bids_root / "participants.json"
        _update_participants_json(participants_json)

        if not args.skip_copy_xlsx:
            sourcedata_dir = bids_root / "sourcedata"
            sourcedata_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(xlsx_path, sourcedata_dir / xlsx_path.name)

        report_path = report_dir / f"participant_id_map_{bids_root.name}.tsv"
        if report_rows:
            _write_participants_tsv(
                report_path,
                ["bids_subject", "bids_xdf", "raw_xdf", "original_participant"],
                report_rows,
            )

        summary = _summarize(updated_rows)
        print(f"{bids_root.name}: mean_age={summary['mean_age']}, sex_counts={summary['sex_counts']}")


if __name__ == "__main__":  # pragma: no cover
    main()
