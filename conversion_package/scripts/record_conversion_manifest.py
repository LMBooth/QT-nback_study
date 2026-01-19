from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore


REQUIRED_PACKAGES = ("numpy", "pyxdf", "mne", "mne-bids", "pybv")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_record(path: Path, repo_root: Path, *, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": _relative_path(path, repo_root),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if include_hash:
        record["sha256"] = _sha256(path)
    return record


def _pip_freeze() -> tuple[list[str], str | None]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover
        return [], str(exc)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines, None


def _git_info(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        info["commit"] = commit
    except Exception:
        return info

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        info["dirty"] = bool(status)
    except Exception:
        pass

    try:
        describe = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        info["describe"] = describe
    except Exception:
        pass
    return info


def _collect_raw_inputs(
    raw_dirs: Iterable[Path],
    repo_root: Path,
    *,
    include_hash: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir_records: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            raise FileNotFoundError(raw_dir)
        xdf_paths = sorted(raw_dir.glob("*.xdf"))
        if not xdf_paths:
            raise FileNotFoundError(f"No .xdf files found in {raw_dir}")
        raw_dir_records.append(
            {
                "path": _relative_path(raw_dir, repo_root),
                "file_count": len(xdf_paths),
            }
        )
        for path in xdf_paths:
            record = _file_record(path, repo_root, include_hash=include_hash)
            record["source_dir"] = _relative_path(raw_dir, repo_root)
            file_records.append(record)
    return raw_dir_records, file_records


def _collect_package_files(
    package_root: Path,
    repo_root: Path,
    *,
    include_hash: bool,
) -> list[dict[str, Any]]:
    include_exts = {".py", ".ps1", ".json", ".md", ".txt"}
    records: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in include_exts:
            continue
        records.append(_file_record(path, repo_root, include_hash=include_hash))
    return records


def _collect_bids_summaries(bids_roots: Iterable[Path], repo_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for bids_root in bids_roots:
        if not bids_root.exists():
            raise FileNotFoundError(bids_root)
        subjects = sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())
        file_count = 0
        total_bytes = 0
        for path in bids_root.rglob("*"):
            if path.is_file():
                file_count += 1
                total_bytes += path.stat().st_size
        summaries.append(
            {
                "path": _relative_path(bids_root, repo_root),
                "subject_count": len(subjects),
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
        )
    return summaries


def _collect_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in REQUIRED_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except Exception:
            packages[name] = "not-installed"

    freeze_lines, freeze_error = _pip_freeze()
    return {
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "pip_freeze": freeze_lines,
        "pip_freeze_error": freeze_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record conversion inputs, environment, and package files for reproducibility."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (defaults to current working directory).",
    )
    parser.add_argument(
        "--raw-dir",
        action="append",
        default=[],
        help="Raw XDF directory (repeatable).",
    )
    parser.add_argument("--xlsx", help="Path to participant_info.xlsx.")
    parser.add_argument(
        "--bids-root",
        action="append",
        default=[],
        help="BIDS dataset root to summarize (repeatable).",
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Include SHA256 hashes for raw inputs (slower).",
    )
    parser.add_argument(
        "--skip-package-hash",
        action="store_true",
        help="Skip hashing conversion_package files.",
    )
    parser.add_argument(
        "--requirements-out",
        help="Optional path to write pip freeze output (requirements.lock.txt).",
    )
    parser.add_argument(
        "--out",
        default=str(Path("conversion_package") / "reports" / "conversion_manifest.json"),
        help="Output JSON path for the manifest.",
    )
    args = parser.parse_args()

    repo_root = _resolve_path(Path.cwd(), args.repo_root)

    raw_dirs = [_resolve_path(repo_root, p) for p in args.raw_dir]
    xlsx_path = _resolve_path(repo_root, args.xlsx) if args.xlsx else None
    bids_roots = [_resolve_path(repo_root, p) for p in args.bids_root]

    raw_dir_records: list[dict[str, Any]] = []
    raw_file_records: list[dict[str, Any]] = []
    if raw_dirs:
        raw_dir_records, raw_file_records = _collect_raw_inputs(
            raw_dirs, repo_root, include_hash=args.hash_inputs
        )

    participant_info: dict[str, Any] | None = None
    if xlsx_path:
        if not xlsx_path.exists():
            raise FileNotFoundError(xlsx_path)
        participant_info = _file_record(xlsx_path, repo_root, include_hash=args.hash_inputs)

    package_root = repo_root / "conversion_package"
    if not package_root.exists():
        raise FileNotFoundError(package_root)
    package_files = _collect_package_files(
        package_root, repo_root, include_hash=not args.skip_package_hash
    )

    bids_summaries: list[dict[str, Any]] = []
    if bids_roots:
        bids_summaries = _collect_bids_summaries(bids_roots, repo_root)

    environment = _collect_environment()
    manifest: dict[str, Any] = {
        "created_utc": _utc_timestamp(),
        "repo_root": _relative_path(repo_root, repo_root),
        "environment": environment,
        "git": _git_info(repo_root),
        "inputs": {
            "raw_dirs": raw_dir_records,
            "raw_xdf": raw_file_records,
            "participant_info_xlsx": participant_info,
        },
        "conversion_package": {
            "files": package_files,
        },
        "bids_outputs": bids_summaries,
    }

    out_path = _resolve_path(repo_root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out_path}")

    if args.requirements_out:
        freeze_error = environment.get("pip_freeze_error")
        if freeze_error:
            raise RuntimeError(f"pip freeze failed: {freeze_error}")
        freeze_lines = environment.get("pip_freeze", [])
        req_path = _resolve_path(repo_root, args.requirements_out)
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text("\n".join(freeze_lines) + "\n", encoding="utf-8")
        print(f"Wrote requirements lock: {req_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
