# XDF to BIDS conversion package

This folder contains only the scripts and configuration needed to rebuild the
`bids_nback` dataset from raw XDF files, plus a concise explanation of the
steps.

## Contents

- `scripts/` conversion pipeline scripts
- `qc/` optional BIDS signal viewer (Qt)
- `config/ecg_channel_map.json` ECG channel map (updated after conversion)
- `config/dataset_description_defaults.json` dataset metadata defaults (Name, Authors, License, SourceDatasets)
- `config/sidecar_defaults.json` EEG/ECG sidecar defaults (reference, hardware, device metadata)
- `requirements.txt` Python dependencies
- `requirements-qc.txt` optional dependencies for the Qt viewer

## Reproducibility policy

- Treat `bids_nback/` as a build artifact; do not hand-edit files in this folder.
- Make changes only in `conversion_package/scripts` or `conversion_package/config`, then rebuild by rerunning Steps 1-5 in order.
- When the pipeline changes, update this README so other labs can reproduce the same outputs.

## Inputs required (outside this folder)

- Raw XDF files:
  - `nback_Data/sourcedata/*.xdf` (place raw XDFs here)
- Demographics: `participant_info.xlsx` with columns for participant, age, sex, handedness (height is intentionally omitted). The XLSX is copied into each BIDS `sourcedata/` folder for provenance, so keep it de-identified.

## Demographics input format

`participant_info.xlsx` is parsed from the first worksheet with a header row. The
pipeline looks for these columns (case-insensitive):

- participant ID: `participant`, `participant no.`, `participant_no`, or `participant no`
- `age`
- `sex` (expected values: `M`, `F`, `O`, or `n/a`)
- `handedness` (or `handidness`; expected values: `R`, `L`, `A`, or `n/a`)

If a cell is empty, the pipeline writes `n/a`. Ages are rounded to whole years
when they parse as numeric.

Template: `conversion_package/templates/participant_info_template.tsv` (open in Excel
and save as `participant_info.xlsx`). By default, Step 5 copies the XLSX into
`bids_nback/sourcedata/`; use `--skip-copy-xlsx` to avoid that.

IDs in BIDS match the original XDF filenames (e.g., `001_nback.xdf` -> `sub-001`).
Missing IDs correspond to excluded participants. Excluded participants are kept
in `participants.tsv` with `analysis_included=false`.

## Assumptions baked into the pipeline

- XDF filenames follow `NNN_nback.xdf` (3-digit IDs).
- EEG stream name: `EEGStream`; pupil stream: `pupil_capture`.
- Marker stream: `n-backMarkers`; dropout stream: `UoHDataOffsetStream`.
- ECG channel is either D1 or D3 (map stored in `config/ecg_channel_map.json`).
- Trial markers are emitted during the blank interval after each letter; events are stored with duration 0.0 s.
- Default line frequency is 50 Hz (override with `--line-freq` in Step 2 if needed).

## Quick start (venv)

From the repo root (`QT-nback_study`), create a virtual environment and
install the conversion dependencies.

Create the venv (use the Python you want for the pipeline, e.g. `python`,
`py -3.12`, or `python`):
```bash
python -m venv .venv_bids
```

Activate it (pick your shell):
```powershell
.\.venv_bids\Scripts\Activate.ps1
```
```cmd
.\.venv_bids\Scripts\activate.bat
```
```bash
source .venv_bids/bin/activate
```

If you have `PYTHONPATH` set (especially to a system site-packages), clear it
before installing so pip does not treat global packages as "already satisfied".
The conversion scripts also scrub any `PYTHONPATH` entries from `sys.path`, so
leaving it set can cause `ModuleNotFoundError` even when packages are installed
globally.
```powershell
Get-ChildItem Env:PYTHONPATH
$env:PYTHONPATH = ""
```

Install requirements:
```bash
python -m pip install -r conversion_package/requirements.txt
```

Optional: install the QC viewer dependencies:
```bash
python -m pip install -r conversion_package/requirements-qc.txt
```

All steps below assume the venv is activated so `python` points to it. On
Windows, if `python` resolves to the Microsoft Store stub or isn't found, use
the venv executable explicitly (for example `.\.venv_bids\Scripts\python.exe`
 To check, run:
```powershell
Get-ChildItem .\.venv_bids\Scripts\python*.exe
```

## Step 0 (optional): Clean prior derived outputs

Use a clean rebuild to avoid mixed states from older runs.

```powershell
Remove-Item -Recurse -Force bids_nback\sub-*
Remove-Item -Recurse -Force bids_nback\sourcedata\xdf
```

## Step 1: Copy raw XDF into BIDS sourcedata

```powershell
python .\conversion_package\scripts\prepare_sourcedata_xdf.py `
  --raw-dir nback_Data\sourcedata `
  --bids-root bids_nback `
  --identity `
  --report conversion_package\reports\participant_id_map_bids_nback.tsv
```

## Step 2: Convert EEGStream to BrainVision + BIDS

```powershell
python .\conversion_package\scripts\convert_xdf_to_bids_eeg.py `
  --xdf-dir nback_Data\sourcedata `
  --bids-root bids_nback `
  --task nback `
  --identity `
  --overwrite
```


## Step 3: Extract marker streams into events.tsv

This step also writes a `bids_stub/` tree with task-only `events.tsv` (onset
relative to the first marker). It is useful for quick marker QC without EEG
alignment. Add `--skip-bids-stub` to disable it or `--bids-stub-root` to redirect it.

```powershell
python .\conversion_package\scripts\extract_nback_markers.py `
  --xdf-dir nback_Data\sourcedata `
  --bids-root bids_nback
```

## Step 4: Postprocess (events metadata, ECG typing, pupil + ECG physio)

This step:
- Adds `marker_stream` to events files and updates `task-*_events.json`.
- Fixes ECG channel typing using `config/ecg_channel_map.json`.
- Applies EEG/ECG sidecar defaults from `config/sidecar_defaults.json`.
- Updates `dataset_description.json` fields from `config/dataset_description_defaults.json` (Name, Authors, License, SourceDatasets).
- Ensures dataset `README` references include the EEG system citations.
- Writes pupil files as `*_pupil.tsv` + `*_eyetrack.json` with `PhysioType=eyetrack` (time is relative to the eye-tracking stream start with `StartTime` carrying the EEG offset; drops Pupil Labs channels with non-finite values, clips `x_coordinate`/`y_coordinate` to [0,1], and omits `gaze_point_3d_*` columns; raw `norm_pos_*` values remain for reference). These sidecar/TSV pairs live under `sub-*/pupil` and are ignored by validators via `.bidsignore`.
- Writes ECG physio files as `*_recording-ecg_physio.tsv` + `.json` derived from the ECG channel, using the BIDS `cardiac` column label. The ECG exports are stored in `sub-*/ecg`, keeping them separate from the EEG directory while remaining at the subject top level.
- Writes `.bidsignore` with `/sub-*/ecg` and `/sub-*/pupil` so validator checks ignore ECG and pupil directories.
- Removes legacy physio files under `sub-*/eeg`, older `.tsv.gz` physio files, and any prior `sub-*/eyetrack` outputs so only the current outputs remain.
- Writes physio TSVs without header rows (column names are specified in each sidecar `Columns` list).
- Annotates numeric physio columns with `Format=number` in the sidecar JSON files.

```powershell
# N-back
python .\conversion_package\scripts\postprocess_bids.py `
  --bids-root bids_nback `
  --overwrite `
  --ecg-map conversion_package\config\ecg_channel_map.json `
  --metadata-config conversion_package\config\sidecar_defaults.json `
  --dataset-description-config conversion_package\config\dataset_description_defaults.json
```

Use `--dataset-description-config` to override dataset metadata defaults.

`config/ecg_channel_map.json` is treated as the source of truth and should be
maintained manually. Do not regenerate it from existing BIDS outputs (that is
circular and can overwrite corrected mappings). If you need to audit mappings,
write to a separate file under `conversion_package/reports/` and pass `--overwrite`.

## Step 5: Update participants.tsv/json

```powershell
python .\conversion_package\scripts\update_participants_from_xlsx.py `
  --bids-root bids_nback `
  --id-mode identity `
  --include-excluded `
  --report-dir conversion_package\reports
```

`update_participants_from_xlsx.py` copies `participant_info.xlsx` into
`bids_nback/sourcedata/` unless `--skip-copy-xlsx` is set.

## Step 6 (recommended for release): Record reproducibility metadata

This captures package versions, input inventories, and conversion_package file checksums.

```powershell
python .\conversion_package\scripts\record_conversion_manifest.py `
  --raw-dir nback_Data\sourcedata `
  --xlsx participant_info.xlsx `
  --bids-root bids_nback `
  --hash-inputs `
  --requirements-out conversion_package\requirements.lock.txt `
  --out conversion_package\reports\conversion_manifest.json
```

Omit `--hash-inputs` if you only need file sizes and mtimes (faster).

## .bidsignore (required)

Ensure `bids_nback/.bidsignore` includes:

```
/sub-*/ecg
/sub-*/pupil
```

Step 4 writes this file automatically; keep it for validation and release builds.

## Optional: QC signal viewer (Qt)

To browse the converted EEG, ECG, and pupil outputs with marker overlays, see
`conversion_package/qc/README.md` for setup and usage. This is optional and not
required for the conversion steps above.

## Optional: Remove physio headers (legacy output)

This script removes header rows when present so physio TSVs contain only data
rows, with column names stored in the sidecar `Columns` field. It scans
`sub-*/ecg`, `sub-*/pupil`, legacy `sub-*/eyetrack`, and `sub-*/eeg`
directories for `.tsv` and `.tsv.gz` physio files.

```powershell
python .\conversion_package\scripts\fix_physio_headers.py `
  --bids-root bids_nback
```

## Validation

```powershell
npx bids-validator@1.15.0 --version
npx bids-validator@1.15.0 bids_nback

# Optional: capture validator outputs for release notes
npx bids-validator@1.15.0 bids_nback > conversion_package\reports\bids_nback_validator.txt
```

## Release checklist (Zenodo)

- `conversion_package/README.md` reflects the exact pipeline and assumptions.
- `conversion_package/requirements.lock.txt` created (Step 6).
- `conversion_package/reports/conversion_manifest.json` created (Step 6).
- Validator outputs saved under `conversion_package/reports`.
