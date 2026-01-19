# BIDS Signal Viewer (Qt)

This folder contains an optional Qt-based viewer for EEG, ECG, and pupil physio
data with marker overlays. It is intended for quick visual QC of converted BIDS
outputs.

## Install (optional)

```powershell
# From the repo root (QT-nback_study)
python -m pip install -r .\conversion_package\requirements-qc.txt
```

If you prefer PySide6 over PyQt5:

```powershell
python -m pip install matplotlib PySide6
```

## Run

```powershell
python .\conversion_package\qc\bids_signal_viewer.py
```

Optional arguments:

```powershell
python .\conversion_package\qc\bids_signal_viewer.py --bids-root bids_nback
python .\conversion_package\qc\bids_signal_viewer.py --ecg-map conversion_package\config\ecg_channel_map.json
```

## Notes

- Dependencies: `mne`, `numpy`, `matplotlib`, and either `PyQt5` or `PySide6`.
- The viewer reads `*_eeg.vhdr`, `*_events.tsv`, and physio TSVs under `sub-*/ecg`
  and `sub-*/pupil` (or legacy eyetrack files if present).
- Marker stream colors can be toggled on/off in the right panel.
- Physio time is aligned using `StartTime` from the sidecar when present.
- Use the Overlay controls to add a secondary-axis channel from another stream
  (for example, pupil `confidence` with EEG `Fp1`/`Fp2`).
- If the overlay trace is too small or large, switch Scale to "Scale to base range".
- Adjust Base/Overlay width to make traces easier to see.
