# QT-nback_study
PyQt5 n-back task (baseline, 1-4 back) that emits Lab Streaming Layer (LSL) marker strings for each trial. The repo also includes an XDF-to-BIDS conversion pipeline under `conversion_package/`.

## Quickstart
1. Create a virtual environment and install dependencies.
2. Run the experiment from the repo root (or `n-back/`).

Create the venv:
```bash
python -m venv .venv
```

Activate it (pick your shell):
```powershell
.\.venv\Scripts\Activate.ps1
```
```cmd
.\.venv\Scripts\activate.bat
```
```bash
source .venv/bin/activate
```

Install requirements and run:
```bash
python -m pip install -r requirements.txt
python n-back/mainExperiment.py
```

Controls:
- Space: respond to a match
- Q: quit the experiment window

## Data capture and conversion
- The experiment emits LSL markers on the `n-backMarkers` stream; use Lab Recorder (or another LSL recorder) to save XDF files.
- Store raw recordings under `nback_Data/sourcedata`.
- The n-back BIDS conversion pipeline lives in `conversion_package/`; follow `conversion_package/README.md` to rebuild `bids_nback` from the XDFs and demographics.
- The QC plotter is included in `conversion_package/qc` for inspecting EEG/ECG/pupil with event overlays after conversion; see `conversion_package/qc/README.md` for usage.

## Requirements
- Python 3.9+
- OS: Windows (developed on Windows 11 Professional)
- PyQt5, pylsl, numpy, playsound (see `requirements.txt`)
- An LSL recorder/receiver if you want to capture markers
- On some Linux setups, `pylsl` may require a separate `liblsl` install

## Reproducibility notes
- The stimulus sequence is randomized using `random` and `numpy`; set seeds if you need deterministic sequences.
- This code does not save data to disk. Record markers with an LSL consumer if you need logs.

## Marker format
Markers are single strings on the `n-backMarkers` LSL stream.
- "Started tutorial n-back" and "Started n-back" are emitted at block start.
- For each trial, the result marker is `Steps:<N> KeyPress:<True|False> Matched:<True|False>`.
- "Finished tutorial n-back" and "Finished n-back" are emitted at block end.

## Timing parameters
Timing is defined in `n-back/mainExperiment.py` via `QTimer.singleShot` calls.
- Tutorial starts with a 60 s fixation ("x") before the first instructions.
- Experiment start uses a 1 s blank before the first instructions.
- The "Remember N steps back" instruction screen shows for 6.0 s.
- Each trial shows a letter for 1.0 s, then a 0.7 s blank before the next letter.
- Between blocks there is a 1.0 s blank, then the next instruction screen.

## Docker (optional)
The GUI requires a display server. On Linux hosts with X11:
```bash
docker build -t qt-nback .
xhost +local:docker
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix qt-nback
```

## Citation
See `CITATION.cff` for citation metadata. Update it with a DOI after creating the Zenodo record.

## License
This repository is released under CC0 1.0 Universal. See `LICENSE`.
