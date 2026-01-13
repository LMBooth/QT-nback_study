# QT-nback_study
Python/PyQt5 n-back task (baseline, 1-4 back) that emits Lab Streaming Layer (LSL) marker strings for each trial.

## Contents
- `n-back/mainExperiment.py`: experiment GUI and logic
- `n-back/Correct.mp3`, `n-back/Incorrect.mp3`: tutorial feedback sounds

## Requirements
- Python 3.9+ with GUI support
- Packages in `requirements.txt`
- Audio output for tutorial feedback
- Optional: an LSL receiver (for example, LabRecorder) if you want to record markers

## Tested environment
Update this section with the exact environment used for your release.
- OS: Windows (developed on Windows 11 Professional)
- Python: 3.x
- PyQt5: 5.15.x
- pylsl: 1.16.x
- numpy: 1.x
- playsound: 1.2.2

To capture exact versions on the acquisition machine:
```
python -c "import platform,sys,importlib.metadata as md; print(platform.platform()); print(sys.version); [print(p, md.version(p)) for p in ['PyQt5','pylsl','numpy','playsound']]"
```

## Installation
```
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running
```
python n-back/mainExperiment.py
```

Controls:
- Space: respond to a match
- Q: quit the experiment window

## LSL output
Markers are emitted on the `n-backMarkers` stream as strings. Examples:
- `Started tutorial n-back`
- `Steps:2 KeyPress:True Matched:True`
- `Finished n-back`

## Reproducibility notes
- The stimulus sequence is randomized using `random` and `numpy`; set seeds if you need deterministic sequences.
- This code does not save data to disk. Record markers with an LSL consumer if you need logs.

## Docker (optional)
The GUI requires a display server. On Linux hosts with X11:
```
docker build -t qt-nback .
xhost +local:docker
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix qt-nback
```
