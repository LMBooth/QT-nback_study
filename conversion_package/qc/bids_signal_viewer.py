from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError:
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError as exc:
        raise SystemExit("Install PyQt5 or PySide6 to run the Qt viewer.") from exc

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


TASK_RE = re.compile(r"_task-([a-zA-Z0-9]+)_")


@dataclass
class PhysioData:
    time: np.ndarray
    data: np.ndarray
    labels: list[str]
    start_time: float


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_events(events_path: Path) -> list[dict[str, str]]:
    if not events_path.exists():
        return []
    with events_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _sidecar_for_tsv(tsv_path: Path) -> Path:
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


def _read_sidecar_metadata(sidecar_path: Path) -> tuple[list[str] | None, float]:
    if not sidecar_path.exists():
        return None, 0.0
    with sidecar_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    columns = data.get("Columns")
    if not isinstance(columns, list) or not columns:
        columns_out = None
    else:
        columns_out = [str(col) for col in columns]
    start_time = data.get("StartTime", 0.0)
    try:
        start_time_val = float(start_time)
    except (TypeError, ValueError):
        start_time_val = 0.0
    if not math.isfinite(start_time_val):
        start_time_val = 0.0
    return columns_out, start_time_val


def _line_has_header(tokens: list[str]) -> bool:
    if not tokens or tokens == [""]:
        return False
    for token in tokens:
        if _safe_float(token) is None:
            return True
    return False


def _load_physio_data(path: Path) -> PhysioData:
    sidecar_path = _sidecar_for_tsv(path)
    columns, start_time = _read_sidecar_metadata(sidecar_path)

    with _open_text(path) as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Empty physio file: {path}")
        header_tokens = first_line.lstrip("\ufeff").split("\t")

        has_header = _line_has_header(header_tokens) or (columns and header_tokens == columns)
        if has_header:
            headers = columns or header_tokens
            data = np.genfromtxt(
                f,
                delimiter="\t",
                dtype=np.float64,
                missing_values="n/a",
                filling_values=np.nan,
            )
        else:
            headers = columns or header_tokens
            with _open_text(path) as full_f:
                data = np.genfromtxt(
                    full_f,
                    delimiter="\t",
                    dtype=np.float64,
                    missing_values="n/a",
                    filling_values=np.nan,
                )
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if "time" in headers:
        time_idx = headers.index("time")
    else:
        time_idx = 0
    data_cols = [idx for idx in range(len(headers)) if idx != time_idx]
    time = data[:, time_idx]
    if math.isfinite(start_time):
        time = time + start_time
    series = data[:, data_cols]
    labels = [headers[idx] for idx in data_cols]
    return PhysioData(time=time, data=series, labels=labels, start_time=start_time)


def _list_subjects(bids_root: Path) -> list[str]:
    return sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())


def _list_tasks(bids_root: Path, subject: str) -> list[str]:
    tasks: set[str] = set()
    eeg_dir = bids_root / subject / "eeg"
    if eeg_dir.exists():
        for path in eeg_dir.glob(f"{subject}_task-*_eeg.vhdr"):
            match = TASK_RE.search(path.name)
            if match:
                tasks.add(match.group(1))
        for path in eeg_dir.glob(f"{subject}_task-*_events.tsv"):
            match = TASK_RE.search(path.name)
            if match:
                tasks.add(match.group(1))

    eyetrack_dir = bids_root / subject / "eyetrack"
    if eyetrack_dir.exists():
        for path in eyetrack_dir.glob(f"{subject}_task-*_eyetrack.tsv*"):
            match = TASK_RE.search(path.name)
            if match:
                tasks.add(match.group(1))
        for path in eyetrack_dir.glob(f"{subject}_task-*_recording-eyetrack_physio.tsv*"):
            match = TASK_RE.search(path.name)
            if match:
                tasks.add(match.group(1))

    pupil_dir = bids_root / subject / "pupil"
    if pupil_dir.exists():
        for path in pupil_dir.glob(f"{subject}_task-*_pupil.tsv*"):
            match = TASK_RE.search(path.name)
            if match:
                tasks.add(match.group(1))

    ecg_dir = bids_root / subject / "ecg"
    if ecg_dir.exists():
        for path in ecg_dir.glob(f"{subject}_task-*_recording-ecg_physio.tsv*"):
            match = TASK_RE.search(path.name)
            if match:
                tasks.add(match.group(1))

    eyetrack_dir = bids_root / subject / "eyetrack"
    if eyetrack_dir.exists():
        for path in eyetrack_dir.glob(f"{subject}_task-*_recording-eyetrack_physio.tsv*"):
            match = TASK_RE.search(path.name)
            if match:
                tasks.add(match.group(1))
    return sorted(tasks)


def _find_eeg_vhdr(bids_root: Path, subject: str, task: str) -> Path | None:
    eeg_dir = bids_root / subject / "eeg"
    if not eeg_dir.exists():
        return None
    candidate = eeg_dir / f"{subject}_task-{task}_eeg.vhdr"
    if candidate.exists():
        return candidate
    matches = list(eeg_dir.glob(f"{subject}_task-*_eeg.vhdr"))
    return matches[0] if matches else None


def _find_pupil_physio(bids_root: Path, subject: str, task: str) -> Path | None:
    pupil_dir = bids_root / subject / "pupil"
    candidates = [
        pupil_dir / f"{subject}_task-{task}_pupil.tsv",
        pupil_dir / f"{subject}_task-{task}_pupil.tsv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for pattern in (f"{subject}_task-*_pupil.tsv", f"{subject}_task-*_pupil.tsv.gz"):
        matches = list(pupil_dir.glob(pattern))
        if matches:
            return matches[0]

    eyetrack_dir = bids_root / subject / "eyetrack"
    candidates = [
        eyetrack_dir / f"{subject}_task-{task}_eyetrack.tsv",
        eyetrack_dir / f"{subject}_task-{task}_eyetrack.tsv.gz",
        eyetrack_dir / f"{subject}_task-{task}_recording-eyetrack_physio.tsv",
        eyetrack_dir / f"{subject}_task-{task}_recording-eyetrack_physio.tsv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in (
        f"{subject}_task-*_eyetrack.tsv",
        f"{subject}_task-*_eyetrack.tsv.gz",
        f"{subject}_task-*_recording-eyetrack_physio.tsv",
        f"{subject}_task-*_recording-eyetrack_physio.tsv.gz",
    ):
        matches = list(eyetrack_dir.glob(pattern))
        if matches:
            return matches[0]

    eeg_dir = bids_root / subject / "eeg"
    candidates = [
        eeg_dir / f"{subject}_task-{task}_recording-eyetrack_physio.tsv.gz",
        eeg_dir / f"{subject}_task-{task}_recording-pupil_physio.tsv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in (
        f"{subject}_task-*_recording-eyetrack_physio.tsv.gz",
        f"{subject}_task-*_recording-pupil_physio.tsv.gz",
    ):
        matches = list(eeg_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_ecg_physio(bids_root: Path, subject: str, task: str) -> Path | None:
    ecg_dir = bids_root / subject / "ecg"
    candidates = [
        ecg_dir / f"{subject}_task-{task}_recording-ecg_physio.tsv",
        ecg_dir / f"{subject}_task-{task}_recording-ecg_physio.tsv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in (
        f"{subject}_task-*_recording-ecg_physio.tsv",
        f"{subject}_task-*_recording-ecg_physio.tsv.gz",
    ):
        matches = list(ecg_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_events(bids_root: Path, subject: str, task: str) -> Path | None:
    eeg_dir = bids_root / subject / "eeg"
    if not eeg_dir.exists():
        return None
    candidate = eeg_dir / f"{subject}_task-{task}_events.tsv"
    if candidate.exists():
        return candidate
    matches = list(eeg_dir.glob(f"{subject}_task-*_events.tsv"))
    return matches[0] if matches else None


def _load_ecg_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _default_root() -> Path:
    cwd = Path.cwd()
    for name in ("bids_nback",):
        candidate = cwd / name
        if candidate.exists():
            return candidate
    return cwd


def _color_palette() -> list[str]:
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


class BidsSignalViewer(QtWidgets.QMainWindow):
    def __init__(self, bids_root: Path, ecg_map_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("BIDS Signal Viewer")

        self._bids_root = bids_root
        self._ecg_map = _load_ecg_map(ecg_map_path) if ecg_map_path else {}
        self._raw = None
        self._raw_path: Path | None = None
        self._pupil: PhysioData | None = None
        self._pupil_path: Path | None = None
        self._ecg_physio: PhysioData | None = None
        self._ecg_physio_path: Path | None = None
        self._events: list[dict[str, str]] = []
        self._events_path: Path | None = None
        self._marker_colors: dict[str, str] = {}
        self._marker_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._overlay_color = "#d62728"
        self._min_time = 0.0
        self._max_time = 0.0
        self._time_scale = 1000.0
        self._time_initialized = False
        self._loading = False

        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._update_plot)

        self._build_ui()
        self._wire_signals()
        self._set_overlay_controls_enabled(self._overlay_enabled())
        self._set_root(self._bids_root)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QWidget(central)
        grid = QtWidgets.QGridLayout(controls)

        self.root_edit = QtWidgets.QLineEdit(str(self._bids_root), controls)
        self.root_browse = QtWidgets.QPushButton("Browse", controls)
        grid.addWidget(QtWidgets.QLabel("BIDS root"), 0, 0)
        grid.addWidget(self.root_edit, 0, 1, 1, 5)
        grid.addWidget(self.root_browse, 0, 6)

        self.subject_combo = QtWidgets.QComboBox(controls)
        self.task_combo = QtWidgets.QComboBox(controls)
        self.source_combo = QtWidgets.QComboBox(controls)
        self.source_combo.addItems(["EEG", "ECG", "Pupil"])
        grid.addWidget(QtWidgets.QLabel("Subject"), 1, 0)
        grid.addWidget(self.subject_combo, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Task"), 1, 2)
        grid.addWidget(self.task_combo, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Source"), 1, 4)
        grid.addWidget(self.source_combo, 1, 5)

        self.channel_combo = QtWidgets.QComboBox(controls)
        self.label_combo = QtWidgets.QComboBox(controls)
        self.label_combo.addItems(["trial_type", "marker"])
        grid.addWidget(QtWidgets.QLabel("Channel"), 2, 0)
        grid.addWidget(self.channel_combo, 2, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Marker label"), 2, 4)
        grid.addWidget(self.label_combo, 2, 5)

        self.overlay_check = QtWidgets.QCheckBox("Enable", controls)
        self.overlay_source_combo = QtWidgets.QComboBox(controls)
        self.overlay_source_combo.addItems(["EEG", "ECG", "Pupil"])
        self.overlay_channel_combo = QtWidgets.QComboBox(controls)
        self.overlay_scale_combo = QtWidgets.QComboBox(controls)
        self.overlay_scale_combo.addItems(["Right axis (raw)", "Scale to base range"])
        grid.addWidget(QtWidgets.QLabel("Overlay"), 3, 0)
        grid.addWidget(self.overlay_check, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Source"), 3, 2)
        grid.addWidget(self.overlay_source_combo, 3, 3)
        grid.addWidget(QtWidgets.QLabel("Channel"), 3, 4)
        grid.addWidget(self.overlay_channel_combo, 3, 5)
        grid.addWidget(QtWidgets.QLabel("Scale"), 3, 6)
        grid.addWidget(self.overlay_scale_combo, 3, 7)

        self.base_width_spin = QtWidgets.QDoubleSpinBox(controls)
        self.base_width_spin.setDecimals(2)
        self.base_width_spin.setRange(0.1, 5.0)
        self.base_width_spin.setSingleStep(0.1)
        self.base_width_spin.setValue(0.7)
        self.overlay_width_spin = QtWidgets.QDoubleSpinBox(controls)
        self.overlay_width_spin.setDecimals(2)
        self.overlay_width_spin.setRange(0.1, 5.0)
        self.overlay_width_spin.setSingleStep(0.1)
        self.overlay_width_spin.setValue(1.2)
        grid.addWidget(QtWidgets.QLabel("Base width"), 4, 0)
        grid.addWidget(self.base_width_spin, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Overlay width"), 4, 2)
        grid.addWidget(self.overlay_width_spin, 4, 3)

        self.start_spin = QtWidgets.QDoubleSpinBox(controls)
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.5)
        self.duration_spin = QtWidgets.QDoubleSpinBox(controls)
        self.duration_spin.setDecimals(3)
        self.duration_spin.setSingleStep(0.5)
        self.step_spin = QtWidgets.QDoubleSpinBox(controls)
        self.step_spin.setDecimals(3)
        self.step_spin.setSingleStep(0.5)
        self.prev_button = QtWidgets.QPushButton("Prev", controls)
        self.next_button = QtWidgets.QPushButton("Next", controls)

        grid.addWidget(QtWidgets.QLabel("Start (s)"), 5, 0)
        grid.addWidget(self.start_spin, 5, 1)
        grid.addWidget(QtWidgets.QLabel("Window (s)"), 5, 2)
        grid.addWidget(self.duration_spin, 5, 3)
        grid.addWidget(QtWidgets.QLabel("Step (s)"), 5, 4)
        grid.addWidget(self.step_spin, 5, 5)
        grid.addWidget(self.prev_button, 5, 6)
        grid.addWidget(self.next_button, 5, 7)

        self.start_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, controls)
        grid.addWidget(self.start_slider, 6, 0, 1, 8)

        main_layout.addWidget(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)
        plot_container = QtWidgets.QWidget(splitter)
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, plot_container)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        marker_panel = QtWidgets.QWidget(splitter)
        marker_layout = QtWidgets.QVBoxLayout(marker_panel)
        marker_layout.addWidget(QtWidgets.QLabel("Marker streams"))
        self.label_every_spin = QtWidgets.QSpinBox(marker_panel)
        self.label_every_spin.setRange(1, 999)
        self.label_every_spin.setValue(1)
        marker_layout.addWidget(QtWidgets.QLabel("Label every N markers"))
        marker_layout.addWidget(self.label_every_spin)

        self.marker_scroll = QtWidgets.QScrollArea(marker_panel)
        self.marker_scroll.setWidgetResizable(True)
        self.marker_list_widget = QtWidgets.QWidget(self.marker_scroll)
        self.marker_list_layout = QtWidgets.QVBoxLayout(self.marker_list_widget)
        self.marker_list_layout.addStretch(1)
        self.marker_scroll.setWidget(self.marker_list_widget)
        marker_layout.addWidget(self.marker_scroll)

        splitter.addWidget(plot_container)
        splitter.addWidget(marker_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _wire_signals(self) -> None:
        self.root_browse.clicked.connect(self._browse_root)
        self.root_edit.editingFinished.connect(self._root_edit_finished)
        self.subject_combo.currentIndexChanged.connect(self._selection_changed)
        self.task_combo.currentIndexChanged.connect(self._selection_changed)
        self.source_combo.currentIndexChanged.connect(self._selection_changed)
        self.channel_combo.currentIndexChanged.connect(self._schedule_plot)
        self.label_combo.currentIndexChanged.connect(self._schedule_plot)
        self.overlay_check.stateChanged.connect(self._overlay_toggled)
        self.overlay_source_combo.currentIndexChanged.connect(self._overlay_selection_changed)
        self.overlay_channel_combo.currentIndexChanged.connect(self._schedule_plot)
        self.overlay_scale_combo.currentIndexChanged.connect(self._schedule_plot)
        self.base_width_spin.valueChanged.connect(self._schedule_plot)
        self.overlay_width_spin.valueChanged.connect(self._schedule_plot)
        self.label_every_spin.valueChanged.connect(self._schedule_plot)
        self.start_spin.valueChanged.connect(self._start_changed)
        self.duration_spin.valueChanged.connect(self._duration_changed)
        self.start_slider.valueChanged.connect(self._slider_changed)
        self.prev_button.clicked.connect(self._step_backward)
        self.next_button.clicked.connect(self._step_forward)

    def _overlay_enabled(self) -> bool:
        return self.overlay_check.isChecked()

    def _set_overlay_controls_enabled(self, enabled: bool) -> None:
        self.overlay_source_combo.setEnabled(enabled)
        self.overlay_channel_combo.setEnabled(enabled)
        self.overlay_scale_combo.setEnabled(enabled)
        self.overlay_width_spin.setEnabled(enabled)

    def _overlay_toggled(self) -> None:
        enabled = self._overlay_enabled()
        self._set_overlay_controls_enabled(enabled)
        if enabled:
            subject = self._current_subject()
            task = self._current_task()
            if subject and task:
                self._refresh_overlay_data(subject, task)
        else:
            self.overlay_channel_combo.clear()
        self._schedule_plot()

    def _overlay_selection_changed(self) -> None:
        subject = self._current_subject()
        task = self._current_task()
        if subject and task:
            self._refresh_overlay_data(subject, task)
        self._schedule_plot()

    def _set_root(self, root: Path) -> None:
        self._bids_root = root
        self.root_edit.setText(str(root))
        subjects = _list_subjects(root)
        with QtCore.QSignalBlocker(self.subject_combo):
            self.subject_combo.clear()
            self.subject_combo.addItems(subjects)
        if subjects:
            self.subject_combo.setCurrentIndex(0)
        self._selection_changed()

    def _browse_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select BIDS root", str(self._bids_root)
        )
        if selected:
            self._set_root(Path(selected))

    def _root_edit_finished(self) -> None:
        path = Path(self.root_edit.text()).expanduser()
        if path.exists():
            self._set_root(path)
        else:
            self.statusBar().showMessage(f"Path not found: {path}")

    def _selection_changed(self) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            subject = self._current_subject()
            if subject:
                tasks = _list_tasks(self._bids_root, subject)
            else:
                tasks = []
            with QtCore.QSignalBlocker(self.task_combo):
                current_task = self._current_task()
                self.task_combo.clear()
                self.task_combo.addItems(tasks)
                if current_task in tasks:
                    self.task_combo.setCurrentText(current_task)
                elif tasks:
                    self.task_combo.setCurrentIndex(0)
            self._time_initialized = False
            self._raw = None
            self._raw_path = None
            self._pupil = None
            self._pupil_path = None
            self._ecg_physio = None
            self._ecg_physio_path = None
            self._events = []
            self._events_path = None
            self._refresh_data()
        finally:
            self._loading = False

    def _current_subject(self) -> str | None:
        return self.subject_combo.currentText() or None

    def _current_task(self) -> str | None:
        return self.task_combo.currentText() or None

    def _current_source(self) -> str:
        return self.source_combo.currentText()

    def _refresh_data(self) -> None:
        subject = self._current_subject()
        task = self._current_task()
        if not subject or not task:
            self._clear_plot()
            return

        self._ensure_events_loaded(subject, task)

        source = self._current_source()
        if source == "EEG":
            self._ensure_raw_loaded(subject, task)
            self._update_channel_list(source, subject, task)
            self._update_time_bounds_from_raw()
        elif source == "ECG":
            self._ensure_ecg_physio_loaded(subject, task)
            if self._ecg_physio is None:
                self._ensure_raw_loaded(subject, task)
            self._update_channel_list(source, subject, task)
            if self._ecg_physio is not None:
                self._update_time_bounds_from_physio(self._ecg_physio)
            else:
                self._update_time_bounds_from_raw()
        else:
            self._ensure_pupil_loaded(subject, task)
            self._update_channel_list(source, subject, task)
            self._update_time_bounds_from_pupil()

        self._refresh_overlay_data(subject, task)
        self._update_marker_controls()
        self._schedule_plot()

    def _refresh_overlay_data(self, subject: str, task: str) -> None:
        if not self._overlay_enabled():
            return

        source = self.overlay_source_combo.currentText()
        if source == "EEG":
            self._ensure_raw_loaded(subject, task)
        elif source == "ECG":
            self._ensure_ecg_physio_loaded(subject, task)
            if self._ecg_physio is None:
                self._ensure_raw_loaded(subject, task)
        else:
            self._ensure_pupil_loaded(subject, task)

        self._update_overlay_channel_list(source, subject, task)

    def _update_overlay_channel_list(self, source: str, subject: str, task: str) -> None:
        if source == "EEG" or (source == "ECG" and self._ecg_physio is None):
            if self._raw is None:
                self.overlay_channel_combo.clear()
                return
            channels = list(self._raw.ch_names)
            with QtCore.QSignalBlocker(self.overlay_channel_combo):
                current = self.overlay_channel_combo.currentText()
                self.overlay_channel_combo.clear()
                self.overlay_channel_combo.addItems(channels)
                preferred = None
                if source == "ECG":
                    preferred = self._ecg_map.get(task, {}).get(subject)
                if not preferred:
                    for candidate in ("Fp1", "Fp2", "Cz"):
                        if candidate in channels:
                            preferred = candidate
                            break
                if current in channels:
                    self.overlay_channel_combo.setCurrentText(current)
                elif preferred and preferred in channels:
                    self.overlay_channel_combo.setCurrentText(preferred)
                elif channels:
                    self.overlay_channel_combo.setCurrentIndex(0)
        elif source == "ECG":
            if self._ecg_physio is None:
                self.overlay_channel_combo.clear()
                return
            labels = list(self._ecg_physio.labels)
            with QtCore.QSignalBlocker(self.overlay_channel_combo):
                current = self.overlay_channel_combo.currentText()
                self.overlay_channel_combo.clear()
                self.overlay_channel_combo.addItems(labels)
                if current in labels:
                    self.overlay_channel_combo.setCurrentText(current)
                elif "cardiac" in labels:
                    self.overlay_channel_combo.setCurrentText("cardiac")
                elif labels:
                    self.overlay_channel_combo.setCurrentIndex(0)
        else:
            if self._pupil is None:
                self.overlay_channel_combo.clear()
                return
            labels = list(self._pupil.labels)
            with QtCore.QSignalBlocker(self.overlay_channel_combo):
                current = self.overlay_channel_combo.currentText()
                self.overlay_channel_combo.clear()
                self.overlay_channel_combo.addItems(labels)
                preferred = None
                if "confidence" in labels:
                    preferred = "confidence"
                else:
                    for candidate in ("diameter0_2d", "diameter1_2d", "diameter0_3d", "diameter1_3d"):
                        if candidate in labels:
                            preferred = candidate
                            break
                if current in labels:
                    self.overlay_channel_combo.setCurrentText(current)
                elif preferred and preferred in labels:
                    self.overlay_channel_combo.setCurrentText(preferred)
                elif labels:
                    self.overlay_channel_combo.setCurrentIndex(0)

    def _ensure_raw_loaded(self, subject: str, task: str) -> None:
        vhdr = _find_eeg_vhdr(self._bids_root, subject, task)
        if vhdr is None:
            self.statusBar().showMessage("EEG file not found for selection.")
            self._raw = None
            self._raw_path = None
            return
        if self._raw_path == vhdr and self._raw is not None:
            return
        self._raw_path = vhdr
        self._raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")

    def _ensure_pupil_loaded(self, subject: str, task: str) -> None:
        physio = _find_pupil_physio(self._bids_root, subject, task)
        if physio is None:
            self.statusBar().showMessage("Pupil physio file not found for selection.")
            self._pupil = None
            self._pupil_path = None
            return
        if self._pupil_path == physio and self._pupil is not None:
            return
        self._pupil_path = physio
        self._pupil = _load_physio_data(physio)

    def _ensure_ecg_physio_loaded(self, subject: str, task: str) -> None:
        physio = _find_ecg_physio(self._bids_root, subject, task)
        if physio is None:
            self._ecg_physio = None
            self._ecg_physio_path = None
            return
        if self._ecg_physio_path == physio and self._ecg_physio is not None:
            return
        self._ecg_physio_path = physio
        self._ecg_physio = _load_physio_data(physio)

    def _ensure_events_loaded(self, subject: str, task: str) -> None:
        events_path = _find_events(self._bids_root, subject, task)
        if events_path is None:
            self._events = []
            self._events_path = None
            return
        if self._events_path == events_path:
            return
        self._events_path = events_path
        self._events = _read_events(events_path)

    def _update_channel_list(self, source: str, subject: str, task: str) -> None:
        if source == "EEG" or (source == "ECG" and self._ecg_physio is None):
            if self._raw is None:
                self.channel_combo.clear()
                return
            channels = list(self._raw.ch_names)
            with QtCore.QSignalBlocker(self.channel_combo):
                current = self.channel_combo.currentText()
                self.channel_combo.clear()
                self.channel_combo.addItems(channels)
                preferred = None
                if source == "ECG":
                    preferred = self._ecg_map.get(task, {}).get(subject)
                if preferred and preferred in channels:
                    self.channel_combo.setCurrentText(preferred)
                elif current in channels:
                    self.channel_combo.setCurrentText(current)
                elif "Cz" in channels:
                    self.channel_combo.setCurrentText("Cz")
                elif channels:
                    self.channel_combo.setCurrentIndex(0)
        elif source == "ECG":
            if self._ecg_physio is None:
                self.channel_combo.clear()
                return
            labels = list(self._ecg_physio.labels)
            with QtCore.QSignalBlocker(self.channel_combo):
                current = self.channel_combo.currentText()
                self.channel_combo.clear()
                self.channel_combo.addItems(labels)
                if current in labels:
                    self.channel_combo.setCurrentText(current)
                elif "cardiac" in labels:
                    self.channel_combo.setCurrentText("cardiac")
                elif labels:
                    self.channel_combo.setCurrentIndex(0)
        else:
            if self._pupil is None:
                self.channel_combo.clear()
                return
            labels = list(self._pupil.labels)
            with QtCore.QSignalBlocker(self.channel_combo):
                current = self.channel_combo.currentText()
                self.channel_combo.clear()
                self.channel_combo.addItems(labels)
                preferred = None
                for candidate in ("diameter0_2d", "diameter1_2d", "diameter0_3d", "diameter1_3d"):
                    if candidate in labels:
                        preferred = candidate
                        break
                if current in labels:
                    self.channel_combo.setCurrentText(current)
                elif preferred:
                    self.channel_combo.setCurrentText(preferred)
                elif labels:
                    self.channel_combo.setCurrentIndex(0)

    def _update_time_bounds_from_raw(self) -> None:
        if self._raw is None:
            return
        self._min_time = float(self._raw.times[0])
        self._max_time = float(self._raw.times[-1])
        self._sync_time_controls()

    def _update_time_bounds_from_physio(self, physio: PhysioData) -> None:
        time = physio.time
        self._min_time = float(np.nanmin(time))
        self._max_time = float(np.nanmax(time))
        self._sync_time_controls()

    def _update_time_bounds_from_pupil(self) -> None:
        if self._pupil is None:
            return
        self._update_time_bounds_from_physio(self._pupil)

    def _sync_time_controls(self) -> None:
        span = max(0.0, self._max_time - self._min_time)
        slider_max = int(round(span * self._time_scale))
        self.start_slider.setRange(0, max(0, slider_max))

        if not self._time_initialized:
            default_window = min(60.0, span) if span > 0 else 10.0
            with QtCore.QSignalBlocker(self.start_spin):
                self.start_spin.setRange(self._min_time, self._max_time)
                self.start_spin.setValue(self._min_time)
            with QtCore.QSignalBlocker(self.duration_spin):
                self.duration_spin.setRange(0.1, max(0.1, span))
                self.duration_spin.setValue(default_window)
            with QtCore.QSignalBlocker(self.step_spin):
                self.step_spin.setRange(0.1, max(0.1, span))
                self.step_spin.setValue(max(0.5, min(10.0, default_window / 2)))
            self._time_initialized = True
        else:
            with QtCore.QSignalBlocker(self.start_spin):
                self.start_spin.setRange(self._min_time, self._max_time)
            with QtCore.QSignalBlocker(self.duration_spin):
                self.duration_spin.setRange(0.1, max(0.1, span))
        self._clamp_start()

    def _clamp_start(self) -> None:
        start = float(self.start_spin.value())
        duration = float(self.duration_spin.value())
        max_start = self._max_time - duration
        if max_start < self._min_time:
            max_start = self._min_time
        if start < self._min_time:
            start = self._min_time
        if start > max_start:
            start = max_start
        with QtCore.QSignalBlocker(self.start_spin):
            self.start_spin.setValue(start)
        self._sync_slider_from_start(start)

    def _sync_slider_from_start(self, start: float) -> None:
        value = int(round((start - self._min_time) * self._time_scale))
        value = max(self.start_slider.minimum(), min(self.start_slider.maximum(), value))
        with QtCore.QSignalBlocker(self.start_slider):
            self.start_slider.setValue(value)

    def _update_marker_controls(self) -> None:
        streams = sorted({row.get("marker_stream", "").strip() or "unknown" for row in self._events})
        palette = _color_palette()
        self._marker_colors = {stream: palette[idx % len(palette)] for idx, stream in enumerate(streams)}

        while self.marker_list_layout.count():
            item = self.marker_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._marker_checks.clear()

        for stream in streams:
            checkbox = QtWidgets.QCheckBox(stream, self.marker_list_widget)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(f"color: {self._marker_colors[stream]};")
            checkbox.stateChanged.connect(self._schedule_plot)
            self.marker_list_layout.addWidget(checkbox)
            self._marker_checks[stream] = checkbox

        if not streams:
            empty_label = QtWidgets.QLabel("No markers found.", self.marker_list_widget)
            self.marker_list_layout.addWidget(empty_label)
        self.marker_list_layout.addStretch(1)

    def _schedule_plot(self) -> None:
        if self._plot_timer.isActive():
            self._plot_timer.stop()
        self._plot_timer.start(50)

    def _start_changed(self) -> None:
        self._clamp_start()
        self._schedule_plot()

    def _duration_changed(self) -> None:
        span = max(0.0, self._max_time - self._min_time)
        duration = float(self.duration_spin.value())
        if duration > span and span > 0:
            with QtCore.QSignalBlocker(self.duration_spin):
                self.duration_spin.setValue(span)
        self._clamp_start()
        self._schedule_plot()

    def _slider_changed(self, value: int) -> None:
        start = self._min_time + (value / self._time_scale)
        with QtCore.QSignalBlocker(self.start_spin):
            self.start_spin.setValue(start)
        self._schedule_plot()

    def _step_backward(self) -> None:
        step = float(self.step_spin.value())
        with QtCore.QSignalBlocker(self.start_spin):
            self.start_spin.setValue(self.start_spin.value() - step)
        self._start_changed()

    def _step_forward(self) -> None:
        step = float(self.step_spin.value())
        with QtCore.QSignalBlocker(self.start_spin):
            self.start_spin.setValue(self.start_spin.value() + step)
        self._start_changed()

    def _clear_plot(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def _get_overlay_series(
        self, source: str, channel: str, start: float, end: float
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        if not channel:
            return None
        if source == "EEG" or (source == "ECG" and self._ecg_physio is None):
            if self._raw is None or channel not in self._raw.ch_names:
                return None
            start_idx, end_idx = self._raw.time_as_index([start, end], use_rounding=True)
            if end_idx <= start_idx:
                return None
            data = self._raw.get_data(picks=[channel], start=start_idx, stop=end_idx)[0] * 1e6
            times = self._raw.times[start_idx:end_idx]
            label = f"{source}: {channel} (uV)"
            return self._sanitize_overlay_series(times, data, label)
        if source == "ECG":
            if self._ecg_physio is None or channel not in self._ecg_physio.labels:
                return None
            idx = self._ecg_physio.labels.index(channel)
            time = self._ecg_physio.time
            mask = (time >= start) & (time <= end)
            times = time[mask]
            data = self._ecg_physio.data[mask, idx]
            if times.size == 0:
                return None
            label = f"ECG: {channel}"
            return self._sanitize_overlay_series(times, data, label)
        if self._pupil is None or channel not in self._pupil.labels:
            return None
        idx = self._pupil.labels.index(channel)
        time = self._pupil.time
        mask = (time >= start) & (time <= end)
        times = time[mask]
        data = self._pupil.data[mask, idx]
        if times.size == 0:
            return None
        label = f"Pupil: {channel}"
        return self._sanitize_overlay_series(times, data, label)

    def _sanitize_overlay_series(
        self, times: np.ndarray, data: np.ndarray, label: str
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        times = np.asarray(times).reshape(-1)
        data = np.asarray(data).reshape(-1)
        if times.size == 0 or data.size == 0:
            return None
        if times.size != data.size:
            n = min(times.size, data.size)
            if n == 0:
                return None
            times = times[:n]
            data = data[:n]
        return times, data, label

    def _update_plot(self) -> None:
        subject = self._current_subject()
        task = self._current_task()
        if not subject or not task:
            self._clear_plot()
            return

        source = self._current_source()
        channel = self.channel_combo.currentText()
        start = float(self.start_spin.value())
        duration = float(self.duration_spin.value())
        end = start + duration

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        base_width = float(self.base_width_spin.value())
        overlay_width = float(self.overlay_width_spin.value())

        base_data = None
        if source == "EEG" or (source == "ECG" and self._ecg_physio is None):
            if self._raw is None or channel not in self._raw.ch_names:
                ax.text(0.5, 0.5, "No EEG/ECG data loaded.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            start_idx, end_idx = self._raw.time_as_index([start, end], use_rounding=True)
            if end_idx <= start_idx:
                ax.text(0.5, 0.5, "Empty time window.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            base_data = self._raw.get_data(picks=[channel], start=start_idx, stop=end_idx)[0] * 1e6
            base_times = self._raw.times[start_idx:end_idx]
            ax.plot(base_times, base_data, color="black", linewidth=base_width)
            ax.set_ylabel("Amplitude (uV)")
        elif source == "ECG":
            if self._ecg_physio is None:
                ax.text(0.5, 0.5, "No ECG physio data loaded.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            if channel not in self._ecg_physio.labels:
                ax.text(0.5, 0.5, "ECG channel not found.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            idx = self._ecg_physio.labels.index(channel)
            time = self._ecg_physio.time
            mask = (time >= start) & (time <= end)
            base_times = time[mask]
            base_data = self._ecg_physio.data[mask, idx]
            if base_times.size == 0:
                ax.text(0.5, 0.5, "Empty time window.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            ax.plot(base_times, base_data, color="black", linewidth=base_width)
            ax.set_ylabel(channel)
        else:
            if self._pupil is None:
                ax.text(0.5, 0.5, "No pupil data loaded.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            if channel not in self._pupil.labels:
                ax.text(0.5, 0.5, "Pupil channel not found.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            idx = self._pupil.labels.index(channel)
            time = self._pupil.time
            mask = (time >= start) & (time <= end)
            base_times = time[mask]
            base_data = self._pupil.data[mask, idx]
            if base_times.size == 0:
                ax.text(0.5, 0.5, "Empty time window.", ha="center", va="center")
                self.canvas.draw_idle()
                return
            ax.plot(base_times, base_data, color="black", linewidth=base_width)
            ax.set_ylabel(channel)

        y_min = float(np.nanmin(base_data)) if base_data.size else 0.0
        y_max = float(np.nanmax(base_data)) if base_data.size else 1.0
        y_range = y_max - y_min
        if not math.isfinite(y_range) or y_range == 0:
            y_range = 1.0
        pad = y_range * 0.2

        if self._overlay_enabled():
            overlay_source = self.overlay_source_combo.currentText()
            overlay_channel = self.overlay_channel_combo.currentText()
            overlay_scale = self.overlay_scale_combo.currentText()
            overlay = self._get_overlay_series(overlay_source, overlay_channel, start, end)
            if overlay is not None:
                overlay_times, overlay_data, overlay_label = overlay
                color = self._overlay_color
                if overlay_scale == "Scale to base range":
                    valid = overlay_data[np.isfinite(overlay_data)]
                    if valid.size:
                        if valid.size > 10:
                            o_low = float(np.percentile(valid, 5))
                            o_high = float(np.percentile(valid, 95))
                        else:
                            o_low = float(np.min(valid))
                            o_high = float(np.max(valid))
                        if not math.isfinite(o_high - o_low) or o_high == o_low:
                            o_low = float(np.min(valid))
                            o_high = float(np.max(valid))
                        if math.isfinite(o_high - o_low) and o_high != o_low:
                            scaled = (overlay_data - o_low) / (o_high - o_low)
                            scaled = np.clip(scaled, 0.0, 1.0)
                            overlay_scaled = y_min + scaled * y_range
                        else:
                            overlay_scaled = np.full_like(overlay_data, y_min)
                        ax.plot(
                            overlay_times,
                            overlay_scaled,
                            color=color,
                            linewidth=overlay_width,
                            alpha=0.9,
                            linestyle="--",
                            zorder=3,
                        )
                else:
                    ax2 = ax.twinx()
                    ax2.plot(
                        overlay_times,
                        overlay_data,
                        color=color,
                        linewidth=overlay_width,
                        alpha=0.9,
                        zorder=3,
                    )
                    ax2.set_ylabel(overlay_label, color=color)
                    ax2.tick_params(axis="y", labelcolor=color)

        ax.set_xlabel("Time (s)")
        ax.set_title(f"{subject} task-{task} {channel} ({start:.2f}-{end:.2f}s)")

        enabled_streams = {
            stream for stream, checkbox in self._marker_checks.items() if checkbox.isChecked()
        }
        label_field = self.label_combo.currentText()
        label_every = int(self.label_every_spin.value())
        label_levels = 3

        if self._events:
            for idx, row in enumerate(self._events):
                onset = _safe_float(row.get("onset", ""))
                if onset is None:
                    continue
                if onset < start or onset > end:
                    continue
                stream = row.get("marker_stream", "").strip() or "unknown"
                if stream not in enabled_streams:
                    continue
                color = self._marker_colors.get(stream, "royalblue")
                ax.axvline(onset, color=color, alpha=0.7, linewidth=0.8)
                if label_every > 0 and (idx % label_every == 0):
                    label = row.get(label_field, "")
                    if label:
                        level = idx % label_levels
                        y_text = y_max + pad * (1 + 0.4 * level)
                        ax.text(
                            onset,
                            y_text,
                            label,
                            rotation=90,
                            fontsize=6,
                            color=color,
                            va="top",
                            ha="center",
                            alpha=0.8,
                        )

        ax.set_xlim(start, end)
        ax.set_ylim(y_min - pad * 0.2, y_max + pad * (1 + 0.4 * label_levels))
        ax.grid(True, alpha=0.2, linewidth=0.5)
        self.figure.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qt viewer for BIDS EEG/ECG/pupil data with markers.")
    parser.add_argument("--bids-root", default=None, help="Path to a BIDS dataset root.")
    parser.add_argument(
        "--ecg-map",
        default=None,
        help="Path to ecg_channel_map.json (defaults to conversion_package/config/ecg_channel_map.json).",
    )
    args = parser.parse_args()

    default_root = Path(args.bids_root) if args.bids_root else _default_root()
    default_ecg_map = None
    if args.ecg_map:
        default_ecg_map = Path(args.ecg_map)
    else:
        default_ecg_map = Path(__file__).resolve().parent.parent / "config" / "ecg_channel_map.json"

    app = QtWidgets.QApplication(sys.argv)
    viewer = BidsSignalViewer(default_root, default_ecg_map)
    viewer.resize(1200, 800)
    viewer.show()
    exit_code = app.exec() if hasattr(app, "exec") else app.exec_()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
