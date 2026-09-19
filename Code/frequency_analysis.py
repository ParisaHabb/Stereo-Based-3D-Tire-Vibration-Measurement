"""Frequency analysis of reconstructed tire-stud trajectories.

Save as:
    frequency_analysis.py

The script opens an interactive 3D view. Use the sliders to change the
high-pass and low-pass limits. Click a stud to open its time-domain and
frequency-domain detail view.
"""

from pathlib import Path
import re
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks


INPUT_FILE = Path("data/processed/final_3d_paths.csv")
FPS = 75.0
MIN_FREQUENCY = 0.5
MAX_DISPLAY_FREQUENCY = 20.0


def extract_frame_number(value: object) -> int:
    """Extract the last integer from a frame identifier."""
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[-1]) if numbers else -1


def load_data(filename: Path) -> pd.DataFrame:
    """Load and validate the reconstructed 3D trajectory table."""
    if not filename.exists():
        raise FileNotFoundError(f"3D data file not found: {filename}")

    data = pd.read_csv(filename)
    required = {"Frame_ID", "Stud_ID", "X", "Y", "Z"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "3D data is missing columns: " + ", ".join(sorted(missing))
        )

    data = data.copy()
    for column in ("X", "Y", "Z"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Frame_ID", "Stud_ID", "X", "Y", "Z"])
    data["Frame_Num"] = data["Frame_ID"].map(extract_frame_number)
    data = data.sort_values(["Stud_ID", "Frame_Num"])

    if data.empty:
        raise ValueError("No valid 3D trajectory samples were found.")
    return data


def safe_filter(signal: np.ndarray, low_cut: float, high_cut: float) -> np.ndarray:
    """Apply stable Butterworth filtering when enough samples are available."""
    output = np.asarray(signal, dtype=float).copy()
    if output.size < 16:
        return output - np.mean(output)

    nyquist = FPS / 2.0
    low_cut = max(0.0, float(low_cut))
    high_cut = min(float(high_cut), nyquist * 0.99)

    if low_cut > 0.1 and high_cut > low_cut:
        b, a = butter(2, [low_cut, high_cut], btype="bandpass", fs=FPS)
    elif low_cut > 0.1:
        b, a = butter(2, low_cut, btype="highpass", fs=FPS)
    elif high_cut < nyquist * 0.98:
        b, a = butter(2, high_cut, btype="lowpass", fs=FPS)
    else:
        return output - np.mean(output)

    pad_length = 3 * max(len(a), len(b))
    if output.size <= pad_length:
        return output - np.mean(output)
    return filtfilt(b, a, output) 


def spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a one-sided Hann-windowed amplitude spectrum."""
    values = np.asarray(signal, dtype=float)
    count = len(values)
    if count < 2:
        return np.array([]), np.array([])

    windowed = (values - np.mean(values)) * np.hanning(count)
    frequencies = fftfreq(count, 1.0 / FPS)[: count // 2]
    magnitude = 2.0 / count * np.abs(fft(windowed)[: count // 2])
    return frequencies, magnitude


def dominant_frequency(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
) -> float:
    """Return the strongest peak above MIN_FREQUENCY."""
    if len(frequencies) == 0 or len(magnitude) == 0:
        return 0.0

    maximum = float(np.max(magnitude))
    if maximum <= 0:
        return 0.0

    peaks, properties = find_peaks(
        magnitude,
        height=maximum * 0.1,
    )
    valid = [
        index
        for index in peaks
        if frequencies[index] > MIN_FREQUENCY
    ]
    if not valid:
        return 0.0
    return float(max(valid, key=lambda index: magnitude[index]))


class DetailWindow:
    """Display one stud's filtered signal and spectrum."""

    def __init__(self, signal_data: Dict[str, np.ndarray], low_cut: float, high_cut: float):
        self.signal_data = signal_data
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.figure, (self.time_axis, self.frequency_axis) = plt.subplots(
            1,
            2,
            figsize=(11, 5),
        )
        self.update()

    def update(self) -> None:
        raw = self.signal_data["y"]
        time = self.signal_data["t"]
        filtered = safe_filter(raw, self.low_cut, self.high_cut)
        frequencies, magnitude = spectrum(filtered)
        peak = dominant_frequency(frequencies, magnitude)

        self.time_axis.clear()
        self.frequency_axis.clear()
        self.time_axis.plot(time, raw, "k--", alpha=0.35, label="Raw")
        self.time_axis.plot(time, filtered, "b-", label="Filtered")
        self.time_axis.set_title("Time domain")
        self.time_axis.set_xlabel("Time (s)")
        self.time_axis.legend()

        self.frequency_axis.plot(frequencies, magnitude, "r-")
        self.frequency_axis.set_xlim(0, MAX_DISPLAY_FREQUENCY)
        self.frequency_axis.set_xlabel("Frequency (Hz)")
        self.frequency_axis.set_title(f"Spectrum | Peak: {peak:.2f} Hz")
        if peak > 0:
            self.frequency_axis.axvline(
                peak,
                color="red",
                linestyle="--",
                alpha=0.5,
            )

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()


class TireFrequencyViewer:
    """Interactive 3D stud view with filter controls."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.stud_ids = list(data["Stud_ID"].drop_duplicates())
        self.low_cut = 0.0
        self.high_cut = min(37.5, FPS * 0.49)
        self.cache: Dict[object, Dict[str, np.ndarray]] = {}
        self.detail_windows = []
        self._prepare_cache()
        self._setup_figure()
        self.update_labels(None)

    def _prepare_cache(self) -> None:
        for stud_id in self.stud_ids:
            subset = self.data[self.data["Stud_ID"] == stud_id].sort_values("Frame_Num")
            self.cache[stud_id] = {
                "y": subset["Y"].to_numpy(dtype=float),
                "t": subset["Frame_Num"].to_numpy(dtype=float) / FPS,
                "x_mean": np.asarray([subset["X"].mean()]),
                "y_mean": np.asarray([subset["Y"].mean()]),
                "z_mean": np.asarray([subset["Z"].mean()]),
                "trail_x": subset["X"].to_numpy(dtype=float),
                "trail_y": subset["Y"].to_numpy(dtype=float),
                "trail_z": subset["Z"].to_numpy(dtype=float),
            }

    def _setup_figure(self) -> None:
        self.figure = plt.figure(figsize=(14, 10))
        self.axis = self.figure.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=0.18)

        high_axis = self.figure.add_axes([0.15, 0.06, 0.30, 0.03])
        low_axis = self.figure.add_axes([0.55, 0.06, 0.30, 0.03])
        self.high_slider = Slider(
            high_axis,
            "High-pass (Hz)",
            0.0,
            min(15.0, FPS * 0.45),
            valinit=0.0,
            color="red",
        )
        self.low_slider = Slider(
            low_axis,
            "Low-pass (Hz)",
            1.0,
            min(37.5, FPS * 0.49),
            valinit=self.high_cut,
            color="green",
        )
        self.high_slider.on_changed(self.update_labels)
        self.low_slider.on_changed(self.update_labels)

        self.scatter_by_stud = {}
        self.text_by_stud = {}
        for stud_id in self.stud_ids:
            values = self.cache[stud_id]
            self.axis.plot(
                values["trail_x"],
                values["trail_y"],
                values["trail_z"],
                color="gray",
                alpha=0.2,
                linewidth=0.5,
            )
            scatter = self.axis.scatter(
                values["x_mean"],
                values["y_mean"],
                values["z_mean"],
                color="blue",
                s=80,
                picker=5,
            )
            scatter.stud_id = stud_id
            self.scatter_by_stud[stud_id] = scatter
            text = self.axis.text(
                values["x_mean"][0],
                values["y_mean"][0],
                values["z_mean"][0],
                "...",
                color="red",
                fontsize=8,
            )
            self.text_by_stud[stud_id] = text

        self.figure.canvas.mpl_connect("pick_event", self.on_pick)
        self.axis.set_title("Frequency analysis: click a stud for details")
        self.axis.set_xlabel("X")
        self.axis.set_ylabel("Y")
        self.axis.set_zlabel("Z")

        coordinates = self.data[["X", "Y", "Z"]].to_numpy(dtype=float)
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        center = (minimum + maximum) / 2.0
        half_range = max(float((maximum - minimum).max() / 2.0), 1e-6)
        self.axis.set_xlim(center[0] - half_range, center[0] + half_range)
        self.axis.set_ylim(center[1] - half_range, center[1] + half_range)
        self.axis.set_zlim(center[2] - half_range, center[2] + half_range)

    def update_labels(self, _: Optional[float]) -> None:
        self.low_cut = float(self.high_slider.val)
        self.high_cut = float(self.low_slider.val)
        if self.high_cut <= self.low_cut:
            self.high_cut = min(self.low_cut + 0.1, FPS * 0.49)

        color = "red" if self.low_cut > 1.5 else "blue"
        for stud_id in self.stud_ids:
            values = self.cache[stud_id]
            filtered = safe_filter(values["y"], self.low_cut, self.high_cut)
            frequencies, magnitude = spectrum(filtered)
            peak = dominant_frequency(frequencies, magnitude)
            self.text_by_stud[stud_id].set_text(f" {peak:.1f} Hz")
            self.scatter_by_stud[stud_id].set_color(color)
        self.figure.canvas.draw_idle()

    def on_pick(self, event) -> None:
        if hasattr(event.artist, "stud_id"):
            stud_id = event.artist.stud_id
            window = DetailWindow(
                self.cache[stud_id],
                self.low_cut,
                self.high_cut,
            )
            self.detail_windows.append(window)
            window.figure.show()


def main() -> None:
    data = load_data(INPUT_FILE)
    print(f"Loaded {len(data)} reconstructed points.")
    viewer = TireFrequencyViewer(data)
    plt.show()
    _ = viewer


if __name__ == "__main__":
    main()
