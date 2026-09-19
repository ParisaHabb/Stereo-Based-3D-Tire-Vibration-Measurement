"""Extract and visualize vibration mode shapes from 3D trajectories.

Save as:
    modal_shapes.py

Input:
    data/processed/final_3d_paths.csv

Outputs:
    data/processed/mode_shapes.csv
    data/processed/modal_frequencies.csv
"""

from pathlib import Path
import re
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


INPUT_FILE = Path("data/processed/final_3d_paths.csv")
OUTPUT_FILE = Path("data/processed/mode_shapes.csv")
FREQUENCY_OUTPUT_FILE = Path("data/processed/modal_frequencies.csv")
FPS = 75.0
COORDINATE_COLUMN = "Y"
MIN_FREQUENCY = 0.5
PEAK_RELATIVE_THRESHOLD = 0.10
NUMBER_OF_MODES = 2
DEFORM_SCALE = 80.0


def extract_frame_number(value: object) -> int:
    """Extract the last integer from a frame identifier."""
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[-1]) if numbers else -1


def load_data(filename: Path) -> pd.DataFrame:
    """Load and validate the reconstructed paths."""
    if not filename.exists():
        raise FileNotFoundError(f"3D trajectory file not found: {filename}")

    data = pd.read_csv(filename)
    required = {"Frame_ID", "Stud_ID", "X", "Y", "Z"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Input file is missing columns: " + ", ".join(sorted(missing))
        )

    data = data.copy()
    for column in ("X", "Y", "Z"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Frame_ID", "Stud_ID", "X", "Y", "Z"])
    data["Frame_Num"] = data["Frame_ID"].map(extract_frame_number)
    data = data.sort_values(["Frame_Num", "Stud_ID"])

    if data.empty:
        raise ValueError("No valid trajectory points were found.")
    return data


def build_uniform_matrix(
    data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a frame-by-stud matrix and interpolate missing samples."""
    pivot = data.pivot_table(
        index="Frame_Num",
        columns="Stud_ID",
        values=COORDINATE_COLUMN,
        aggfunc="mean",
    ).sort_index()

    if pivot.empty:
        raise ValueError("Could not build the frame-by-stud matrix.")

    full_frames = np.arange(
        int(pivot.index.min()),
        int(pivot.index.max()) + 1,
        dtype=int,
    )
    pivot = pivot.reindex(full_frames)
    pivot = pivot.interpolate(method="linear", axis=0, limit_direction="both")
    pivot = pivot.ffill().bfill()

    matrix = pivot.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("The trajectory matrix still contains invalid values.")

    return (
        matrix,
        full_frames,
        pivot.columns.to_numpy(),
        data.groupby("Stud_ID")[["X", "Y", "Z"]].mean().reindex(pivot.columns).to_numpy(),
    )


def find_modal_peaks(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
) -> List[int]:
    """Find distinct modal peaks above the selected frequency threshold."""
    if magnitude.size == 0 or np.max(magnitude) <= 0:
        return []

    peaks, properties = find_peaks(
        magnitude,
        height=np.max(magnitude) * PEAK_RELATIVE_THRESHOLD,
    )
    valid = [
        int(index)
        for index in peaks
        if frequencies[index] >= MIN_FREQUENCY
    ]
    return sorted(valid, key=lambda index: magnitude[index], reverse=True)


def signed_mode(complex_mode: np.ndarray) -> np.ndarray:
    """Remove global phase and return a normalized real mode shape."""
    values = np.asarray(complex_mode, dtype=complex)
    reference = int(np.argmax(np.abs(values)))
    if abs(values[reference]) < 1e-12:
        return np.zeros(values.shape, dtype=float)

    rotated = values * np.exp(-1j * np.angle(values[reference]))
    shape = np.real(rotated)
    scale = np.max(np.abs(shape))
    return shape / scale if scale > 1e-12 else np.zeros(shape.shape)


def calculate_modes(
    matrix: np.ndarray,
    frame_numbers: np.ndarray,
    stud_ids: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, np.ndarray]]:
    """Calculate spectra, modal frequencies and mode-shape vectors."""
    detrended = matrix - np.mean(matrix, axis=0, keepdims=True)
    window = np.hanning(len(detrended))[:, None]
    transformed = rfft(detrended * window, axis=0)
    frequencies = rfftfreq(len(detrended), d=1.0 / FPS)
    global_magnitude = np.mean(np.abs(transformed), axis=1)

    peak_indices = find_modal_peaks(global_magnitude, frequencies)
    if len(peak_indices) < NUMBER_OF_MODES:
        raise RuntimeError(
            f"Only {len(peak_indices)} valid modal peaks were found. "
            "Reduce PEAK_RELATIVE_THRESHOLD or MIN_FREQUENCY."
        )

    selected = peak_indices[:NUMBER_OF_MODES]
    modes: Dict[int, np.ndarray] = {}
    modal_rows = []
    output = {"Stud_ID": stud_ids}

    for mode_number, peak_index in enumerate(selected, start=1):
        complex_vector = transformed[peak_index, :]
        normalized_complex = complex_vector / max(
            np.max(np.abs(complex_vector)),
            1e-12,
        )
        shape = signed_mode(normalized_complex)
        modes[mode_number] = shape
        output[f"mode{mode_number}_mag"] = np.abs(normalized_complex)
        output[f"mode{mode_number}_phase_rad"] = np.angle(normalized_complex)
        output[f"mode{mode_number}_shape_signed"] = shape
        modal_rows.append(
            {
                "Mode": mode_number,
                "Frequency_Hz": float(frequencies[peak_index]),
                "FFT_Index": int(peak_index),
                "Global_Magnitude": float(global_magnitude[peak_index]),
            }
        )

    output["Frame_Count"] = len(frame_numbers)
    return (
        pd.DataFrame(output),
        pd.DataFrame(modal_rows),
        modes,
    )


def set_equal_axes(axis, positions: np.ndarray, deformed: np.ndarray) -> None:
    """Use one scale for all coordinates in a 3D mode plot."""
    all_points = np.vstack((positions, deformed))
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    center = (minimum + maximum) / 2.0
    half_range = max(float(np.max(maximum - minimum) / 2.0), 1e-6)
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)


def plot_modes(
    positions: np.ndarray,
    modal_frequencies: pd.DataFrame,
    modes: Dict[int, np.ndarray],
    stud_ids: np.ndarray,
) -> None:
    """Show 3D deformations and circumferential mode-shape plots."""
    mode_count = len(modes)
    figure = plt.figure(figsize=(7 * mode_count, 6))

    for mode_number, shape in modes.items():
        deformed = positions.copy()
        deformed[:, 1] += DEFORM_SCALE * shape
        axis = figure.add_subplot(1, mode_count, mode_number, projection="3d")

        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color="lightgray",
            s=20,
            alpha=0.5,
            label="Undeformed",
        )
        colors = np.where(shape >= 0, "red", "blue")
        sizes = 20 + 80 * np.abs(shape)
        axis.scatter(
            deformed[:, 0],
            deformed[:, 1],
            deformed[:, 2],
            color=colors,
            s=sizes,
            alpha=0.9,
            label="Mode shape",
        )

        for original, displaced in zip(positions, deformed):
            axis.plot(
                [original[0], displaced[0]],
                [original[1], displaced[1]],
                [original[2], displaced[2]],
                color="black",
                alpha=0.3,
                linewidth=0.8,
            )

        frequency = modal_frequencies.loc[
            modal_frequencies["Mode"] == mode_number,
            "Frequency_Hz",
        ].iloc[0]
        axis.set_title(f"Mode {mode_number} @ {frequency:.2f} Hz")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.legend(fontsize=8)
        set_equal_axes(axis, positions, deformed)

    figure.tight_layout()

    theta = np.arctan2(
        positions[:, 2] - np.mean(positions[:, 2]),
        positions[:, 0] - np.mean(positions[:, 0]),
    )
    order = np.argsort(theta)
    shape_figure, shape_axis = plt.subplots(figsize=(10, 4))
    for mode_number, shape in modes.items():
        frequency = modal_frequencies.loc[
            modal_frequencies["Mode"] == mode_number,
            "Frequency_Hz",
        ].iloc[0]
        shape_axis.plot(
            theta[order],
            shape[order],
            "o-",
            label=f"Mode {mode_number} ({frequency:.2f} Hz)",
        )
    shape_axis.axhline(0, color="black", linewidth=0.8)
    shape_axis.set_xlabel("Circumferential angle (rad)")
    shape_axis.set_ylabel("Normalized signed displacement")
    shape_axis.set_title("Mode shapes along tire circumference")
    shape_axis.grid(True, alpha=0.3)
    shape_axis.legend()
    shape_figure.tight_layout()

    amplitude_figure, amplitude_axis = plt.subplots(figsize=(10, 4))
    for mode_number, shape in modes.items():
        frequency = modal_frequencies.loc[
            modal_frequencies["Mode"] == mode_number,
            "Frequency_Hz",
        ].iloc[0]
        amplitude_axis.plot(
            stud_ids,
            np.abs(shape),
            "o-",
            label=f"Mode {mode_number} ({frequency:.2f} Hz)",
        )
    amplitude_axis.set_xlabel("Stud ID")
    amplitude_axis.set_ylabel("Normalized amplitude")
    amplitude_axis.set_title("Mode-shape amplitude by stud")
    amplitude_axis.grid(True, alpha=0.3)
    amplitude_axis.legend()
    amplitude_figure.tight_layout()


def main() -> None:
    data = load_data(INPUT_FILE)
    matrix, frame_numbers, stud_ids, positions = build_uniform_matrix(data)
    print(f"Matrix shape: {matrix.shape} (frames x studs)")

    mode_table, frequency_table, modes = calculate_modes(
        matrix,
        frame_numbers,
        stud_ids,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode_table.to_csv(OUTPUT_FILE, index=False)
    frequency_table.to_csv(FREQUENCY_OUTPUT_FILE, index=False)

    print(f"Mode shapes saved to: {OUTPUT_FILE.resolve()}")
    print(f"Frequencies saved to: {FREQUENCY_OUTPUT_FILE.resolve()}")
    print(frequency_table.to_string(index=False))

    plot_modes(positions, frequency_table, modes, stud_ids)
    plt.show()


if __name__ == "__main__":
    main()
