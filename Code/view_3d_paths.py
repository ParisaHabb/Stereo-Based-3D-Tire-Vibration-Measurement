"""Interactive 3D viewer for reconstructed tire-stud trajectories.

Save as:
    view_3d_paths.py

Input CSV columns:
    Frame_ID, Stud_ID, X, Y, Z
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/final_3d_paths.csv")


def frame_number(value: object) -> int:
    """Extract a numeric value from a frame identifier."""
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[-1]) if numbers else -1


def load_trajectories(filename: Path) -> pd.DataFrame:
    """Load and validate reconstructed 3D trajectories."""
    if not filename.exists():
        raise FileNotFoundError(f"3D trajectory file not found: {filename}")

    data = pd.read_csv(filename)
    required = {"Frame_ID", "Stud_ID", "X", "Y", "Z"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "3D CSV is missing columns: " + ", ".join(sorted(missing))
        )

    data = data.copy()
    for column in ("X", "Y", "Z"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Frame_ID", "Stud_ID", "X", "Y", "Z"])

    if data.empty:
        raise ValueError("The 3D trajectory file contains no valid points.")

    data["Frame_Num"] = data["Frame_ID"].map(frame_number)
    data = data.sort_values(["Frame_Num", "Stud_ID"]).reset_index(drop=True)
    return data


def set_equal_axes(ax, data: pd.DataFrame) -> None:
    """Set equal-looking limits on all three axes."""
    coordinates = data[["X", "Y", "Z"]].to_numpy(dtype=float)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center = (minimum + maximum) / 2.0
    half_range = max(float((maximum - minimum).max() / 2.0), 1e-6)

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def main() -> None:
    data = load_trajectories(INPUT_FILE)
    frame_ids = data["Frame_ID"].drop_duplicates().tolist()
    stud_ids = data["Stud_ID"].drop_duplicates().tolist()

    if not frame_ids:
        raise ValueError("No frame identifiers were found.")

    print(f"Loaded {len(frame_ids)} frames and {len(stud_ids)} studs.")

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.24)

    for stud_id in stud_ids:
        track = data[data["Stud_ID"] == stud_id]
        axis.plot(
            track["X"].to_numpy(),
            track["Y"].to_numpy(),
            track["Z"].to_numpy(),
            color="gray",
            alpha=0.25,
            linewidth=0.6,
        )

    first_frame = data[data["Frame_ID"] == frame_ids[0]]
    scatter = axis.scatter(
        first_frame["X"].to_numpy(),
        first_frame["Y"].to_numpy(),
        first_frame["Z"].to_numpy(),
        color="red",
        s=50,
        depthshade=False,
        label="Studs",
    )

    set_equal_axes(axis, data)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(f"3D Playback: Frame {frame_ids[0]}")
    axis.legend()

    slider_axis = figure.add_axes([0.2, 0.09, 0.65, 0.035])
    slider = Slider(
        slider_axis,
        "Frame",
        0,
        max(0, len(frame_ids) - 1),
        valinit=0,
        valstep=1,
    )

    def update(value: float) -> None:
        index = int(value)
        frame_id = frame_ids[index]
        frame_data = data[data["Frame_ID"] == frame_id]
        scatter._offsets3d = (
            frame_data["X"].to_numpy(),
            frame_data["Y"].to_numpy(),
            frame_data["Z"].to_numpy(),
        )
        axis.set_title(f"3D Playback: Frame {frame_id}")
        figure.canvas.draw_idle()

    slider.on_changed(update)
    print("Viewer ready. Use the slider to inspect 3D motion.")
    plt.show()


if __name__ == "__main__":
    main()
