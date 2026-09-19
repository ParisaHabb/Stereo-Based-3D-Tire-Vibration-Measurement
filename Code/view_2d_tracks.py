"""Interactive viewer for 2D tracked tire studs.

Save as:
    view_2d_tracks.py

Controls:
    T      toggle camera
    Space  play/pause
    Q      quit
"""

from pathlib import Path
import csv
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ------------------------- User configuration -------------------------
INPUT_DIR = Path("data/raw/selected_pairs")
TRACKS_CSV = Path("data/intermediate/cleaned_tracks.csv")
WINDOW_NAME = "2D Track Viewer"
# ----------------------------------------------------------------------


def frame_number(filename: str) -> int:
    """Extract the last integer from a frame filename."""
    values = re.findall(r"\d+", Path(filename).stem)
    return int(values[-1]) if values else -1


def load_tracks(
    filename: Path,
) -> Tuple[
    Dict[str, Dict[str, List[Dict[str, float]]]],
    Dict[str, Dict[int, List[Tuple[float, float]]]],
]:
    """Load per-frame positions and complete paths from the track CSV."""
    if not filename.exists():
        raise FileNotFoundError(f"Track CSV not found: {filename}")

    frame_data: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "camA": {},
        "camB": {},
    }
    full_paths: Dict[str, Dict[int, List[Tuple[float, float]]]] = {
        "camA": {},
        "camB": {},
    }

    with filename.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Frame_Name", "Camera", "Track_ID", "X", "Y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Track CSV is missing columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            camera = row["Camera"]
            if camera not in frame_data:
                frame_data[camera] = {}
                full_paths[camera] = {}

            filename_value = row["Frame_Name"]
            track_id = int(row["Track_ID"])
            x = float(row["X"])
            y = float(row["Y"])

            frame_data[camera].setdefault(filename_value, []).append(
                {"id": track_id, "x": x, "y": y}
            )
            full_paths[camera].setdefault(track_id, []).append((x, y))

    if not any(frame_data.values()):
        raise ValueError("No track points were found in the CSV file.")

    return frame_data, full_paths


def collect_camera_frames(input_dir: Path) -> Dict[str, List[Path]]:
    """Collect images from camA/camB subfolders or a shared folder."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    result: Dict[str, List[Path]] = {"camA": [], "camB": []}

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    subfolders = {
        "camA": input_dir / "camA",
        "camB": input_dir / "camB",
    }
    has_subfolders = all(path.is_dir() for path in subfolders.values())

    if has_subfolders:
        for camera, folder in subfolders.items():
            result[camera] = sorted(
                [
                    path
                    for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in extensions
                ],
                key=lambda path: (frame_number(path.name), path.name),
            )
        return result

    shared_files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    for path in shared_files:
        name = path.name.lower()
        camera = "camB" if any(
            token in name for token in ("camb", "cam_b", "camera2")
        ) else "camA"
        result[camera].append(path)

    for camera in result:
        result[camera].sort(
            key=lambda path: (frame_number(path.name), path.name)
        )
    return result


def make_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a deterministic BGR color for a track ID."""
    rng = np.random.default_rng(track_id)
    color = rng.integers(50, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracks(
    image: np.ndarray,
    camera: str,
    filename: str,
    frame_index: int,
    total_frames: int,
    frame_data: Dict[str, Dict[str, List[Dict[str, float]]]],
    full_paths: Dict[str, Dict[int, List[Tuple[float, float]]]],
) -> np.ndarray:
    """Draw full paths and current track positions."""
    output = image.copy()

    for track_id, points in full_paths.get(camera, {}).items():
        if len(points) < 2:
            continue
        points_array = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            output,
            [points_array],
            False,
            make_color(track_id),
            1,
            cv2.LINE_AA,
        )

    for track in frame_data.get(camera, {}).get(filename, []):
        x = int(round(track["x"]))
        y = int(round(track["y"]))
        track_id = int(track["id"])
        color = make_color(track_id)

        cv2.circle(output, (x, y), 6, color, -1)
        cv2.circle(output, (x, y), 8, (255, 255, 255), 1)
        cv2.putText(
            output,
            str(track_id),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"Camera: {camera.upper()}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Frame: {frame_index + 1}/{total_frames}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"File: {filename}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    frame_data, full_paths = load_tracks(TRACKS_CSV)
    camera_files = collect_camera_frames(INPUT_DIR)

    for camera in ("camA", "camB"):
        if not camera_files[camera]:
            raise ValueError(f"No images found for {camera}.")

    current_camera = "camA"
    current_files = camera_files[current_camera]
    frame_index = 0
    playing = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 800)
    cv2.createTrackbar(
        "Frame",
        WINDOW_NAME,
        0,
        max(0, len(current_files) - 1),
        lambda value: None,
    )

    print("Controls: T=toggle camera, Space=play/pause, Q=quit")

    try:
        while True:
            frame_index = cv2.getTrackbarPos("Frame", WINDOW_NAME)
            frame_index = min(frame_index, len(current_files) - 1)
            image_path = current_files[frame_index]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")

            preview = draw_tracks(
                image,
                current_camera,
                image_path.name,
                frame_index,
                len(current_files),
                frame_data,
                full_paths,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(30 if playing else 10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                current_camera = "camB" if current_camera == "camA" else "camA"
                current_files = camera_files[current_camera]
                cv2.setTrackbarMax(
                    "Frame",
                    WINDOW_NAME,
                    max(0, len(current_files) - 1),
                )
                cv2.setTrackbarPos("Frame", WINDOW_NAME, 0)
                print(f"Switched to {current_camera}")
            elif key == 32:
                playing = not playing

            if playing:
                next_index = frame_index + 1
                if next_index >= len(current_files):
                    playing = False
                else:
                    cv2.setTrackbarPos("Frame", WINDOW_NAME, next_index)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
