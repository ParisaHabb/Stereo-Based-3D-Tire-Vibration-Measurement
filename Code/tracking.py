"""Detect and track tire studs in two camera views.

Save as:
    tracking.py

Input:
    Images in INPUT_DIR.
    ROI contour CSV from roi_edges.py.

Output:
    Frame_Name,Camera,Track_ID,X,Y,Skipped

The tracker uses a constant-velocity Kalman filter and Hungarian
assignment. Camera A and camera B are processed independently.
"""

from pathlib import Path
import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------- User configuration -------------------------
INPUT_DIR = Path("data/raw/selected_pairs")
ROI_CSV = Path("data/intermediate/roi_edges.csv")
DETECTION_CSV = Path("data/intermediate/detections.csv")
OUTPUT_CSV = Path("data/intermediate/cleaned_tracks.csv")
WINDOW_NAME = "Tracker"

MIN_LIFESPAN_PERCENT = 0.20
MAX_SKIPPED_FRAMES = 10

PARAMS = {
    "camA": {
        "threshold": 19,
        "min_area": 14,
        "max_area": 2000,
        "min_circularity": 0.51,
        "min_convexity": 0.28,
        "min_inertia": 0.01,
        "kalman_q": 30.0,
        "max_jump": 40.0,
    },
    "camB": {
        "threshold": 23,
        "min_area": 12,
        "max_area": 2000,
        "min_circularity": 0.47,
        "min_convexity": 0.40,
        "min_inertia": 0.01,
        "kalman_q": 30.0,
        "max_jump": 40.0,
    },
}
# ----------------------------------------------------------------------


def extract_frame_number(filename: str) -> int:
    """Extract the last number from a filename for numeric sorting."""
    values = re.findall(r"\d+", Path(filename).stem)
    return int(values[-1]) if values else -1


def load_roi_data(filename: Path) -> Dict[str, Dict[str, List[List[int]]]]:
    """Load ROI contours grouped by frame name."""
    if not filename.exists():
        raise FileNotFoundError(f"ROI CSV not found: {filename}")

    roi_data: Dict[str, Dict[str, List[List[int]]]] = {}
    with filename.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Frame_Name", "Edge_Type", "X", "Y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "ROI CSV is missing columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            edge_type = row["Edge_Type"]
            if edge_type not in {"Outer", "Inner"}:
                continue
            frame_name = row["Frame_Name"]
            if frame_name not in roi_data:
                roi_data[frame_name] = {"Outer": [], "Inner": []}
            roi_data[frame_name][edge_type].append(
                [int(float(row["X"])), int(float(row["Y"]))]
            )

    return roi_data


def load_detections(filename: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Load previously generated detections when available."""
    if not filename.exists():
        raise FileNotFoundError(f"Detection CSV not found: {filename}")

    detections: Dict[str, List[Tuple[float, float]]] = {}
    with filename.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Frame_Name", "X", "Y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Detection CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            frame_name = row["Frame_Name"]
            detections.setdefault(frame_name, []).append(
                (float(row["X"]), float(row["Y"]))
            )

    return detections


def collect_image_files(folder: Path) -> List[Path]:
    """Collect supported images in numeric filename order."""
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda path: (extract_frame_number(path.name), path.name))


@dataclass
class Track:
    track_id: int
    kalman: cv2.KalmanFilter
    last_position: np.ndarray
    skipped: int = 0
    history: List[Dict[str, object]] = field(default_factory=list)


class TrackerEngine:
    """Constant-velocity Kalman tracker with Hungarian assignment."""

    def __init__(self, kalman_q: float, max_jump: float):
        self.kalman_q = float(kalman_q)
        self.max_jump = float(max_jump)
        self.tracks: List[Track] = []
        self.next_id = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 0

    def _new_kalman(self, position: np.ndarray) -> cv2.KalmanFilter:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * (
            self.kalman_q / 1000.0
        )
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        state = np.array(
            [[position[0]], [position[1]], [0.0], [0.0]],
            dtype=np.float32,
        )
        kalman.statePre = state.copy()
        kalman.statePost = state.copy()
        return kalman

    def _create_track(
        self,
        position: np.ndarray,
        frame_name: str,
    ) -> Track:
        track = Track(
            track_id=self.next_id,
            kalman=self._new_kalman(position),
            last_position=position.copy(),
        )
        track.history.append(
            {
                "frame": frame_name,
                "x": float(position[0]),
                "y": float(position[1]),
                "skipped": 0,
            }
        )
        self.tracks.append(track)
        self.next_id += 1
        return track

    def process_frame(
        self,
        frame_name: str,
        detections: Sequence[Tuple[float, float]],
    ) -> List[Dict[str, object]]:
        """Update tracks using detections from one frame."""
        predictions: List[np.ndarray] = []
        for track in self.tracks:
            prediction = track.kalman.predict()
            predictions.append(
                np.array([prediction[0, 0], prediction[1, 0]], dtype=float)
            )

        detection_array = np.asarray(detections, dtype=float).reshape(-1, 2)
        assigned_tracks = set()
        assigned_detections = set()

        if predictions and len(detection_array) > 0:
            prediction_array = np.asarray(predictions, dtype=float)
            cost_matrix = np.linalg.norm(
                prediction_array[:, None, :] - detection_array[None, :, :],
                axis=2,
            )
            rows, columns = linear_sum_assignment(cost_matrix)

            for row, column in zip(rows, columns):
                if cost_matrix[row, column] <= self.max_jump:
                    track = self.tracks[row]
                    measurement = detection_array[column]
                    track.kalman.correct(
                        np.asarray(measurement, dtype=np.float32).reshape(2, 1)
                    )
                    track.last_position = measurement.copy()
                    track.skipped = 0
                    track.history.append(
                        {
                            "frame": frame_name,
                            "x": float(measurement[0]),
                            "y": float(measurement[1]),
                            "skipped": 0,
                        }
                    )
                    assigned_tracks.add(row)
                    assigned_detections.add(column)

        for index, track in enumerate(self.tracks):
            if index not in assigned_tracks:
                track.skipped += 1
                if index < len(predictions):
                    track.last_position = predictions[index]
                track.history.append(
                    {
                        "frame": frame_name,
                        "x": float(track.last_position[0]),
                        "y": float(track.last_position[1]),
                        "skipped": 1,
                    }
                )

        for detection_index, detection in enumerate(detection_array):
            if detection_index not in assigned_detections:
                self._create_track(detection, frame_name)

        self.tracks = [
            track
            for track in self.tracks
            if track.skipped <= MAX_SKIPPED_FRAMES
        ]

        snapshot = []
        for track in self.tracks:
            latest = track.history[-1]
            snapshot.append(
                {
                    "id": track.track_id,
                    "x": latest["x"],
                    "y": latest["y"],
                    "skipped": latest["skipped"],
                }
            )
        return snapshot

    def histories(self) -> Dict[int, List[Dict[str, object]]]:
        """Return track histories, including real and predicted samples."""
        return {
            track.track_id: track.history.copy()
            for track in self.tracks
        }


def filter_histories(
    histories: Dict[int, List[Dict[str, object]]],
    total_frames: int,
) -> Dict[int, List[Dict[str, object]]]:
    """Keep tracks that contain enough observed samples."""
    minimum_frames = max(1, int(np.ceil(total_frames * MIN_LIFESPAN_PERCENT)))
    cleaned = {}

    for track_id, history in histories.items():
        observed_count = sum(
            1 for item in history if int(item["skipped"]) == 0
        )
        if observed_count >= minimum_frames:
            cleaned[track_id] = [
                item for item in history if int(item["skipped"]) == 0
            ]

    print(
        f"Kept {len(cleaned)} tracks from {len(histories)}. "
        f"Minimum observed samples: {minimum_frames}."
    )
    return cleaned


def infer_camera(frame_name: str) -> str:
    """Infer the camera label from a frame filename."""
    name = frame_name.lower()
    return "camB" if any(token in name for token in ("camb", "cam_b", "camera2")) else "camA"


def process_camera(
    image_files: List[Path],
    detections: Dict[str, List[Tuple[float, float]]],
    camera_name: str,
) -> Dict[int, List[Dict[str, object]]]:
    """Track one camera sequence."""
    if not image_files:
        raise ValueError(f"No images found for {camera_name}.")

    params = PARAMS[camera_name]
    tracker = TrackerEngine(params["kalman_q"], params["max_jump"])

    for index, image_path in enumerate(image_files, start=1):
        frame_name = image_path.name
        frame_detections = detections.get(frame_name, [])
        tracker.process_frame(frame_name, frame_detections)
        if index % 50 == 0 or index == len(image_files):
            print(f"{camera_name}: processed {index}/{len(image_files)} frames")

    return filter_histories(tracker.histories(), len(image_files))


def save_tracks(
    output_file: Path,
    camera_histories: Dict[str, Dict[int, List[Dict[str, object]]]],
) -> None:
    """Save cleaned tracks in a stable schema."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame_Name", "Camera", "Track_ID", "X", "Y"])

        for camera_name, histories in camera_histories.items():
            for track_id, history in histories.items():
                for item in history:
                    writer.writerow(
                        [
                            item["frame"],
                            camera_name,
                            track_id,
                            round(float(item["x"]), 3),
                            round(float(item["y"]), 3),
                        ]
                    )


def main() -> None:
    detections = load_detections(DETECTION_CSV)
    image_files = collect_image_files(INPUT_DIR)

    files_by_camera = {"camA": [], "camB": []}
    for image_path in image_files:
        files_by_camera[infer_camera(image_path.name)].append(image_path)

    camera_histories = {}
    for camera_name in ("camA", "camB"):
        camera_histories[camera_name] = process_camera(
            files_by_camera[camera_name],
            detections,
            camera_name,
        )

    save_tracks(OUTPUT_CSV, camera_histories)
    total_points = sum(
        len(history)
        for camera_data in camera_histories.values()
        for history in camera_data.values()
    )
    print(f"Saved {total_points} cleaned track points to {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
