"""Detect bright tire studs inside ROI masks.

Save as:
    blob_detection.py

Expected ROI CSV columns:
    Frame_Name, Edge_Type, Point_ID, X, Y

Output CSV columns:
    Frame_Name, Blob_ID, X, Y, Diameter

Controls:
    S      save current frame
    A      process and save all remaining frames
    Space  skip current frame
    Q      quit
"""

from pathlib import Path
import csv
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ------------------------- User configuration -------------------------
INPUT_DIR = Path("data/raw/selected_pairs")
ROI_CSV_FILENAME = Path("data/intermediate/roi_edges.csv")
BLOB_OUTPUT_CSV = Path("data/intermediate/detections.csv")
WINDOW_NAME = "Blob Detector"

CURRENT_CAMERA = "camA"
PARAMS = {
    "camA": {
        "threshold": 19,
        "min_area": 14,
        "max_area": 2000,
        "min_circularity": 0.51,
        "min_convexity": 0.28,
        "min_inertia": 0.01,
    },
    "camB": {
        "threshold": 23,
        "min_area": 12,
        "max_area": 2000,
        "min_circularity": 0.47,
        "min_convexity": 0.40,
        "min_inertia": 0.01,
    },
}
# ----------------------------------------------------------------------


def noop(_: int) -> None:
    """Trackbar callback."""


def load_rois_from_csv(filename: Path) -> Dict[str, Dict[str, List[List[int]]]]:
    """Load edge points grouped by frame name and edge type."""
    if not filename.exists():
        raise FileNotFoundError(f"ROI file not found: {filename}")

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
            frame_name = row["Frame_Name"]
            edge_type = row["Edge_Type"]
            if edge_type not in {"Outer", "Inner"}:
                continue

            try:
                point = [int(float(row["X"])), int(float(row["Y"]))]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid ROI coordinate in row: {row}"
                ) from error

            if frame_name not in roi_data:
                roi_data[frame_name] = {"Outer": [], "Inner": []}
            roi_data[frame_name][edge_type].append(point)

    if not roi_data:
        raise ValueError(f"No ROI points were loaded from {filename}")
    return roi_data


def sorted_frame_names(roi_data: Dict[str, object]) -> List[str]:
    """Return frame names in their CSV order."""
    return list(roi_data.keys())


def make_roi_mask(
    image_shape: Tuple[int, int],
    outer_points: List[List[int]],
    inner_points: List[List[int]],
) -> np.ndarray:
    """Create a filled outer mask with the inner hole removed."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(outer_points) >= 3:
        outer = np.asarray(outer_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [outer], 255)

    if len(inner_points) >= 3:
        inner = np.asarray(inner_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [inner], 0)

    return mask


def camera_profile(frame_name: str) -> str:
    """Choose a camera profile from the filename, with a safe fallback."""
    name = frame_name.lower()
    if "camb" in name or "cam_b" in name or "camera2" in name:
        return "camB"
    return "camA"


def create_detector(parameters: Dict[str, float]) -> cv2.SimpleBlobDetector:
    """Create a SimpleBlobDetector using the selected camera profile."""
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.minThreshold = 0
    blob_params.maxThreshold = 255
    blob_params.thresholdStep = 10

    blob_params.filterByColor = True
    blob_params.blobColor = 255

    blob_params.filterByArea = True
    blob_params.minArea = max(1.0, float(parameters["min_area"]))
    blob_params.maxArea = max(
        blob_params.minArea + 1.0,
        float(parameters["max_area"]),
    )

    blob_params.filterByCircularity = True
    blob_params.minCircularity = float(
        np.clip(parameters["min_circularity"], 0.01, 1.0)
    )

    blob_params.filterByConvexity = True
    blob_params.minConvexity = float(
        np.clip(parameters["min_convexity"], 0.01, 1.0)
    )

    blob_params.filterByInertia = True
    blob_params.minInertiaRatio = float(
        np.clip(parameters["min_inertia"], 0.01, 1.0)
    )
    blob_params.maxInertiaRatio = 1.0

    return cv2.SimpleBlobDetector_create(blob_params)


def detect_blobs(
    frame: np.ndarray,
    mask: np.ndarray,
    parameters: Dict[str, float],
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Threshold the frame and detect blobs inside the ROI."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray,
        int(parameters["threshold"]),
        255,
        cv2.THRESH_BINARY,
    )
    blob_input = cv2.bitwise_and(binary, binary, mask=mask)
    detector = create_detector(parameters)
    keypoints = detector.detect(blob_input)
    keypoints.sort(key=lambda keypoint: (keypoint.pt[1], keypoint.pt[0]))
    return keypoints, blob_input


def save_blobs(
    writer: csv.writer,
    frame_name: str,
    keypoints: List[cv2.KeyPoint],
) -> None:
    """Write detected blob positions to the output CSV."""
    for blob_id, keypoint in enumerate(keypoints):
        writer.writerow(
            [
                frame_name,
                blob_id,
                round(float(keypoint.pt[0]), 2),
                round(float(keypoint.pt[1]), 2),
                round(float(keypoint.size), 2),
            ]
        )


def make_preview(
    frame: np.ndarray,
    blob_input: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    frame_name: str,
    profile: str,
    auto_mode: bool,
) -> np.ndarray:
    """Create a side-by-side detection preview."""
    detected = cv2.drawKeypoints(
        frame,
        keypoints,
        None,
        (0, 255, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    for keypoint in keypoints:
        center = tuple(np.round(keypoint.pt).astype(int))
        cv2.circle(detected, center, 2, (0, 0, 255), -1)

    mode = "AUTO" if auto_mode else "MANUAL"
    cv2.putText(
        detected,
        f"{frame_name} | {profile} | {mode} | Blobs: {len(keypoints)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    binary_bgr = cv2.cvtColor(blob_input, cv2.COLOR_GRAY2BGR)
    if detected.shape != binary_bgr.shape:
        binary_bgr = cv2.resize(
            binary_bgr,
            (detected.shape[1], detected.shape[0]),
        )
    return np.hstack((detected, binary_bgr))


def setup_trackbars(profile: str) -> None:
    """Create trackbars initialized with the selected profile."""
    values = PARAMS[profile]
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 700)
    cv2.createTrackbar(
        "Threshold", WINDOW_NAME, int(values["threshold"]), 255, noop
    )
    cv2.createTrackbar(
        "Min Area", WINDOW_NAME, int(values["min_area"]), 200, noop
    )
    cv2.createTrackbar(
        "Min Circ %",
        WINDOW_NAME,
        int(values["min_circularity"] * 100),
        100,
        noop,
    )
    cv2.createTrackbar(
        "Min Conv %",
        WINDOW_NAME,
        int(values["min_convexity"] * 100),
        100,
        noop,
    )
    cv2.createTrackbar(
        "Min Inertia %",
        WINDOW_NAME,
        int(values["min_inertia"] * 100),
        100,
        noop,
    )


def read_trackbar_parameters() -> Dict[str, float]:
    """Read current manual values from the OpenCV controls."""
    return {
        "threshold": cv2.getTrackbarPos("Threshold", WINDOW_NAME),
        "min_area": max(1, cv2.getTrackbarPos("Min Area", WINDOW_NAME)),
        "max_area": 2000,
        "min_circularity": max(
            0.01,
            cv2.getTrackbarPos("Min Circ %", WINDOW_NAME) / 100.0,
        ),
        "min_convexity": max(
            0.01,
            cv2.getTrackbarPos("Min Conv %", WINDOW_NAME) / 100.0,
        ),
        "min_inertia": max(
            0.01,
            cv2.getTrackbarPos("Min Inertia %", WINDOW_NAME) / 100.0,
        ),
    }


def set_profile_trackbars(profile: str) -> None:
    """Update controls when the inferred camera profile changes."""
    values = PARAMS[profile]
    cv2.setTrackbarPos("Threshold", WINDOW_NAME, int(values["threshold"]))
    cv2.setTrackbarPos("Min Area", WINDOW_NAME, int(values["min_area"]))
    cv2.setTrackbarPos(
        "Min Circ %", WINDOW_NAME, int(values["min_circularity"] * 100)
    )
    cv2.setTrackbarPos(
        "Min Conv %", WINDOW_NAME, int(values["min_convexity"] * 100)
    )
    cv2.setTrackbarPos(
        "Min Inertia %", WINDOW_NAME, int(values["min_inertia"] * 100)
    )


def main() -> None:
    roi_data = load_rois_from_csv(ROI_CSV_FILENAME)
    frame_names = sorted_frame_names(roi_data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Frame_Name", "Blob_ID", "X", "Y", "Diameter"])

        setup_trackbars(CURRENT_CAMERA)
        print(f"Loaded {len(frame_names)} frames.")
        print("Controls: S=save, A=auto-process, Space=skip, Q=quit")

        auto_mode = False
        frame_index = 0
        active_profile = CURRENT_CAMERA

        try:
            while frame_index < len(frame_names):
                frame_name = frame_names[frame_index]
                profile = camera_profile(frame_name)

                if profile != active_profile:
                    active_profile = profile
                    if not auto_mode:
                        set_profile_trackbars(profile)

                frame_path = INPUT_DIR / frame_name
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"Could not read image: {frame_path}")
                    frame_index += 1
                    continue

                roi = roi_data[frame_name]
                mask = make_roi_mask(
                    frame.shape[:2],
                    roi["Outer"],
                    roi["Inner"],
                )

                parameters = PARAMS[profile] if auto_mode else read_trackbar_parameters()
                keypoints, blob_input = detect_blobs(frame, mask, parameters)
                preview = make_preview(
                    frame,
                    blob_input,
                    keypoints,
                    frame_name,
                    profile,
                    auto_mode,
                )
                cv2.imshow(WINDOW_NAME, preview)

                if auto_mode:
                    save_blobs(writer, frame_name, keypoints)
                    output_file.flush()
                    print(f"Saved {frame_name}: {len(keypoints)} blobs")
                    frame_index += 1
                    cv2.waitKey(1)
                    continue

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    save_blobs(writer, frame_name, keypoints)
                    output_file.flush()
                    print(f"Saved {frame_name}: {len(keypoints)} blobs")
                    frame_index += 1
                elif key == ord("a"):
                    auto_mode = True
                    print("Automatic processing started.")
                elif key == 32:
                    print(f"Skipped {frame_name}")
                    frame_index += 1
        finally:
            cv2.destroyAllWindows()

    print(f"Blob detections saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
