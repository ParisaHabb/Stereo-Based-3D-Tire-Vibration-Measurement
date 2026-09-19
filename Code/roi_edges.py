"""Detect and save inner/outer tire contours from image frames.

Save as:
    roi_edges.py

Controls:
    S      save current frame and continue
    A      automatically save all remaining frames
    Space  skip current frame
    Q      quit
"""

from pathlib import Path
import csv
import re
from typing import Optional, Tuple

import cv2
import numpy as np


# ------------------------- User configuration -------------------------
INPUT_DIR = Path("data/raw/selected_pairs")
OUTPUT_CSV = Path("data/intermediate/roi_edges.csv")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
START_FRAME: Optional[int] = None
END_FRAME: Optional[int] = None

WINDOW_NAME = "ROI Edge Selector"
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
INITIAL_THRESHOLD = 60
MIN_OUTER_AREA = 5000.0
MIN_INNER_AREA = 1000.0
# ----------------------------------------------------------------------


def extract_frame_number(filename: str) -> Optional[int]:
    """Extract the last numeric group from a filename."""
    numbers = re.findall(r"\d+", Path(filename).stem)
    return int(numbers[-1]) if numbers else None


def collect_target_files() -> list[Path]:
    """Collect and optionally filter input images."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    files = sorted(
        (
            path
            for path in INPUT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: (
            extract_frame_number(path.name)
            if extract_frame_number(path.name) is not None
            else -1,
            path.name.lower(),
        ),
    )

    if START_FRAME is not None:
        files = [
            path
            for path in files
            if extract_frame_number(path.name) is not None
            and extract_frame_number(path.name) >= START_FRAME
        ]
    if END_FRAME is not None:
        files = [
            path
            for path in files
            if extract_frame_number(path.name) is not None
            and extract_frame_number(path.name) <= END_FRAME
        ]

    return files


def find_tire_contours(
    image: np.ndarray,
    threshold: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Find the largest valid outer contour and a likely inner contour."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, binary = cv2.threshold(
        blurred,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None, None, binary

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_INNER_AREA
    ]
    if not valid_contours:
        return None, None, binary

    outer = max(valid_contours, key=cv2.contourArea)
    outer_area = cv2.contourArea(outer)
    if outer_area < MIN_OUTER_AREA:
        outer = None

    inner = None
    if outer is not None and hierarchy is not None:
        outer_index = contours.index(outer)
        child_index = hierarchy[0][outer_index][2]
        if child_index >= 0:
            candidate = contours[child_index]
            if cv2.contourArea(candidate) >= MIN_INNER_AREA:
                inner = candidate

    if inner is None:
        candidates = [
            contour
            for contour in valid_contours
            if outer is None or not np.array_equal(contour, outer)
        ]
        if candidates:
            candidate = max(candidates, key=cv2.contourArea)
            if cv2.contourArea(candidate) < outer_area:
                inner = candidate

    return outer, inner, binary


def draw_preview(
    image: np.ndarray,
    filename: str,
    outer: Optional[np.ndarray],
    inner: Optional[np.ndarray],
    auto_save: bool,
) -> np.ndarray:
    """Draw detected contours and status information."""
    display = image.copy()

    if outer is not None:
        cv2.drawContours(display, [outer], -1, (0, 255, 0), 2)
    if inner is not None:
        cv2.drawContours(display, [inner], -1, (0, 0, 255), 2)

    status = "AUTO-SAVE ON" if auto_save else "Press S to save"
    cv2.putText(
        display,
        f"{filename} | {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def append_contours(
    csv_writer: csv.writer,
    filename: str,
    outer: Optional[np.ndarray],
    inner: Optional[np.ndarray],
) -> None:
    """Write contour points using a stable CSV schema."""
    for edge_type, contour in (("Outer", outer), ("Inner", inner)):
        if contour is None:
            continue
        for point_id, point in enumerate(contour.reshape(-1, 2)):
            x, y = point
            csv_writer.writerow([filename, edge_type, point_id, int(x), int(y)])


def main() -> None:
    files = collect_target_files()
    if not files:
        raise FileNotFoundError("No input images matched the selected settings.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame_Name", "Edge_Type", "Point_ID", "X", "Y"])

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.createTrackbar(
            "Threshold",
            WINDOW_NAME,
            INITIAL_THRESHOLD,
            255,
            lambda value: None,
        )

        auto_save = False
        file_index = 0
        print(f"Found {len(files)} images.")
        print("Controls: S=save, A=auto-save, Space=skip, Q=quit")

        try:
            while file_index < len(files):
                image_path = files[file_index]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Could not read: {image_path}")
                    file_index += 1
                    continue

                threshold = cv2.getTrackbarPos("Threshold", WINDOW_NAME)
                outer, inner, _ = find_tire_contours(image, threshold)
                preview = draw_preview(
                    image,
                    image_path.name,
                    outer,
                    inner,
                    auto_save,
                )
                cv2.imshow(WINDOW_NAME, preview)

                if auto_save:
                    append_contours(writer, image_path.name, outer, inner)
                    csv_file.flush()
                    print(f"Saved: {image_path.name}")
                    file_index += 1
                    cv2.waitKey(1)
                    continue

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    append_contours(writer, image_path.name, outer, inner)
                    csv_file.flush()
                    print(f"Saved: {image_path.name}")
                    file_index += 1
                elif key == ord("a"):
                    auto_save = True
                elif key == 32:
                    print(f"Skipped: {image_path.name}")
                    file_index += 1
        finally:
            cv2.destroyAllWindows()

    print(f"Saved ROI edges to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
