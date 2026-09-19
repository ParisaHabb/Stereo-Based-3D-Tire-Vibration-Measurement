"""Stereo matching and 3D reconstruction of tracked tire studs.

Save as:
    stereo_reconstruction.py

Input track CSV columns:
    Frame_Name, Camera, Track_ID, X, Y

Calibration NPZ must contain at least:
    K1, K2, R, T

Preferred calibration files also contain P1 and P2. Coordinates are
triangulated in the same physical unit used for the calibration translation.
"""

from pathlib import Path
from collections import Counter
import csv
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ------------------------- User configuration -------------------------
TRACKS_CSV = Path("data/intermediate/cleaned_tracks.csv")
CALIB_NPZ = Path("data/calibration/stereo_calibration.npz")
OUTPUT_MATCHES_2D = Path("data/processed/stereo_matches_2d.csv")
OUTPUT_3D = Path("data/processed/final_3d_paths.csv")

EPIPOLAR_THRESHOLD = 15.0
MIN_MATCH_FRACTION = 0.05
Y_SHIFT = 0.0
# ----------------------------------------------------------------------


def extract_frame_number(value: str) -> int:
    """Extract a numeric frame index for stable sorting."""
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[-1]) if numbers else -1


def load_calibration(filename: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or construct F, P1 and P2 from a calibration NPZ file."""
    if not filename.exists():
        raise FileNotFoundError(f"Calibration file not found: {filename}")

    with np.load(filename) as calibration:
        required = {"K1", "K2", "R", "T"}
        missing = required - set(calibration.files)
        if missing:
            raise ValueError(
                "Calibration file is missing: " + ", ".join(sorted(missing))
            )

        k1 = np.asarray(calibration["K1"], dtype=float)
        k2 = np.asarray(calibration["K2"], dtype=float)
        rotation = np.asarray(calibration["R"], dtype=float)
        translation = np.asarray(calibration["T"], dtype=float).reshape(3, 1)

        if "P1" in calibration.files:
            p1 = np.asarray(calibration["P1"], dtype=float)
        else:
            p1 = k1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

        if "P2" in calibration.files:
            p2 = np.asarray(calibration["P2"], dtype=float)
        else:
            p2 = k2 @ np.hstack((rotation, translation))

        if "F" in calibration.files:
            fundamental = np.asarray(calibration["F"], dtype=float)
        else:
            skew_t = np.array(
                [
                    [0.0, -translation[2, 0], translation[1, 0]],
                    [translation[2, 0], 0.0, -translation[0, 0]],
                    [-translation[1, 0], translation[0, 0], 0.0],
                ]
            )
            fundamental = np.linalg.inv(k2).T @ skew_t @ rotation @ np.linalg.inv(k1)

    scale = np.linalg.norm(fundamental)
    if scale > 0:
        fundamental = fundamental / scale

    return fundamental, p1, p2


def load_tracks(filename: Path) -> pd.DataFrame:
    """Load tracks and normalize older ID column names."""
    if not filename.exists():
        raise FileNotFoundError(f"Track CSV not found: {filename}")

    tracks = pd.read_csv(filename)
    if "Track_ID" not in tracks.columns and "ID" in tracks.columns:
        tracks = tracks.rename(columns={"ID": "Track_ID"})

    required = {"Frame_Name", "Camera", "Track_ID", "X", "Y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(
            "Track CSV is missing: " + ", ".join(sorted(missing))
        )

    tracks = tracks.copy()
    tracks["Camera"] = tracks["Camera"].astype(str)
    tracks["Track_ID"] = pd.to_numeric(tracks["Track_ID"], errors="raise").astype(int)
    tracks["X"] = pd.to_numeric(tracks["X"], errors="raise")
    tracks["Y"] = pd.to_numeric(tracks["Y"], errors="raise")
    tracks["Pair_ID"] = tracks["Frame_Name"].astype(str)
    return tracks


def epipolar_cost(
    points_a: np.ndarray,
    points_b: np.ndarray,
    fundamental: np.ndarray,
    y_shift: float = 0.0,
) -> np.ndarray:
    """Calculate point-to-epipolar-line distances."""
    shifted_b = points_b.copy()
    shifted_b[:, 1] += y_shift

    points_a_h = np.column_stack((points_a, np.ones(len(points_a))))
    points_b_h = np.column_stack((shifted_b, np.ones(len(shifted_b))))
    lines_b = (fundamental @ points_a_h.T).T
    numerators = np.abs(lines_b @ points_b_h.T)
    denominators = np.sqrt(lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2)
    denominators = np.maximum(denominators, 1e-12)
    return numerators / denominators[:, None]


def match_frame(
    frame: pd.DataFrame,
    fundamental: np.ndarray,
    threshold: float,
    y_shift: float,
) -> List[Dict[str, float]]:
    """Match camera-A and camera-B tracks in one synchronized frame."""
    cam_a = frame[frame["Camera"].str.lower() == "cama"]
    cam_b = frame[frame["Camera"].str.lower() == "camb"]
    if cam_a.empty or cam_b.empty:
        return []

    points_a = cam_a[["X", "Y"]].to_numpy(dtype=float)
    points_b = cam_b[["X", "Y"]].to_numpy(dtype=float)
    costs = epipolar_cost(points_a, points_b, fundamental, y_shift)
    rows, columns = linear_sum_assignment(costs)

    matches = []
    for row, column in zip(rows, columns):
        distance = float(costs[row, column])
        if distance <= threshold:
            matches.append(
                {
                    "id_a": int(cam_a.iloc[row]["Track_ID"]),
                    "x_a": float(cam_a.iloc[row]["X"]),
                    "y_a": float(cam_a.iloc[row]["Y"]),
                    "id_b": int(cam_b.iloc[column]["Track_ID"]),
                    "x_b": float(cam_b.iloc[column]["X"]),
                    "y_b": float(cam_b.iloc[column]["Y"]),
                    "epipolar_distance": distance,
                }
            )
    return matches


def build_consensus(
    frame_matches: Dict[str, List[Dict[str, float]]],
    total_frames: int,
    minimum_fraction: float,
) -> Dict[int, int]:
    """Choose one-to-one track-ID links supported across multiple frames."""
    votes: Counter = Counter()
    for matches in frame_matches.values():
        for match in matches:
            votes[(int(match["id_a"]), int(match["id_b"]))] += 1

    minimum_votes = max(1, int(np.ceil(total_frames * minimum_fraction)))
    mapping: Dict[int, int] = {}
    used_b = set()

    for (id_a, id_b), count in votes.most_common():
        if count < minimum_votes:
            continue
        if id_a not in mapping and id_b not in used_b:
            mapping[id_a] = id_b
            used_b.add(id_b)

    return mapping


def triangulate(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Triangulate one corresponding image-point pair."""
    points_a = np.asarray(point_a, dtype=float).reshape(2, 1)
    points_b = np.asarray(point_b, dtype=float).reshape(2, 1)
    homogeneous = cv2.triangulatePoints(p1, p2, points_a, points_b)
    w = float(homogeneous[3, 0])
    if abs(w) < 1e-12:
        raise ValueError("Triangulation produced a point at infinity.")
    return (homogeneous[:3, 0] / w).astype(float)


def main() -> None:
    tracks = load_tracks(TRACKS_CSV)
    fundamental, p1, p2 = load_calibration(CALIB_NPZ)

    frame_names = sorted(
        tracks["Pair_ID"].unique(),
        key=extract_frame_number,
    )
    frame_matches: Dict[str, List[Dict[str, float]]] = {}

    for index, frame_name in enumerate(frame_names, start=1):
        frame = tracks[tracks["Pair_ID"] == frame_name]
        matches = match_frame(
            frame,
            fundamental,
            EPIPOLAR_THRESHOLD,
            Y_SHIFT,
        )
        frame_matches[frame_name] = matches
        if index % 50 == 0 or index == len(frame_names):
            print(f"Matched {index}/{len(frame_names)} frames")

    mapping = build_consensus(
        frame_matches,
        len(frame_names),
        MIN_MATCH_FRACTION,
    )
    print(f"Accepted {len(mapping)} persistent stereo track links.")

    matches_2d = []
    points_3d = []

    for frame_name in frame_names:
        frame = tracks[tracks["Pair_ID"] == frame_name]
        matches = frame_matches[frame_name]

        for match in matches:
            if mapping.get(int(match["id_a"])) != int(match["id_b"]):
                continue

            point_3d = triangulate(
                (match["x_a"], match["y_a"]),
                (match["x_b"], match["y_b"]),
                p1,
                p2,
            )
            matches_2d.append(
                [
                    frame_name,
                    match["id_a"],
                    match["x_a"],
                    match["y_a"],
                    match["id_b"],
                    match["x_b"],
                    match["y_b"],
                    match["epipolar_distance"],
                ]
            )
            points_3d.append(
                [
                    frame_name,
                    match["id_a"],
                    point_3d[0],
                    point_3d[1],
                    point_3d[2],
                ]
            )

    OUTPUT_MATCHES_2D.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_3D.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        matches_2d,
        columns=[
            "Frame_ID",
            "Track_ID_A",
            "X_A",
            "Y_A",
            "Track_ID_B",
            "X_B",
            "Y_B",
            "Epipolar_Distance",
        ],
    ).to_csv(OUTPUT_MATCHES_2D, index=False)

    pd.DataFrame(
        points_3d,
        columns=["Frame_ID", "Stud_ID", "X", "Y", "Z"],
    ).to_csv(OUTPUT_3D, index=False)

    print(f"Saved 2D matches to: {OUTPUT_MATCHES_2D.resolve()}")
    print(f"Saved {len(points_3d)} 3D points to: {OUTPUT_3D.resolve()}")


if __name__ == "__main__":
    main()
