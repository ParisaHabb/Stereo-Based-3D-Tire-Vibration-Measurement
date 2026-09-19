"""Optional edge-based mean-radius analysis.

Save as:
    edge_mean_radius.py

This method is mainly for comparison with the stud-tracking pipeline. It
can suppress bending modes because circumferential averaging may cancel
positive and negative lobes.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


PAIRS_FOLDER = Path("data/raw/selected_pairs")
CAM_A_TOKEN = "camA"
CAM_B_TOKEN = "camB"
OUTPUT_SIGNAL = Path("data/processed/edge_mean_radius_signal.csv")
OUTPUT_SPECTRUM = Path("data/processed/edge_mean_radius_spectrum.csv")

FPS = 75.0
POINTS_PER_RING = 360
EPIPOLAR_DISTANCE_THRESHOLD = 1.0
FREQUENCY_MIN = 0.5
TOP_K_PEAKS = 5

KALMAN_UNUSED = False


def pair_key(filename: str) -> Optional[Tuple[int, int, str]]:
    """Parse pair_00282_camA_f00291.png-like filenames."""
    match = re.search(
        r"pair_(\d+)_([A-Za-z0-9]+)_f(\d+)\.(?:png|jpg|jpeg|bmp)$",
        filename,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(3)), match.group(2)


def load_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """Match camera A/B files by pair index."""
    if not folder.exists():
        raise FileNotFoundError(f"Pair folder not found: {folder}")

    camera_a = {}
    camera_b = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        parsed = pair_key(path.name)
        if parsed is None:
            continue
        pair_id, frame_id, camera_token = parsed
        target = camera_a if CAM_A_TOKEN.lower() in camera_token.lower() else camera_b
        target.setdefault(pair_id, path)

    common_ids = sorted(set(camera_a) & set(camera_b))
    pairs = [(camera_a[index], camera_b[index]) for index in common_ids]
    print(f"Matched {len(pairs)} camera pairs.")
    return pairs


def resample_contour(contour: np.ndarray, count: int) -> Optional[np.ndarray]:
    """Resample a closed contour uniformly by arc length."""
    points = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return None
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))

    segment = np.diff(points, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = cumulative[-1]
    if total <= 1e-9:
        return None

    targets = np.linspace(0.0, total, count, endpoint=False)
    output = np.empty((count, 2), dtype=float)
    for index, target in enumerate(targets):
        segment_index = np.searchsorted(cumulative, target, side="right") - 1
        segment_index = int(np.clip(segment_index, 0, len(segment) - 1))
        denominator = lengths[segment_index]
        ratio = (
            (target - cumulative[segment_index]) / denominator
            if denominator > 1e-12
            else 0.0
        )
        output[index] = (
            points[segment_index] * (1.0 - ratio)
            + points[segment_index + 1] * ratio
        )
    return output


def detect_inner_ring(
    frame: np.ndarray,
    points_per_ring: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Detect a tire-like outer contour and its largest inner child."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    parent_indices = [
        index
        for index, values in enumerate(hierarchy)
        if values[3] == -1
    ]
    if not parent_indices:
        return None

    outer_index = max(
        parent_indices,
        key=lambda index: cv2.contourArea(contours[index]),
    )
    outer = contours[outer_index]
    moments = cv2.moments(outer)
    if abs(moments["m00"]) < 1e-12:
        return None

    center = np.array(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ],
        dtype=float,
    )
    children = [
        index
        for index, values in enumerate(hierarchy)
        if values[3] == outer_index
    ]

    if children:
        inner_index = max(
            children,
            key=lambda index: cv2.contourArea(contours[index]),
        )
        ring = resample_contour(contours[inner_index], points_per_ring)
    else:
        outer_points = resample_contour(outer, points_per_ring)
        ring = None if outer_points is None else center + 0.75 * (outer_points - center)

    if ring is None:
        return None

    top = int(np.argmin(ring[:, 1]))
    ring = np.roll(ring, -top, axis=0)
    return center, ring


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3-vector."""
    x, y, z = np.asarray(vector).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def load_calibration(filename: Path):
    """Load camera matrices and build projection matrices."""
    if not filename.exists():
        raise FileNotFoundError(f"Calibration file not found: {filename}")
    with np.load(filename) as calibration:
        required = {"K1", "D1", "K2", "D2", "R", "T"}
        missing = required - set(calibration.files)
        if missing:
            raise ValueError("Calibration file missing: " + ", ".join(sorted(missing)))
        k1 = calibration["K1"]
        d1 = calibration["D1"]
        k2 = calibration["K2"]
        d2 = calibration["D2"]
        rotation = calibration["R"]
        translation = calibration["T"].reshape(3, 1)

    p1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    p2 = np.hstack((rotation, translation))
    fundamental = np.linalg.inv(k2).T @ skew(translation) @ rotation @ np.linalg.inv(k1)
    return k1, d1, k2, d2, p1, p2, fundamental


def triangulate_rings(
    rings_a: np.ndarray,
    rings_b: np.ndarray,
    k1: np.ndarray,
    d1: np.ndarray,
    k2: np.ndarray,
    d2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Undistort and triangulate all ring points."""
    frame_count, point_count, _ = rings_a.shape
    result = np.full((frame_count, point_count, 3), np.nan, dtype=float)

    for frame_index in range(frame_count):
        points_a = rings_a[frame_index].astype(np.float32).reshape(-1, 1, 2)
        points_b = rings_b[frame_index].astype(np.float32).reshape(-1, 1, 2)
        valid = np.isfinite(points_a).all(axis=(1, 2)) & np.isfinite(points_b).all(axis=(1, 2))
        if not np.any(valid):
            continue

        undistorted_a = cv2.undistortPoints(points_a[valid], k1, d1).reshape(-1, 2).T
        undistorted_b = cv2.undistortPoints(points_b[valid], k2, d2).reshape(-1, 2).T
        homogeneous = cv2.triangulatePoints(p1, p2, undistorted_a, undistorted_b)
        denominator = homogeneous[3]
        valid_denominator = np.abs(denominator) > 1e-12
        points_3d = np.full((np.count_nonzero(valid), 3), np.nan)
        points_3d[valid_denominator] = (
            homogeneous[:3, valid_denominator] / denominator[valid_denominator]
        ).T
        result[frame_index, np.flatnonzero(valid)] = points_3d

    return result


def fill_nan_series(values: np.ndarray) -> np.ndarray:
    """Interpolate a one-dimensional series where possible."""
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(values)
    if finite.sum() == 1:
        return np.full_like(values, values[finite][0])
    indices = np.arange(len(values))
    values[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return values


def mean_radius_signal(points_3d: np.ndarray) -> np.ndarray:
    """Compute mean radial distance from each frame's point-cloud center."""
    signal = np.full(points_3d.shape[0], np.nan, dtype=float)
    for frame_index, frame_points in enumerate(points_3d):
        center = np.nanmean(frame_points, axis=0)
        radius = np.linalg.norm(frame_points - center, axis=1)
        signal[frame_index] = np.nanmean(radius)
    return fill_nan_series(signal)


def estimate_spectrum(signal: np.ndarray):
    """Calculate the one-sided FFT spectrum and dominant peaks."""
    values = fill_nan_series(signal)
    values = values - np.polyval(
        np.polyfit(np.arange(len(values)), values, 1),
        np.arange(len(values)),
    )
    windowed = values * np.hanning(len(values))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(values), 1.0 / FPS)
    valid = frequencies >= FREQUENCY_MIN
    peak_indices, _ = find_peaks(
        np.where(valid, spectrum, 0.0),
        height=np.max(spectrum[valid]) * 0.1 if np.any(valid) else 0.0,
    )
    ranked = sorted(
        peak_indices,
        key=lambda index: spectrum[index],
        reverse=True,
    )[:TOP_K_PEAKS]
    return values, frequencies, spectrum, ranked


def main() -> None:
    calibration_file = Path("data/calibration/stereo_calibration.npz")
    pairs = load_pairs(PAIRS_FOLDER)
    if not pairs:
        raise RuntimeError("No synchronized camera pairs were found.")

    rings_a = np.full((len(pairs), POINTS_PER_RING, 2), np.nan)
    rings_b = np.full_like(rings_a, np.nan)
    for index, (path_a, path_b) in enumerate(pairs):
        image_a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
        image_b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
        if image_a is None or image_b is None:
            continue
        detected_a = detect_inner_ring(image_a, POINTS_PER_RING)
        detected_b = detect_inner_ring(image_b, POINTS_PER_RING)
        if detected_a is not None:
            rings_a[index] = detected_a[1]
        if detected_b is not None:
            rings_b[index] = detected_b[1]

    k1, d1, k2, d2, p1, p2, _ = load_calibration(calibration_file)
    points_3d = triangulate_rings(rings_a, rings_b, k1, d1, k2, d2, p1, p2)
    signal = mean_radius_signal(points_3d)
    detrended, frequencies, spectrum, peaks = estimate_spectrum(signal)

    OUTPUT_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    pd_signal = np.column_stack((np.arange(len(signal)), signal, detrended))
    np.savetxt(
        OUTPUT_SIGNAL,
        pd_signal,
        delimiter=",",
        header="Frame_Index,Mean_Radius,Detrended_Mean_Radius",
        comments="",
    )
    np.savetxt(
        OUTPUT_SPECTRUM,
        np.column_stack((frequencies, spectrum)),
        delimiter=",",
        header="Frequency_Hz,Amplitude",
        comments="",
    )

    print(f"Saved signal to: {OUTPUT_SIGNAL.resolve()}")
    print(f"Saved spectrum to: {OUTPUT_SPECTRUM.resolve()}")
    for peak in peaks:
        print(f"Peak: {frequencies[peak]:.4f} Hz")

    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, spectrum)
    for peak in peaks:
        plt.axvline(frequencies[peak], linestyle="--", alpha=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT amplitude")
    plt.title("Edge-based mean-radius spectrum")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
