"""Optional edge-based single-point vibration analysis.

Save as:
    edge_single_point.py

This script reconstructs an edge ring from synchronized camera images,
selects one ring point, and analyzes one 3D coordinate over time. It is a
qualitative comparison method; the blob-based pipeline remains the main
result of the project.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


PAIRS_FOLDER = Path("data/raw/selected_pairs")
CALIBRATION_FILE = Path("data/calibration/stereo_calibration.npz")
OUTPUT_SIGNAL = Path("data/processed/edge_single_point_signal.csv")
OUTPUT_SPECTRUM = Path("data/processed/edge_single_point_spectrum.csv")

CAM_A_TOKEN = "camA"
CAM_B_TOKEN = "camB"
FPS = 75.0
POINTS_PER_RING = 180
POINT_INDEX = 0
COMPONENT = "y"
EPIPOLAR_DISTANCE_THRESHOLD = 1.0

USE_FILTER = True
FILTER_TYPE = "highpass"
HIGH_PASS_HZ = 3.0
LOW_PASS_HZ = 15.0
FILTER_ORDER = 4
FREQUENCY_MIN = 0.5
TOP_K_PEAKS = 5


def parse_filename(filename: str) -> Optional[Tuple[int, str]]:
    """Parse pair ID and camera token from a synchronized image filename."""
    match = re.search(
        r"pair_(\d+)_([A-Za-z0-9]+)_f\d+\.(?:png|jpg|jpeg|bmp)$",
        filename,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def load_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """Pair camera files by pair ID."""
    if not folder.exists():
        raise FileNotFoundError(f"Pair folder not found: {folder}")

    camera_a = {}
    camera_b = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        pair_id, camera_token = parsed
        if CAM_A_TOKEN.lower() in camera_token.lower():
            camera_a.setdefault(pair_id, path)
        elif CAM_B_TOKEN.lower() in camera_token.lower():
            camera_b.setdefault(pair_id, path)

    common_ids = sorted(set(camera_a) & set(camera_b))
    return [(camera_a[index], camera_b[index]) for index in common_ids]


def resample_contour(contour: np.ndarray, count: int) -> Optional[np.ndarray]:
    """Uniformly resample a closed contour by arc length."""
    points = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return None
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = cumulative[-1]
    if total_length <= 1e-9:
        return None

    targets = np.linspace(0.0, total_length, count, endpoint=False)
    result = np.empty((count, 2), dtype=float)
    for index, target in enumerate(targets):
        segment_index = np.searchsorted(cumulative, target, side="right") - 1
        segment_index = int(np.clip(segment_index, 0, len(segments) - 1))
        length = lengths[segment_index]
        ratio = (
            (target - cumulative[segment_index]) / length
            if length > 1e-12
            else 0.0
        )
        result[index] = (
            points[segment_index] * (1.0 - ratio)
            + points[segment_index + 1] * ratio
        )
    return result


def detect_inner_ring(
    frame: np.ndarray,
    points_per_ring: int,
) -> Optional[np.ndarray]:
    """Detect and resample the inner tire contour."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        binary,
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
    child_indices = [
        index
        for index, values in enumerate(hierarchy)
        if values[3] == outer_index
    ]

    if child_indices:
        inner_index = max(
            child_indices,
            key=lambda index: cv2.contourArea(contours[index]),
        )
        ring = resample_contour(contours[inner_index], points_per_ring)
    else:
        outer_points = resample_contour(outer, points_per_ring)
        ring = None if outer_points is None else center + 0.75 * (outer_points - center)

    if ring is None:
        return None

    top_index = int(np.argmin(ring[:, 1]))
    return np.roll(ring, -top_index, axis=0)


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a vector."""
    x, y, z = np.asarray(vector).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def load_calibration(filename: Path):
    """Load calibration data and construct normalized projection matrices."""
    if not filename.exists():
        raise FileNotFoundError(f"Calibration file not found: {filename}")

    with np.load(filename) as calibration:
        required = {"K1", "D1", "K2", "D2", "R", "T"}
        missing = required - set(calibration.files)
        if missing:
            raise ValueError(
                "Calibration file is missing: " + ", ".join(sorted(missing))
            )
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


def epipolar_refine(
    points_a: np.ndarray,
    points_b: np.ndarray,
    fundamental: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Move camera-B points onto their camera-B epipolar lines when close."""
    refined = points_b.copy()
    for index, (point_a, point_b) in enumerate(zip(points_a, points_b)):
        if not np.isfinite(point_a).all() or not np.isfinite(point_b).all():
            continue
        line = fundamental @ np.array([point_a[0], point_a[1], 1.0])
        a, b, c = line
        denominator = a * a + b * b
        if denominator <= 1e-12:
            continue
        distance = abs(a * point_b[0] + b * point_b[1] + c) / np.sqrt(denominator)
        if distance > threshold:
            continue
        correction = -(a * point_b[0] + b * point_b[1] + c) / denominator
        refined[index] = point_b + correction * np.array([a, b])
    return refined


def triangulate_frames(
    rings_a: np.ndarray,
    rings_b: np.ndarray,
    k1: np.ndarray,
    d1: np.ndarray,
    k2: np.ndarray,
    d2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Triangulate all valid ring points for all frames."""
    frame_count, point_count, _ = rings_a.shape
    result = np.full((frame_count, point_count, 3), np.nan, dtype=float)

    for frame_index in range(frame_count):
        valid = np.isfinite(rings_a[frame_index]).all(axis=1) & np.isfinite(
            rings_b[frame_index]
        ).all(axis=1)
        if not np.any(valid):
            continue

        points_a = rings_a[frame_index, valid].astype(np.float32).reshape(-1, 1, 2)
        points_b = rings_b[frame_index, valid].astype(np.float32).reshape(-1, 1, 2)
        undistorted_a = cv2.undistortPoints(points_a, k1, d1).reshape(-1, 2).T
        undistorted_b = cv2.undistortPoints(points_b, k2, d2).reshape(-1, 2).T
        homogeneous = cv2.triangulatePoints(p1, p2, undistorted_a, undistorted_b)
        denominator = homogeneous[3]
        finite = np.abs(denominator) > 1e-12
        reconstructed = np.full((len(denominator), 3), np.nan)
        reconstructed[finite] = (
            homogeneous[:3, finite] / denominator[finite]
        ).T
        result[frame_index, np.flatnonzero(valid)] = reconstructed

    return result


def interpolate(values: np.ndarray) -> np.ndarray:
    """Fill finite one-dimensional samples by linear interpolation."""
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(values)
    if finite.sum() == 1:
        return np.full_like(values, values[finite][0])
    indices = np.arange(len(values))
    values[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return values


def build_signal(points_3d: np.ndarray) -> np.ndarray:
    """Extract one coordinate of one ring point over time."""
    point_index = int(np.clip(POINT_INDEX, 0, points_3d.shape[1] - 1))
    component_index = {"x": 0, "y": 1, "z": 2}.get(COMPONENT.lower(), 1)
    signal = points_3d[:, point_index, component_index]
    signal = interpolate(signal)
    return signal - np.mean(signal)


def apply_filter(signal: np.ndarray) -> Tuple[np.ndarray, str]:
    """Apply a validated zero-phase Butterworth filter."""
    if not USE_FILTER or FILTER_TYPE.lower() == "none":
        return signal, "raw"

    nyquist = FPS / 2.0
    filter_type = FILTER_TYPE.lower()
    if filter_type == "highpass":
        normalized = HIGH_PASS_HZ / nyquist
        cutoff_label = f"highpass {HIGH_PASS_HZ:.2f} Hz"
        btype = "highpass"
        cutoff = normalized
    elif filter_type == "lowpass":
        normalized = LOW_PASS_HZ / nyquist
        cutoff_label = f"lowpass {LOW_PASS_HZ:.2f} Hz"
        btype = "lowpass"
        cutoff = normalized
    elif filter_type == "bandpass":
        normalized = [HIGH_PASS_HZ / nyquist, LOW_PASS_HZ / nyquist]
        cutoff_label = f"bandpass {HIGH_PASS_HZ:.2f}-{LOW_PASS_HZ:.2f} Hz"
        btype = "bandpass"
        cutoff = normalized
    else:
        raise ValueError(f"Unknown FILTER_TYPE: {FILTER_TYPE}")

    if np.any(np.asarray(cutoff) <= 0) or np.any(np.asarray(cutoff) >= 1):
        raise ValueError("Filter cutoffs must lie strictly between 0 and Nyquist.")
    if filter_type == "bandpass" and HIGH_PASS_HZ >= LOW_PASS_HZ:
        raise ValueError("HIGH_PASS_HZ must be lower than LOW_PASS_HZ.")

    try:
        from scipy.signal import butter, filtfilt
    except ImportError as error:
        raise ImportError("Install scipy to use filtering.") from error

    coefficients = butter(FILTER_ORDER, cutoff, btype=btype)
    pad_length = 3 * max(len(coefficients[0]), len(coefficients[1]))
    if len(signal) <= pad_length:
        raise ValueError("Signal is too short for the selected filter.")
    return filtfilt(*coefficients, signal), cutoff_label


def analyze_spectrum(signal: np.ndarray):
    """Return FFT frequencies, amplitudes and strongest valid peaks."""
    detrended = signal - np.polyval(
        np.polyfit(np.arange(len(signal)), signal, 1),
        np.arange(len(signal)),
    )
    transformed = np.fft.rfft(detrended * np.hanning(len(detrended)))
    frequencies = np.fft.rfftfreq(len(detrended), 1.0 / FPS)
    amplitude = np.abs(transformed)
    valid = frequencies >= FREQUENCY_MIN
    peaks, _ = find_peaks(
        np.where(valid, amplitude, 0.0),
        height=np.max(amplitude[valid]) * 0.1 if np.any(valid) else 0.0,
    )
    peaks = sorted(peaks, key=lambda index: amplitude[index], reverse=True)
    return detrended, frequencies, amplitude, peaks[:TOP_K_PEAKS]


def main() -> None:
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
            rings_a[index] = detected_a
        if detected_b is not None:
            rings_b[index] = detected_b

    k1, d1, k2, d2, p1, p2, fundamental = load_calibration(CALIBRATION_FILE)
    refined_b = np.full_like(rings_b, np.nan)
    for index in range(len(pairs)):
        valid = np.isfinite(rings_a[index]).all(axis=1) & np.isfinite(rings_b[index]).all(axis=1)
        refined_b[index] = rings_b[index]
        if np.any(valid):
            refined_b[index, valid] = epipolar_refine(
                rings_a[index, valid],
                rings_b[index, valid],
                fundamental,
                EPIPOLAR_DISTANCE_THRESHOLD,
            )

    points_3d = triangulate_frames(
        rings_a,
        refined_b,
        k1,
        d1,
        k2,
        d2,
        p1,
        p2,
    )
    raw_signal = build_signal(points_3d)
    filtered_signal, filter_label = apply_filter(raw_signal)
    detrended, frequencies, amplitude, peaks = analyze_spectrum(filtered_signal)

    OUTPUT_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT_SIGNAL,
        np.column_stack((np.arange(len(raw_signal)), raw_signal, filtered_signal)),
        delimiter=",",
        header="Frame_Index,Raw_Signal,Filtered_Signal",
        comments="",
    )
    np.savetxt(
        OUTPUT_SPECTRUM,
        np.column_stack((frequencies, amplitude)),
        delimiter=",",
        header="Frequency_Hz,Amplitude",
        comments="",
    )

    print(f"Filter: {filter_label}")
    print(f"Saved signal: {OUTPUT_SIGNAL.resolve()}")
    print(f"Saved spectrum: {OUTPUT_SPECTRUM.resolve()}")
    for peak in peaks:
        print(f"Peak: {frequencies[peak]:.4f} Hz")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    time = np.arange(len(raw_signal)) / FPS
    axes[0].plot(time, raw_signal, "k--", alpha=0.4, label="Raw")
    axes[0].plot(time, filtered_signal, "b-", label=filter_label)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(f"3D {COMPONENT.upper()} coordinate")
    axes[0].set_title("Single-point edge signal")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(frequencies, amplitude, "r-")
    for peak in peaks:
        axes[1].axvline(frequencies[peak], linestyle="--", alpha=0.5)
    axes[1].set_xlim(0, FPS / 2.0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("FFT amplitude")
    axes[1].set_title("Single-point spectrum")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
