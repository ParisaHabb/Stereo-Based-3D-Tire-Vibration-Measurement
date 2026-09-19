"""Stereo camera calibration for the tire-vibration project.

Expected input:
    Two folders containing corresponding checkerboard images.

The checkerboard dimensions are the number of INNER corners, not the
number of squares. For example, a board with 14 x 21 inner corners must
be passed as pattern_cols=14 and pattern_rows=21.
"""

from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


ImagePair = Tuple[Path, Path]
PreviewCallback = Callable[[np.ndarray], None]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MIN_VALID_PAIRS = 10


def collect_images(folder: str) -> List[Path]:
    """Return supported image files in deterministic order."""
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    files = [
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.name.lower())


def pair_images(folder_a: str, folder_b: str) -> List[ImagePair]:
    """Pair images by sorted order after validating the folder lengths."""
    files_a = collect_images(folder_a)
    files_b = collect_images(folder_b)

    if not files_a or not files_b:
        raise FileNotFoundError(
            "No supported calibration images were found in one or both folders."
        )

    if len(files_a) != len(files_b):
        raise ValueError(
            "The two camera folders contain different numbers of images: "
            f"CamA={len(files_a)}, CamB={len(files_b)}. "
            "Ensure that corresponding images are present in both folders."
        )

    return list(zip(files_a, files_b))


def make_object_points(
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
) -> np.ndarray:
    """Create checkerboard 3D points on the Z=0 plane."""
    if pattern_cols < 2 or pattern_rows < 2:
        raise ValueError("pattern_cols and pattern_rows must both be at least 2.")
    if square_size_mm <= 0:
        raise ValueError("square_size_mm must be greater than zero.")

    object_points = np.zeros(
        (pattern_rows * pattern_cols, 3), dtype=np.float32
    )
    grid = np.mgrid[0:pattern_cols, 0:pattern_rows]
    object_points[:, :2] = (
        grid.T.reshape(-1, 2).astype(np.float32) * square_size_mm
    )
    return object_points


def create_preview(
    image_a: np.ndarray,
    image_b: np.ndarray,
    corners_a: Optional[np.ndarray],
    corners_b: Optional[np.ndarray],
    pattern_size: Tuple[int, int],
    scale_width: int = 1400,
) -> np.ndarray:
    """Create a side-by-side preview of detected checkerboard corners."""
    display_a = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
    display_b = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)

    if corners_a is not None:
        cv2.drawChessboardCorners(
            display_a, pattern_size, corners_a, corners_a is not None
        )
    if corners_b is not None:
        cv2.drawChessboardCorners(
            display_b, pattern_size, corners_b, corners_b is not None
        )

    height = max(display_a.shape[0], display_b.shape[0])
    width = display_a.shape[1] + display_b.shape[1]
    combined = np.zeros((height, width, 3), dtype=np.uint8)
    combined[: display_a.shape[0], : display_a.shape[1]] = display_a
    combined[: display_b.shape[0], display_a.shape[1] :] = display_b

    if combined.shape[1] > scale_width:
        scale = scale_width / combined.shape[1]
        combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale)

    return combined


def calculate_reprojection_error(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    rotation_vectors: List[np.ndarray],
    translation_vectors: List[np.ndarray],
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Return mean reprojection error and per-image errors in pixels."""
    per_image_errors = []

    for object_point, image_point, rotation_vector, translation_vector in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
    ):
        projected, _ = cv2.projectPoints(
            object_point,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion_coefficients,
        )
        observed = image_point.reshape(-1, 2)
        predicted = projected.reshape(-1, 2)
        error = np.linalg.norm(observed - predicted, axis=1).mean()
        per_image_errors.append(float(error))

    errors = np.asarray(per_image_errors, dtype=np.float64)
    return float(errors.mean()), errors


def stereo_calibrate(
    folder_a: str,
    folder_b: str,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    max_pairs: Optional[int] = 50,
    random_seed: int = 42,
    progress_callback: Optional[PreviewCallback] = None,
) -> Dict[str, object]:
    """Calibrate two cameras using corresponding checkerboard images."""
    pattern_size = (pattern_cols, pattern_rows)
    all_pairs = pair_images(folder_a, folder_b)

    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be None or greater than zero.")

    selected_pairs = all_pairs.copy()
    if max_pairs is not None and len(selected_pairs) > max_pairs:
        rng = random.Random(random_seed)
        selected_pairs = rng.sample(selected_pairs, max_pairs)
        selected_pairs.sort(key=lambda pair: pair[0].name.lower())

    object_template = make_object_points(
        pattern_cols,
        pattern_rows,
        square_size_mm,
    )

    object_points: List[np.ndarray] = []
    image_points_a: List[np.ndarray] = []
    image_points_b: List[np.ndarray] = []
    valid_pairs: List[ImagePair] = []
    last_preview = None
    image_size = None

    detection_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    termination_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path_a, image_path_b in selected_pairs:
        image_a = cv2.imread(str(image_path_a), cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(str(image_path_b), cv2.IMREAD_GRAYSCALE)

        if image_a is None or image_b is None:
            continue

        if image_a.shape != image_b.shape:
            raise ValueError(
                "Camera image sizes do not match for pair: "
                f"{image_path_a.name} / {image_path_b.name}."
            )

        if image_size is None:
            image_size = (image_a.shape[1], image_a.shape[0])
        elif image_size != (image_a.shape[1], image_a.shape[0]):
            raise ValueError("All CamA images must have the same resolution.")

        found_a, corners_a = cv2.findChessboardCorners(
            image_a,
            pattern_size,
            flags=detection_flags,
        )
        found_b, corners_b = cv2.findChessboardCorners(
            image_b,
            pattern_size,
            flags=detection_flags,
        )

        if found_a and found_b:
            corners_a = cv2.cornerSubPix(
                image_a,
                corners_a,
                (11, 11),
                (-1, -1),
                termination_criteria,
            )
            corners_b = cv2.cornerSubPix(
                image_b,
                corners_b,
                (11, 11),
                (-1, -1),
                termination_criteria,
            )

            object_points.append(object_template.copy())
            image_points_a.append(corners_a)
            image_points_b.append(corners_b)
            valid_pairs.append((image_path_a, image_path_b))

        last_preview = create_preview(
            image_a,
            image_b,
            corners_a if found_a else None,
            corners_b if found_b else None,
            pattern_size,
        )

        if progress_callback is not None and last_preview is not None:
            progress_callback(last_preview)

    if len(valid_pairs) < MIN_VALID_PAIRS:
        raise RuntimeError(
            f"Only {len(valid_pairs)} valid stereo pairs were found. "
            f"At least {MIN_VALID_PAIRS} are required. "
            "Use clear checkerboard images and verify the inner-corner size."
        )

    if image_size is None:
        raise RuntimeError("No readable calibration images were found.")

    width, height = image_size

    rms_a, k1, d1, rvecs_a, tvecs_a = cv2.calibrateCamera(
        object_points,
        image_points_a,
        image_size,
        None,
        None,
    )
    rms_b, k2, d2, rvecs_b, tvecs_b = cv2.calibrateCamera(
        object_points,
        image_points_b,
        image_size,
        None,
        None,
    )

    stereo_rms, k1, d1, k2, d2, rotation, translation, essential, fundamental = (
        cv2.stereoCalibrate(
            object_points,
            image_points_a,
            image_points_b,
            k1,
            d1,
            k2,
            d2,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    )

    projection_1 = k1 @ np.hstack(
        (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
    )
    projection_2 = k2 @ np.hstack((rotation, translation))

    reprojection_error_a, errors_a = calculate_reprojection_error(
        object_points,
        image_points_a,
        rvecs_a,
        tvecs_a,
        k1,
        d1,
    )
    reprojection_error_b, errors_b = calculate_reprojection_error(
        object_points,
        image_points_b,
        rvecs_b,
        tvecs_b,
        k2,
        d2,
    )

    return {
        "valid_pairs": len(valid_pairs),
        "valid_pair_names": [
            (path_a.name, path_b.name) for path_a, path_b in valid_pairs
        ],
        "image_size": image_size,
        "pattern_size": pattern_size,
        "square_size_mm": float(square_size_mm),
        "rms_cam_a": float(rms_a),
        "rms_cam_b": float(rms_b),
        "stereo_rms": float(stereo_rms),
        "reprojection_error_cam_a": reprojection_error_a,
        "reprojection_error_cam_b": reprojection_error_b,
        "reprojection_errors_cam_a": errors_a,
        "reprojection_errors_cam_b": errors_b,
        "K1": k1,
        "D1": d1,
        "K2": k2,
        "D2": d2,
        "R": rotation,
        "T": translation,
        "E": essential,
        "F": fundamental,
        "P1": projection_1,
        "P2": projection_2,
        "preview_img": last_preview,
    }


def save_calibration(result: Dict[str, object], output_file: str) -> None:
    """Save calibration matrices and metadata in a compressed NPZ file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        K1=result["K1"],
        D1=result["D1"],
        K2=result["K2"],
        D2=result["D2"],
        R=result["R"],
        T=result["T"],
        E=result["E"],
        F=result["F"],
        P1=result["P1"],
        P2=result["P2"],
        image_size=np.asarray(result["image_size"], dtype=np.int32),
        pattern_size=np.asarray(result["pattern_size"], dtype=np.int32),
        square_size_mm=np.asarray(result["square_size_mm"], dtype=np.float64),
        rms_cam_a=np.asarray(result["rms_cam_a"], dtype=np.float64),
        rms_cam_b=np.asarray(result["rms_cam_b"], dtype=np.float64),
        stereo_rms=np.asarray(result["stereo_rms"], dtype=np.float64),
        reprojection_error_cam_a=np.asarray(
            result["reprojection_error_cam_a"], dtype=np.float64
        ),
        reprojection_error_cam_b=np.asarray(
            result["reprojection_error_cam_b"], dtype=np.float64
        ),
    )


def main() -> None:
    """Example command-line configuration."""
    folder_a = "data/calibration/camA"
    folder_b = "data/calibration/camB"
    output_file = "data/calibration/stereo_calibration.npz"

    result = stereo_calibrate(
        folder_a=folder_a,
        folder_b=folder_b,
        pattern_cols=13,
        pattern_rows=20,
        square_size_mm=19.0,
        max_pairs=50,
        random_seed=42,
    )
    save_calibration(result, output_file)

    print(f"Valid stereo pairs: {result['valid_pairs']}")
    print(f"CamA RMS error: {result['rms_cam_a']:.4f} px")
    print(f"CamB RMS error: {result['rms_cam_b']:.4f} px")
    print(f"Stereo RMS error: {result['stereo_rms']:.4f} px")
    print(f"Saved calibration to: {output_file}")


if __name__ == "__main__":
    main()
