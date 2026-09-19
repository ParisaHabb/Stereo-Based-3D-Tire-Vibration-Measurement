"""Triggered image acquisition using a Hikrobot camera and external trigger.

Save this file as:
    acquisition.py

The Hikrobot SDK folder must be available as:
    <project_root>/MvImport/

Expected SDK modules include:
    MvCameraControl_class.py
    CameraParams_header.py
"""

from ctypes import POINTER, byref, c_ubyte, cast, memmove
from pathlib import Path
import csv
import queue
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


# ------------------------- User configuration -------------------------
CAMERA_LABEL = "CAM_A"
OUTPUT_ROOT = Path("data/raw")

EXPOSURE_US = 3000.0
FRAME_TIMEOUT_MS = 2000
VIDEO_FPS = 80.0
QUEUE_SIZE = 300
SAVE_VIDEO = True
SHOW_PREVIEW = True
SDK_FOLDER_NAME = "MvImport"
# ----------------------------------------------------------------------


def load_hikrobot_sdk():
    """Import the Hikrobot SDK from the project-local MvImport folder."""
    script_dir = Path(__file__).resolve().parent
    sdk_dir = script_dir / SDK_FOLDER_NAME

    if not sdk_dir.exists():
        raise ImportError(
            f"Hikrobot SDK folder was not found: {sdk_dir}. "
            "Place the MvImport folder beside this script."
        )

    sdk_dir_string = str(sdk_dir)
    if sdk_dir_string not in sys.path:
        sys.path.insert(0, sdk_dir_string)

    try:
        from MvCameraControl_class import (  # type: ignore
            MV_ACCESS_Exclusive,
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_CC_EnumDevices,
            MV_CC_GetImageBuffer,
            MV_CC_FreeImageBuffer,
            MV_GIGE_DEVICE,
            MV_OK,
            MV_USB_DEVICE,
            MvCamera,
            MV_FRAME_OUT,
            MV_TRIGGER_MODE_ON,
            PixelType_Gvsp_BGR8_Packed,
            PixelType_Gvsp_Mono8,
        )
    except ImportError as error:
        raise ImportError(
            "Could not import the Hikrobot SDK. Check the MvImport folder "
            "and the installed SDK files."
        ) from error

    return {
        "MV_ACCESS_Exclusive": MV_ACCESS_Exclusive,
        "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
        "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
        "MV_CC_EnumDevices": MV_CC_EnumDevices,
        "MV_CC_GetImageBuffer": MV_CC_GetImageBuffer,
        "MV_CC_FreeImageBuffer": MV_CC_FreeImageBuffer,
        "MV_GIGE_DEVICE": MV_GIGE_DEVICE,
        "MV_OK": MV_OK,
        "MV_USB_DEVICE": MV_USB_DEVICE,
        "MvCamera": MvCamera,
        "MV_FRAME_OUT": MV_FRAME_OUT,
        "MV_TRIGGER_MODE_ON": MV_TRIGGER_MODE_ON,
        "PixelType_Gvsp_BGR8_Packed": PixelType_Gvsp_BGR8_Packed,
        "PixelType_Gvsp_Mono8": PixelType_Gvsp_Mono8,
    }


def error_code_to_hex(error_code: int) -> str:
    """Format a Hikrobot error code."""
    value = int(error_code)
    if value < 0:
        value += 2**32
    return hex(value)


def frame_to_bgr(frame_out, pixel_mono8: int, pixel_bgr8: int) -> Optional[np.ndarray]:
    """Copy an SDK frame buffer and convert it to an OpenCV BGR image."""
    info = frame_out.stFrameInfo
    width = int(info.nWidth)
    height = int(info.nHeight)
    frame_length = int(info.nFrameLen)

    if frame_out.pBufAddr is None or frame_length <= 0:
        return None

    raw_buffer = (c_ubyte * frame_length)()
    memmove(byref(raw_buffer), frame_out.pBufAddr, frame_length)
    data = np.frombuffer(raw_buffer, dtype=np.uint8)

    if info.enPixelType == pixel_mono8:
        expected_size = width * height
        if data.size < expected_size:
            return None
        gray = data[:expected_size].reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if info.enPixelType == pixel_bgr8:
        expected_size = width * height * 3
        if data.size < expected_size:
            return None
        return data[:expected_size].reshape(height, width, 3).copy()

    return None


def save_worker(
    write_queue: queue.Queue,
    csv_path: Path,
    video_writer: Optional[cv2.VideoWriter],
) -> None:
    """Write captured frames and timestamps without blocking acquisition."""
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_idx", "timestamp_ns"])

        while True:
            item = write_queue.get()
            try:
                if item is None:
                    return

                frame_idx, image, timestamp_ns = item
                if video_writer is not None:
                    video_writer.write(image)
                writer.writerow([frame_idx, timestamp_ns])
                csv_file.flush()
            finally:
                write_queue.task_done()


def configure_camera(camera, sdk: dict) -> None:
    """Configure trigger mode, trigger source, pixel format and exposure."""
    ok = sdk["MV_OK"]

    result = camera.MV_CC_SetEnumValue(
        "TriggerMode",
        sdk["MV_TRIGGER_MODE_ON"],
    )
    if result != ok:
        raise RuntimeError(
            f"Could not enable trigger mode: {error_code_to_hex(result)}"
        )

    result = camera.MV_CC_SetEnumValueByString("TriggerSource", "Line0")
    if result != ok:
        result = camera.MV_CC_SetEnumValue("TriggerSource", 0)
    if result != ok:
        raise RuntimeError(
            f"Could not configure trigger source: {error_code_to_hex(result)}"
        )

    result = camera.MV_CC_SetEnumValueByString("PixelFormat", "Mono8")
    if result != ok:
        raise RuntimeError(
            f"Could not set Mono8 pixel format: {error_code_to_hex(result)}"
        )

    result = camera.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
    if result != ok:
        raise RuntimeError(
            f"Could not disable auto exposure: {error_code_to_hex(result)}"
        )

    result = camera.MV_CC_SetFloatValue("ExposureTime", float(EXPOSURE_US))
    if result != ok:
        raise RuntimeError(
            f"Could not set exposure time: {error_code_to_hex(result)}"
        )


def open_first_camera(sdk: dict):
    """Enumerate and open the first connected GigE or USB camera."""
    device_list = sdk["MV_CC_DEVICE_INFO_LIST"]()
    device_types = sdk["MV_GIGE_DEVICE"] | sdk["MV_USB_DEVICE"]
    result = sdk["MvCamera"].MV_CC_EnumDevices(device_types, device_list)

    if result != sdk["MV_OK"]:
        raise RuntimeError(
            f"Camera enumeration failed: {error_code_to_hex(result)}"
        )
    if device_list.nDeviceNum == 0:
        raise RuntimeError("No GigE or USB cameras were found.")

    device_info = cast(
        device_list.pDeviceInfo[0],
        POINTER(sdk["MV_CC_DEVICE_INFO"]),
    ).contents

    camera = sdk["MvCamera"]()
    result = camera.MV_CC_CreateHandle(device_info)
    if result != sdk["MV_OK"]:
        raise RuntimeError(
            f"CreateHandle failed: {error_code_to_hex(result)}"
        )

    result = camera.MV_CC_OpenDevice(
        sdk["MV_ACCESS_Exclusive"],
        0,
    )
    if result != sdk["MV_OK"]:
        camera.MV_CC_DestroyHandle()
        raise RuntimeError(
            f"OpenDevice failed: {error_code_to_hex(result)}"
        )

    return camera


def run_recording() -> Tuple[Path, Path, int]:
    """Capture one triggered burst and return output paths and frame count."""
    sdk = load_hikrobot_sdk()
    camera = None
    writer = None
    worker = None
    write_queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
    output_dir = None
    csv_path = None
    video_path = None
    received_count = 0
    frame_out = sdk["MV_FRAME_OUT"]()

    try:
        camera = open_first_camera(sdk)
        configure_camera(camera, sdk)

        result = camera.MV_CC_StartGrabbing()
        if result != sdk["MV_OK"]:
            raise RuntimeError(
                f"StartGrabbing failed: {error_code_to_hex(result)}"
            )

        timestamp_label = time.strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_ROOT / f"{CAMERA_LABEL}_{timestamp_label}"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "timestamps.csv"
        video_path = output_dir / "output.avi"

        print(f"{CAMERA_LABEL} is ready. Waiting for Arduino trigger...")

        recording_started = False
        frame_idx = 0
        last_report_time = time.perf_counter()
        last_report_frame = 0

        while True:
            result = camera.MV_CC_GetImageBuffer(
                frame_out,
                FRAME_TIMEOUT_MS,
            )
            timestamp_ns = time.perf_counter_ns()

            if result != sdk["MV_OK"] or frame_out.pBufAddr is None:
                if received_count > 0:
                    print("Capture timeout: burst finished.")
                    break
                continue

            received_count += 1
            frame_idx += 1
            recording_started = True

            width = int(frame_out.stFrameInfo.nWidth)
            height = int(frame_out.stFrameInfo.nHeight)

            if writer is None and SAVE_VIDEO:
                codec = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(
                    str(video_path),
                    codec,
                    VIDEO_FPS,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open video writer: {video_path}"
                    )

            image = frame_to_bgr(
                frame_out,
                sdk["PixelType_Gvsp_Mono8"],
                sdk["PixelType_Gvsp_BGR8_Packed"],
            )
            sdk["MV_CC_FreeImageBuffer"](camera, frame_out)

            if image is None:
                print(f"Warning: unsupported or invalid frame {frame_idx}.")
                continue

            if SHOW_PREVIEW:
                preview_width = min(width, 800)
                preview_height = int(height * preview_width / width)
                preview = cv2.resize(
                    image,
                    (preview_width, preview_height),
                )
                cv2.imshow(f"Preview - {CAMERA_LABEL}", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Recording stopped by user.")
                    break

            try:
                write_queue.put_nowait((frame_idx, image, timestamp_ns))
            except queue.Full:
                print(f"Warning: saving queue full; frame {frame_idx} dropped.")

            current_time = time.perf_counter()
            if current_time - last_report_time >= 0.5:
                elapsed = current_time - last_report_time
                current_fps = (frame_idx - last_report_frame) / elapsed
                print(
                    f"Captured: {frame_idx} | "
                    f"Queue: {write_queue.qsize()} | "
                    f"FPS: {current_fps:.1f}",
                    end="\r",
                )
                last_report_time = current_time
                last_report_frame = frame_idx

        if not recording_started:
            raise RuntimeError("No trigger frames were received.")

        print("\nWaiting for queued frames to be saved...")
        write_queue.join()

        return video_path, csv_path, received_count

    finally:
        if worker is not None and worker.is_alive():
            write_queue.put(None)
            write_queue.join()
            worker.join(timeout=10)

        if writer is not None:
            writer.release()

        if camera is not None:
            try:
                camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                camera.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                camera.MV_CC_DestroyHandle()
            except Exception:
                pass

        cv2.destroyAllWindows()


def main() -> None:
    try:
        video_path, csv_path, frame_count = run_recording()
        print(f"Finished. Frames received: {frame_count}")
        print(f"Video: {video_path.resolve()}")
        print(f"Timestamps: {csv_path.resolve()}")
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Recording failed: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
