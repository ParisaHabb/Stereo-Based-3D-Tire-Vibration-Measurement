# Stereo-Based 3D Tire Vibration Measurement

This project presents a non-contact computer-vision method for measuring tire vibration using a synchronized stereo-camera system.

Two high-speed cameras observe a tire equipped with bright reflective studs. The system tracks the studs in both camera views and reconstructs their three-dimensional motion using stereo vision.

The reconstructed 3D trajectories can then be used to study the dynamic response of the tire, identify dominant vibration frequencies, and visualize vibration mode shapes.

The project also includes edge-based analysis methods for comparison with the primary stud-tracking approach.

## Project Overview

The main concept of the project is:

```text
Stereo camera images
        ↓
Reflective stud tracking
        ↓
3D trajectory reconstruction
        ↓
Tire vibration analysis
```

This repository contains the source code and documentation developed for the stereo-based measurement and analysis of tire vibration.

## Main Components

- Stereo-camera calibration.
- Hardware-triggered image acquisition.
- Tire-region and edge detection.
- Reflective-stud detection.
- Two-dimensional multi-object tracking.
- Stereo matching and 3D reconstruction.
- Three-dimensional trajectory visualization.
- Frequency analysis.
- Vibration mode-shape visualization.
- Optional edge-based comparison methods.

## Methods

The primary method is based on detecting and tracking reflective studs distributed on the tire surface.

Additional edge-based methods are included to compare their behavior with the stud-tracking method. These methods analyze the reconstructed tire edge using mean-radius and single-point approaches.

## Documentation

Detailed information about the purpose, configuration, inputs, outputs, and usage of each code file is available in the project manual:

[Code Manual](docs/code_manual_english.md)

## Project Scope

This repository focuses on the computer-vision and stereo-reconstruction workflow used to measure tire vibration. The acquisition stage depends on the camera hardware, external trigger system, and the corresponding camera SDK.

Raw videos, private datasets, calibration results, and other experiment-specific files are not included in this repository.
