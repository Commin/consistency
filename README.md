# Consistency-aware Performance Monitoring and Model Retraining for Object Detection

This repository contains the implementation of **Prediction Consistency**, a label-free metric that measures the temporal stability of detection results across consecutive frames in time-series images (e.g., videos from onboard cameras). 
While we use YOLOv12 as the base object detection model, the core contribution of this repository focuses on runtime reliability assessment and dynamic data construction.

## Overview

Adjacent frames captured by onboard cameras often show high visual similarity. A reliable object detection model should produce stable predictions across these frames, including consistent bounding box locations and object class labels. Large variations in predictions across visually similar frames may indicate degraded model reliability or environmental mismatch. 

Instead of relying on ground-truth annotations to compute detection accuracy, the core idea of the proposed approach is to leverage **Prediction Consistency** as a label-free indicator to assess model reliability during runtime. 

Our method:
1. Evaluates bounding box overlap (IoU) and class agreement between adjacent frames.
2. Adapts the stability threshold dynamically using image similarity (SSIM).
3. Enables lightweight runtime monitoring directly on edge devices to detect abnormal drift.

## Project Structure and Implementation

The repository consists of several core components that implement our proposed methodologies:

### Core Algorithm
- **`consistency.py`**: The core logic for calculating Prediction Consistency. Computes the Intersection over Union (IoU) of bounding boxes and class alignments across consecutive frames using an optimal greedy matching approach to handle varying numbers of object detections.
- **`calibrate.py`**: Implements the Adaptive Consistency Envelope calibration. It uses Quantile Regression to model the relationship between visual similarity (SSIM) and expected spatial overlap (IoU). It provides the `detect_abnormal_drift` function for real-time model monitoring against environmental mismatches.

### Incremental Learning & Dataset Construction
- **`construct.py`**: Handles dynamic dataset construction for model retraining or incremental learning. It leverages the consistency scores (and other modes like confidence or random sampling) to intelligently select frames (e.g., picking 10% of frames with the lowest consistency for annotation and retraining).
- **`increment.py`**: The main incremental pipeline that orchestrates the end-to-end multi-stage process (e.g., sequentially from `clear_00` to `clear_50`). It seamlessly integrates YOLO training, evaluation, validation, and dynamic dataset construction based on the selected target metric.
- **`simulate_construct.py`**: A faster, lightweight simulation script that verifies the dataset generation and stage transition logic without needing to execute full YOLO training iterations.

### Model Training & Evaluation Handlers (YOLOv12 Base)
- **`train.py`**: YOLOv12 training wrapper script with per-epoch validations.
- **`val_test.py`**: Batch validation script to evaluate models across different incremental stages.
- **`test_yolo.py`**: Standard YOLOv12 inference script on video input.

## Usage

### 1. Calibrate Consistency Envelope and Detect Drift
Calibrate the threshold boundary based on a historical sequence and evaluate for abnormal drifts:
```bash
python calibrate.py --lbl-dir path/to/labels --img-dir path/to/images --quantile 0.95 --plot
```
This script generates an optimal boundary envelope `calibration_plot.png` and runs abnormal drift detection over the sequence, outputting instances of detected instability (adaptive boundary drift, fixed boundary, confidence jitter, center-shift).

### 2. Incremental Learning Pipeline
To run the full incremental training and dataset construction based heavily on Prediction Consistency, configure the inside of the script (e.g., `LEARNING_MODE = "consistency"`) and run:
```bash
python increment.py
```

### 3. Simulation of Dataset Construction
To verify the dataset construction and split logic without executing heavy model training, use:
```bash
python simulate_construct.py
```
