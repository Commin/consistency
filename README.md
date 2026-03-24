# Consistency-aware Performance Monitoring and Model Retraining for Object Detection

This [repository](https://github.com/Commin/consistency) contains the implementation of **Prediction Consistency**, a label-free metric that measures the temporal stability of detection results across consecutive frames in time-series images (e.g., videos from onboard cameras). 
While we use YOLOv12 as the base object detection model, the core contribution of this repository focuses on runtime reliability assessment and dynamic data construction.

**Please clone YOLOv12 before you use this repository.**

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

### Model Training & Evaluation Handlers (YOLOv12 Base)
- **`train.py`**: YOLOv12 training wrapper script with per-epoch validations.
- **`val_test.py`**: Batch validation script to evaluate models across different incremental stages.
- **`test_yolo.py`**: Standard YOLOv12 inference script on video input.

## Usage

### 1. SSIM Calculation

```bash
python ssim.py --gt-dir path/to/ground_truth --output-dir path/to/output --csv-name ssim_results.csv
```

### 2. Calibrate Consistency Envelope
Calibrate the threshold boundary based on a historical sequence and evaluate for abnormal drifts:
```bash
python calibrate_video.py --lbl-dir path/to/labels --img-dir path/to/images --quantile 0.95 --plot
```
This script generates an optimal boundary envelope `calibration_plot.png` and runs drift detection over the sequence, outputting instances of detected instability.


### 3. Incremental Learning Pipeline
To run the full incremental training and dataset construction based heavily on Prediction Consistency, configure the inside of the script (e.g., `LEARNING_MODE = "consistency"`) and run:
```bash
python increment.py --mode consistency --train --increment --stage 0
```

### 4. Evaluation Experiment

To use the `exp` folder python files for the evaluation experiment:

**1. Generate aggregate agreement score, match coverage ratio:**
Outputs: `all_val_data_all/labels/`, `all_consistency_eval/frame_consistency_.csv`, `all_consistency_eval/object_consistency_.json`
```bash
python val_consistency.py --data dataset_path/data.yaml --weights weights_path/best.pt --project project_path
```

**2. Generate ground-truth multi-object tracking (`gt_track`):**
```bash
python generate_gt_consistency_reference_tracker.py --gt-dir dataset_path/labels/ --save-path grouped_gt_tracker
```

**3. Generate oracle consistency boundary, index summary, and calibration metrics:**
```bash
python generate_gt_envelope_from_tracker.py --grouped-gt-root grouped_gt_tracker \
--ssim-path ssim_results.csv \
--save-path gt_envelope_out_with_unmatched \
```

**4. Generate and evaluate calibrated consistency boundary based on oracle boundary:**
Outputs: `best_consistency_boundary.png`, `index_summary.json`, `per_prefix_calibration_metrics.csv`
```bash
python eval_calibrate_robust.py --frame-path object_consistency_all_data.json --ssim-path all_ssim.csv --save-path save_path --plot
```

**5. Generate global metrics, per prefix metrics, and method configs:**
Outputs: `global_metrics.csv`, `per_prefix_metrics.csv`, `method_configs.json`
```bash
python eval_consistency_accuracy_filtered.py --frame-path object_consistency_all_data.json --ssim-path all_ssim.csv --grouped-gt-root grouped_gt_tracker/ --calibration-root grouped_robust_calibrate_results/ --save-path save_path
```

## Dataset

The experiments in our paper are conducted on the Singapore Maritime Dataset (SMD), a large-scale maritime image dataset collected from onboard cameras. The dataset contains diverse maritime scenarios, including various weather conditions, lighting variations, and object types (e.g., vessels, buoys). The SMD provides a rich testbed for evaluating the robustness and reliability of object detection models in real-world maritime environments.

[SMD official](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset)
[SMD image dataset](https://universe.roboflow.com/maritime-cumkb/singapore-maritime)