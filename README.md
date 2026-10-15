# SolarEnergyOpt

**Exploring solar-energy optimization through computer vision and machine learning.**

SolarEnergyOpt is a Python research prototype with two experimental tracks: measuring cloud coverage from sky images, and predicting a solar panel’s maximum-power-point voltage (`V_mpp`) from labeled electrical measurements.

## What is implemented

- **Sky recording:** capture camera video and log frame brightness statistics.
- **Cloud segmentation:** use HSV color thresholds and morphological filtering to estimate cloud coverage while excluding very bright glare pixels.
- **Feature extraction:** export cloud coverage and brightness statistics from saved frames; compute electrical features such as power, changes over time, and rolling statistics.
- **Model comparison:** train Ridge regression and a small neural network on labeled electrical data, report MAE and R², and save the model with the lower test MAE.
- **Visualization:** inspect segmentation overlays and cloud-coverage plots.

The vision scripts and electrical-model training are separate workflows. The repository does not yet provide an integrated solar controller or measured energy-yield improvements.

## Quick start: explore the included frames

Use Python 3.11 or later. Run commands from the repository root.

```bash
git clone https://github.com/coder-anirud/SolarEnergyOpt.git
cd SolarEnergyOpt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/extract_cloud_features.py --frames recordings/frames_20260227_001753 --fps 10 --out recordings/cloud_demo.csv
python src/visualize.py
```

The extraction command processes the included frames without requiring a camera. It writes `frame_id`, `t_sec`, `cloud_coverage`, `mean_gray`, and `std_gray`. The visualization script opens the alphabetically latest `cloud_*.csv`; with the included files, `cloud_demo.csv` will be selected.

These recordings are development samples, not a labeled benchmark for segmentation accuracy.

## Capture and inspect a recording

1. Set `CAM_INDEX` in [collect_video.py](src/collect_video.py) for your camera; it currently defaults to `1`. Adjust `FPS`, `DURATION_SEC`, and `RES` as needed.
2. Run `python src/collect_video.py`. Allow camera access when prompted. The script tries macOS AVFoundation first, then the default OpenCV backend.
3. Run `python src/visualize_segmentation.py` to inspect the latest MP4 with a cloud-mask overlay. Press **q** to close the viewer.

The capture script writes an MP4 and a brightness CSV. It does not currently export individual frames. Cloud-feature extraction requires a separate `frames_*` folder containing `frame_*.jpg`, `frame_*.jpeg`, or `frame_*.png` files, such as the included example.

## Train a maximum-power-point voltage model

Supply your own labeled CSV; the electrical training dataset is not included. Each row must contain:

| Column | Meaning |
| --- | --- |
| `V` | Measured panel voltage |
| `I` | Measured panel current |
| `duty` | Converter duty-cycle value |
| `V_mpp_label` | Target maximum-power-point voltage |

Use consistent units and duty-cycle scaling across your dataset. The training script assumes a 0.02-second sampling interval and computes a 50-sample rolling window. It drops rows containing missing values and needs enough complete labeled rows for training and evaluation.

```bash
python src/train_model.py --csv path/to/solar_log_labeled.csv
```

The script uses the first 80% of processed rows for training and the remaining 20% for testing. It fits the scaler on training data, compares Ridge and MLP models, and writes a `vmpp_model_*.joblib` artifact.

The separate [features.py](src/features.py) utility offers additional rolling and optional vision features:

```bash
python src/features.py --in path/to/solar_log.csv --out dataset_features.csv
```

The current training script builds its own smaller feature set; it does not directly consume the full output of `features.py`.

## Repository guide

| Path | Purpose |
| --- | --- |
| [src/collect_video.py](src/collect_video.py) | Camera capture and brightness logging |
| [src/segmentation.py](src/segmentation.py) | Cloud masks, coverage estimates, and overlays |
| [src/extract_cloud_features.py](src/extract_cloud_features.py) | Saved-frame processing and CSV export |
| [src/features.py](src/features.py) | Electrical and optional vision feature engineering |
| [src/train_model.py](src/train_model.py) | Regression training, evaluation, and model export |
| [src/visualize.py](src/visualize.py) | Cloud coverage over time |
| [src/visualize_segmentation.py](src/visualize_segmentation.py) | Video mask and overlay viewer |
| [recordings/](recordings/) | Existing development recordings, frames, and CSV logs |

## Current limitations and next steps

- HSV thresholds depend on lighting and camera conditions; segmentation accuracy has not been established against labeled masks.
- Electrical training data, trained model artifacts, and reproducible performance results are not included.
- Model selection currently uses the test-set MAE; a separate validation set and untouched final test set would make evaluation stronger.
- Synchronizing vision and electrical measurements, evaluating across separate recording sessions, and connecting predictions to a controller remain future work.

Dependencies are listed from the source imports. Version compatibility and camera behavior still need to be checked on each target system.
