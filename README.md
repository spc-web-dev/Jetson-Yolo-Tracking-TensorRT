# YOLO TensorRT Tracking and Jetson Camera Toolkit

A production-ready toolkit for real-time YOLO detection, multi-object tracking, and interactive object locking with TensorRT acceleration. Optimized for NVIDIA Jetson devices and discrete GPUs, with first-class support for the Jetson ARGUS camera via GStreamer.

## Features
- TensorRT-accelerated YOLO inference (Ultralytics)
- Multi-object tracking with BoT-SORT/ByteTrack
- Interactive object locking (mouse click, class-based, or ID-based)
- Real-time overlays: FPS, status, track trails
- Jetson ARGUS camera pipelines with multiple fallbacks
- Video/file stream support and optional saving of annotated video and tracks

## Repository Layout
- `yolo_track_lock.py`: Main script for detection, tracking, and interactive object locking across video files, streams, and Jetson camera.
- `export_tensorrt_engine.py`: Exports a YOLO model to a TensorRT `.engine` with conservative, Jetson-friendly settings.
- `jetson_camera_inference.py`: Minimal example to run YOLO TensorRT inference on the Jetson ARGUS camera, with performance overlays.
- `jetson_camera_test.py`: Quick camera diagnostics to find a working GStreamer pipeline for the ARGUS camera.
- `requirements.txt`: Pinned dependencies for x86_64 and Jetson (with platform markers).

## Clone and Setup
```bash
# Clone the repository
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# Create a virtual environment (recommended)
python -m venv .venv
# Windows
. .venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Notes for Jetson:
- Prefer system OpenCV with GStreamer (provided by JetPack). The requirements file avoids pip-OpenCV on aarch64.
- Install TensorRT and CUDA via JetPack. The Python package `nvidia-tensorrt` is generally not needed on Jetson.

## Model Preparation
You can use either a prebuilt TensorRT engine (`.engine`) or export one from a YOLO model (`.pt`).

### 1) Export a TensorRT Engine
```bash
python export_tensorrt_engine.py
```
This script loads `yolo11nvisrec71.pt` and creates `yolo11nvisrec71.engine` with static shapes and FP16, suitable for Jetson.

### 2) Use an Existing Engine
Place your engine file, e.g., `best.engine` or `yolo11nvisrec71.engine`, in the repository root (or pass a path via `--model`).

## Usage

### yolo_track_lock.py
Multi-object tracking with interactive object locking.

```bash
python yolo_track_lock.py \
  --model yolo11nvisrec71.engine \
  --source 0 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --imgsz 640 \
  --tracker botsort.yaml \
  --enable-lock \
  --show-tracks \
  --save-vid
```

Key arguments:
- `--model`: Path to YOLO model (`.engine` preferred, `.pt` also supported)
- `--source`: Video file, directory, camera index (e.g., `0`), or stream URL (`rtsp://`, `http://`, `https://`, `rtmp://`).
- `--conf-thres`, `--iou-thres`, `--max-det`, `--agnostic-nms`, `--classes`
- `--imgsz [H W]` or single `--imgsz 640`, `--half`, `--batch`, `--visualize`, `--augment`, `--retina-masks`
- `--tracker botsort.yaml|bytetrack.yaml`
- `--output-dir`, `--save-vid`, `--save-tracks`
- `--track-history`, `--show-tracks`, `--track-thickness`
- `--device` (e.g., `0`, `cpu`), `--display`, `--verbose`, `--show-fps`
- Locking: `--enable-lock`, `--lock-id <int>`, `--lock-class <name|id>`, `--highlight-color R,G,B`

Controls during runtime:
- `l`: toggle lock mode
- `c`: clear locked object
- `i`: enter a track ID (press Enter to confirm)
- Mouse click: lock object under cursor
- `p`/Space: pause/resume
- `s`: save current frame
- `q`: quit

Examples:
- From a video file:
  ```bash
  python yolo_track_lock.py --model yolo11nvisrec71.engine --source sample.mp4 --save-vid --enable-lock
  ```
- From a USB camera:
  ```bash
  python yolo_track_lock.py --model yolo11nvisrec71.engine --source 0 --show-tracks --enable-lock
  ```
- From Jetson ARGUS camera:
  ```bash
  python yolo_track_lock.py --model yolo11nvisrec71.engine --source jetson --enable-lock --display
  ```

### jetson_camera_inference.py
Minimal camera inference with FPS overlays (TensorRT engine required):
```bash
python jetson_camera_inference.py
```
If loading fails, ensure `yolo11n.engine` exists or export via `export_tensorrt_engine.py`.

### jetson_camera_test.py
Probe for a working GStreamer pipeline on Jetson and show a short preview:
```bash
python jetson_camera_test.py
```
On success, it prints a working pipeline and suggests running `jetson_camera_inference.py`.

### export_tensorrt_engine.py
Export a YOLO model to TensorRT with static shapes and FP16:
```bash
python export_tensorrt_engine.py
```
Outputs: `yolo11nvisrec71.engine` in the project root.

## Outputs
- Annotated video: `<source_name>_locked.mp4` (when `--save-vid`)
- Track data (CSV): `<source_name>_tracks.txt` (when `--save-tracks`)
- Run summary: `<source_name>_tracking_summary.txt`

## Tips and Troubleshooting
- If OpenCV fails to open ARGUS pipelines on Jetson, ensure no `nvgstcapture-1.0` process is running and OpenCV is built with GStreamer.
- If FPS is low, reduce `--imgsz`, increase `--vid-stride`, or use FP16 (`--half`).
- For trackers, ensure dependencies are installed (see `requirements.txt`, includes `lapx`, `scipy`).
- If `.engine` loading fails, regenerate the engine to match your device and TensorRT version.

## License
This repository is provided for research and production use. See source headers for details. 