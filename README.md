# MIRAGE-HRI

A laptop prototype that will turn webcam video into timestamped interaction events for a simulated robot consumer.

**Status: Phase 1 only.** The webcam opens, body landmarks are drawn, and each frame is labeled `detected`, `unknown`, or `unavailable`. Hand-raise detection, temporal filtering, the robot consumer, and evaluation are not implemented. No measurements are claimed.

## Research question

Can temporal filtering and confidence-aware abstention improve the reliability of frame-level visual cues for downstream interaction, and what latency trade-off do they introduce?

This repository does not answer that question yet.

## What Phase 1 does

```text
Webcam -> MediaPipe Pose landmarks -> on-screen status
```

A frame is `detected` when the pose model returns at least `min_visible_landmarks` points whose own visibility field is at least `min_landmark_visibility`. That visibility value is passed through from the model. It is not a calibrated probability.

`unknown` means the frame was processed and the pose was missing or too incomplete to draw as a usable body. `unavailable` means the camera frame or the model step failed. Missing landmarks are left missing.

## Setup

Use Python 3.12, 64-bit. Create a virtual environment. The system-wide `pip` on the development machine used for this prototype was broken, so install into `.venv` rather than the base interpreter.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_pose_model.py
```

The model is Google's published Pose Landmarker lite model (float16, version 1). This project does not train it. See `models/README.md`.

## Run

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py
```

Quit the window with `q` or `Esc`.

Check a fixed number of frames without a window:

```powershell
.\.venv\Scripts\python.exe app.py --frames 60 --headless
```

The printed summary is a run log, not an evaluation result.

## Layout

| Path | Phase 1 role |
| --- | --- |
| `app.py` | Webcam demo |
| `config.yaml` | Camera, model, and visibility gate |
| `perception/pose_detector.py` | Landmark inference and frame status |
| `perception/camera.py` | Open and read the webcam |
| `perception/draw.py` | Draw landmarks that cleared the visibility gate |
| `demo_loop.py` | Keep running through unknown frames and short dropouts |
| `tests/` | Unit tests for status, camera failure, and the loop |

Planned and not built: `hand_raised` detection, event schema, temporal filter, simulated robot consumer, offline evaluation, pilot report.

## Not in this version

Audio, gaze, touch sensors, model training, ROS 2, and any emotion, engagement, attention, or clinical labels.
