# MIRAGE-HRI

A laptop prototype that will turn webcam video into timestamped interaction events for a simulated robot consumer.

**Status: Phase 4.** The webcam opens, each frame gets a raw `hand_raised` label, and a temporal filter sends one gesture event to a simulated robot after the raise has held. Offline evaluation is not implemented. No measurements are claimed.

## Research question

Can temporal filtering and confidence-aware abstention improve the reliability of frame-level visual cues for downstream interaction, and what latency trade-off do they introduce?

This repository does not answer that question yet.

## What this version does

```text
Webcam -> MediaPipe Pose landmarks -> raw hand_raised label -> JSON event -> simulated robot
```

The hand-raise label is still computed one frame at a time and written to the log as a raw record. The robot does not hear those frames directly. A raise must stay `raised` for `filter.activation_ms` before a start event is sent. Any other label cancels that wait. The gesture ends after a non-raised label lasts `filter.release_ms`, and new raises are ignored for `filter.cooldown_ms` after that. `delay_ms` on the filtered start record is the wait from the first frame of the run that succeeded. `detector_score` remains the uncalibrated wrist-to-shoulder clearance.

For each body side, the wrist and shoulder must both be visible inside the frame. Image y grows downward. That side is raised when `shoulder.y - wrist.y` is at least `hand_raise.raise_margin`. The frame is `raised` if either side is raised, `not_raised` if at least one side is usable and neither is raised, and `insufficient` when neither side is usable. "Left" and "right" are the person's sides in the pose model, not the left and right edges of the picture. The overlay prints those landmarks' visibility values. Visibility and the raise margin are not probabilities.

## What Phase 1 still does

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
| `perception/hand_raise.py` | Raw wrist/shoulder hand-raise rule |
| `perception/camera.py` | Open and read the webcam |
| `perception/draw.py` | Draw landmarks, visibility, the raw label, and the robot response |
| `events/schema.py` | Timestamped event schema and raw-frame records |
| `events/consumer.py` | Simulated robot. Responds only to status `detected` |
| `events/log.py` | JSONL log of raw labels and filtered events |
| `events/temporal_filter.py` | Activation, release, and cooldown for one gesture |
| `demo_loop.py` | Keep running through unknown frames and short dropouts |
| `tests/` | Unit tests for pose status, the hand-raise rule, events, the filter, and the loop |

The session log is `logs/session.jsonl`. Raw rows and filtered rows are both in that file. It is not committed.

Planned and not built: offline evaluation, pilot report.

## Not in this version

Audio, gaze, touch sensors, model training, ROS 2, and any emotion, engagement, attention, or clinical labels.
