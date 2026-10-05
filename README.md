# MIRAGE-HRI

A laptop prototype that will turn webcam video into timestamped interaction events for a simulated robot consumer.

**Status: Phase 6.** The pilot protocol, held-out split, and report writer are in place. No confirmed adult trial is stored, so the pilot report claims no result.

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

Score the hand-checked example. A detection counts as a match only when the event is emitted within **1000 ms** of the annotated start. That window was chosen before the scores below were calculated, and the same window is used for the raw baseline and the filtered events. Each annotation and each prediction can match only once, so a second prediction in the same window is a false trigger. The baseline treats every raw `raised` frame as its own event. The filtered condition uses the gesture-start events.

```powershell
.\.venv\Scripts\python.exe evaluation\run_evaluation.py
```

On this example the baseline precision is 2/62 and the mean delay is 0 ms. The filtered condition precision is 1 and the mean delay is 300 ms. Those figures describe the constructed example only.

The pilot uses the same 1000 ms window and the same frozen filter (300 ms activation, 400 ms release, 500 ms cooldown). Held-out trials are the positive, no-cue, transition, distance, lighting, and occlusion conditions. `positive_tuning` is recorded and left out of the scores. The protocol is `data/PROTOCOL.md`. No video is saved.

```powershell
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition positive --consent
.\.venv\Scripts\python.exe -m evaluation.record_trial --confirm data\pilot\P01_positive
.\.venv\Scripts\python.exe -m evaluation.pilot_report
```

`reports/pilot.md` scores only confirmed adult held-out trials. Until those trials exist, it says the result is not measured. That report is a pilot, not a general validation.

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
| `evaluation/metrics.py` | Event precision, recall, F1, false triggers, and delay |
| `evaluation/run_evaluation.py` | Score the hand-checked trial |
| `data/hand_checked/` | Annotations and log with known answers |
| `data/PROTOCOL.md` | Consented-adult recording and annotation protocol |
| `evaluation/pilot.py` | Held-out pilot report |
| `evaluation/record_trial.py` | Record one trial without saving video |
| `reports/pilot.md` | Pilot report generated from saved trials |
| `tests/` | Unit tests for pose status, the hand-raise rule, events, the filter, metrics, the pilot gates, and the loop |

The session log is `logs/session.jsonl`. Raw rows and filtered rows are both in that file. It is not committed.

Planned and not built: portfolio packaging for a reviewer who was not in the room.

## Not in this version

Audio, gaze, touch sensors, model training, ROS 2, and any emotion, engagement, attention, or clinical labels.
