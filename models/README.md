# Pose model

Phase 1 uses Google's published MediaPipe Pose Landmarker lite model
(`pose_landmarker_lite`, float16, version 1). This repository does not train it.

The file is downloaded and is not committed:

```powershell
.\.venv\Scripts\python.exe scripts\download_pose_model.py
```

Source:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
