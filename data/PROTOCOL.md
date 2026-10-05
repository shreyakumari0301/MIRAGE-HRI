# Pilot recording and annotation protocol

This protocol is for adults who agree to take part. It records one observable cue, `hand_raised`. It does not record audio. It does not save images or video. It does not label emotion, attention, engagement, or any clinical state.

The match window is **1000 ms**. The filter is **300 ms** to start, **400 ms** to release, and **500 ms** of cooldown. The wrist must clear the shoulder by **0.08** in normalized image coordinates. These values were fixed before any pilot score. Do not change them after looking at the report. One extra seated trial, `positive_tuning`, is recorded with the same settings and is left out of the scores.

Each scored trial lasts 20 seconds from the moment the window opens. The annotation clock starts at the first saved frame. A raise counts as annotated only during the intervals below.

| Condition | Role | What to do | Raises after the first frame |
| --- | --- | --- | --- |
| positive | held out | Sit about an arm's length away, in normal room light. | 3000–6000 ms and 12000–15000 ms |
| no_cue | held out | Same seat and light. Keep both hands down. | none |
| transitions | held out | Same seat and light. Raise, lower, and raise again. | 2000–4000, 7000–9000, and 13000–16000 ms |
| distance | held out | Step back about two meters. | 4000–7000 ms |
| lighting | held out | Dim the room and stay in frame. | 4000–7000 ms |
| occlusion | held out | Hold a folder so one shoulder is partly hidden. | 4000–7000 ms |
| positive_tuning | tuning, not scored | Repeat the seated raise. | 3000–6000 ms |

The window says `RAISE and hold` during a raise interval and `Hands down` otherwise. Outside those intervals the hand stays down, including the transitions between raises.

## Who can take part

The person is 18 or older and can see the window. They can stop a trial by pressing `q`. Stopping early leaves the trial unconfirmed, and an unconfirmed trial is not scored. Store a code such as `P01`, not a name.

## What gets saved

`data/pilot/<code>_<condition>/predictions.jsonl` stores raw labels and filtered events. `trial.yaml` stores the condition, the frozen settings, and the instructed times. `video_saved` stays false. Do not copy webcam pictures into this repository.

## Steps

1. The participant reads this page and agrees. The operator then passes `--consent`.
2. Record each condition once:

```powershell
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition positive --consent
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition no_cue --consent
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition transitions --consent
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition distance --consent
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition lighting --consent
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition occlusion --consent
.\.venv\Scripts\python.exe -m evaluation.record_trial --participant P01 --condition positive_tuning --consent
```

3. After each trial, the participant says whether their hand was up only while the window said `RAISE`. If it was, confirm that folder. If it was not, leave `followed_script` false and do not edit the times to match the log.

```powershell
.\.venv\Scripts\python.exe -m evaluation.record_trial --confirm data\pilot\P01_positive
```

4. Write the report from the saved files:

```powershell
.\.venv\Scripts\python.exe -m evaluation.pilot_report
```

The report scores held-out trials only. It compares the raw frame baseline with the filtered gesture starts on those same trials. It calls the outcome a pilot, not a general validation. Until a confirmed held-out trial exists, the report says the result is not measured.
