# MIRAGE-HRI pilot report

This is a pilot on a small set of consented adult trials. It is not a general validation.

## Trial count

- Held-out trials scored: 0
- Tuning trials kept out of the scores: 0
- Trials present but not scored: 0

## Conditions

Positive raises, a no-cue period, transitions, and harder distance, lighting, and occlusion trials are all in the schedule. The tuning trial uses the same seated raise and stays out of the table.

| Condition | Role | Instructed raises from the first frame |
| --- | --- | --- |
| positive | held_out | 3000–6000 ms, 12000–15000 ms |
| no_cue | held_out | none |
| transitions | held_out | 2000–4000 ms, 7000–9000 ms, 13000–16000 ms |
| distance | held_out | 4000–7000 ms |
| lighting | held_out | 4000–7000 ms |
| occlusion | held_out | 4000–7000 ms |
| positive_tuning | tuning | 3000–6000 ms |

## Configuration

These values were fixed before any pilot score. Baseline events are every raw `raised` frame. Filtered events are gesture starts. A match must fall within 1000 ms of the annotated start, and each annotation and each prediction is used once.

- tolerance_ms: 1000
- activation_ms: 300
- release_ms: 400
- cooldown_ms: 500
- raise_margin: 0.08
- duration_ms: 20000

## Results

Not measured. No confirmed held-out adult trial is available, so no pilot result is claimed.

## Limitations

- The sample is a pilot. It does not validate the system for other people, rooms, or cameras.
- Annotations are the instructed times, shifted to the first saved frame, and they count only after the participant confirms they followed that script.
- No participant video is stored. Compliance is not checked with a second view.
- `detector_score` is uncalibrated wrist-to-shoulder clearance, not a probability.
- The only cue is `hand_raised`.
- The hand-checked example under `data/hand_checked/` checks the scorer. It is not a pilot trial.
- Live logs in `logs/` have no annotations and are not scored.
- The tuning trial is reserved so a later threshold change can be tried without reusing the held-out trials. This report does not change the frozen settings.
