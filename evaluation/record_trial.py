"""Record one consented adult pilot trial. No image or video file is written."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.log_reader import stream_bounds_ms
from evaluation.protocol import (
    DURATION_MS,
    check_participant_code,
    condition_named,
    confirm_trial,
    prompt_at,
    shift_raises,
    trial_document,
    write_trial_file,
)
from perception.errors import ConfigError, MirageError


def record_trial(participant_code: str, condition_name: str, *, config_path: Path, destination: Path) -> Path:
    """Open the webcam, show the script, and store a log plus an unconfirmed trial file."""

    from app import parse_args, run
    from settings import load_settings

    code = check_participant_code(participant_code)
    condition = condition_named(condition_name)
    trial_id = f"{code}_{condition.name}"
    trial_dir = destination / trial_id
    if trial_dir.exists():
        raise ConfigError(f"{trial_dir} already exists. Refusing to overwrite a recording.")
    trial_dir.mkdir(parents=True)
    log_path = trial_dir / "predictions.jsonl"
    settings = load_settings(config_path)
    settings = replace(settings, events=replace(settings.events, log_path=log_path))
    started = time.perf_counter()

    def should_stop() -> bool:
        return (time.perf_counter() - started) * 1000 >= DURATION_MS

    def prompt() -> str:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return prompt_at(condition, elapsed_ms)

    print(condition.instruction, flush=True)
    print(
        "The window shows RAISE when the hand should be up. "
        "No video is saved. Press q to abort.",
        flush=True,
    )
    args = parse_args(["--config", str(config_path)])
    exit_code = run(settings, args, should_stop=should_stop, prompt=prompt)
    bounds = stream_bounds_ms(log_path)
    if bounds is None:
        raise ConfigError("The recording produced no raw frames.")
    document = trial_document(
        trial_id=trial_id,
        participant_code=code,
        condition=condition,
        events=shift_raises(bounds[0], condition),
        followed_script=False,
    )
    write_trial_file(trial_dir / "trial.yaml", document)
    print(
        f"Saved {trial_dir}. followed_script is false until the participant confirms the script.",
        flush=True,
    )
    if exit_code != 0:
        print("The camera stopped early. Leave this trial unconfirmed.", flush=True)
    return trial_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record or confirm one MIRAGE-HRI pilot trial")
    parser.add_argument("--participant", default=None, help="Code such as P01. Not a name.")
    parser.add_argument("--condition", default=None, help="Schedule name, such as positive or no_cue.")
    parser.add_argument("--consent", action="store_true", help="Confirm this adult agreed to the protocol.")
    parser.add_argument("--confirm", type=Path, default=None, help="Trial folder the participant says matches the script.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--destination", type=Path, default=Path("data/pilot"))
    args = parser.parse_args(argv)
    try:
        if args.confirm is not None:
            trial_id = confirm_trial(args.confirm)
            print(f"Confirmed {trial_id}.")
            return 0
        if not args.consent:
            raise ConfigError("Refusing to record without --consent.")
        if args.participant is None or args.condition is None:
            raise ConfigError("Recording needs --participant and --condition.")
        record_trial(
            args.participant,
            args.condition,
            config_path=args.config,
            destination=args.destination,
        )
        return 0
    except MirageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
