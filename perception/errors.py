"""Expected failures for the Phase 1 webcam and pose demo."""


class MirageError(Exception):
    """Failure the demo can report without a stack trace."""


class CameraUnavailableError(MirageError):
    """The webcam could not be opened or stayed unreadable."""


class PoseModelError(MirageError):
    """The pose model is missing or could not be loaded."""


class ConfigError(MirageError):
    """config.yaml is missing a required value or has an invalid one."""
