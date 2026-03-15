"""Path resolution shared by the evaluation scripts."""

from os.path import exists, join


class PathResolutionError(Exception):
    """Raised when a required model directory or checkpoint is missing."""


def resolve_ctrl_file(logdir: str | None) -> str:
    if not logdir:
        raise PathResolutionError("--logdir is required (where models are stored)")
    ctrl_file = join(logdir, "ctrl", "best.tar")
    if not exists(ctrl_file):
        raise PathResolutionError(f"controller was not trained: {ctrl_file} is missing")
    return ctrl_file
