"""Isaac Sim startup utilities.

Two-step launch pattern — keep them separate so scripts can parse args
before the expensive sim launch, and so the launch step is explicit:

    args = parse_args()
    sim_app = launch(args)

    # --- all isaaclab / omni imports go here ---
    import robolab_envs  # registers gym tasks
"""

from __future__ import annotations

import argparse
import sys


def parse_args(parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    """Parse CLI args including all IsaacLab AppLauncher args.

    Args:
        parser: Optional existing parser to extend.  If ``None``, a default
            one is created.

    Returns:
        Parsed :class:`argparse.Namespace`.  Hydra / extra args are forwarded
        back into ``sys.argv`` for downstream consumers.
    """
    from .app_launcher import add_launcher_args

    if parser is None:
        parser = argparse.ArgumentParser(description="Robolab IsaacLab runner.")

    add_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Forward remaining args (e.g. Hydra overrides) back to sys.argv
    sys.argv = [sys.argv[0]] + hydra_args

    return args_cli


def launch(args_cli: argparse.Namespace, enable_cameras: bool = True):
    """Launch Isaac Sim.

    **Must be called before any ``isaaclab`` / ``omni`` imports.**

    Args:
        args_cli: Parsed args returned by :func:`parse_args`.
        enable_cameras: Force camera rendering on.  Required for image-based
            tasks.  Overrides the CLI value when ``True``.

    Returns:
        The running ``omni.isaac.kit.SimulationApp`` instance.
    """
    from .app_launcher import create_launcher
    args_cli.enable_cameras = enable_cameras
    launcher = create_launcher(args_cli)
    return launcher.app
