"""IsaacSession — context manager for Isaac Sim app lifecycle.

``IsaacSession`` accepts parsed args, launches Isaac Sim on ``__enter__``,
and closes it on ``__exit__``.  Environment creation happens inside the block.

Usage::

    from robolab_backend.isaac import parse_args, IsaacSession

    args = parse_args()          # must be before any isaaclab imports

    with IsaacSession(args) as session:
        from robolab_envs import get_env         # safe to import now
        env = get_env(args.task, seed=args.seed)
        runner.learn(env)
"""

from __future__ import annotations

import argparse


class IsaacSession:
    """Context manager that launches and owns the Isaac Sim app.

    Args:
        args_cli: Parsed args returned by :func:`~robolab_backend.isaac.startup.parse_args`.
        enable_cameras: Force camera rendering on (required for image tasks).
    """

    def __init__(self, args_cli: argparse.Namespace, enable_cameras: bool = True):
        self._args_cli = args_cli
        self._enable_cameras = enable_cameras
        self._simulation_app = None

    @property
    def simulation_app(self):
        """The running ``SimulationApp`` instance (available after ``__enter__``)."""
        return self._simulation_app

    def __enter__(self) -> "IsaacSession":
        from .startup import launch
        self._simulation_app = launch(self._args_cli, enable_cameras=self._enable_cameras)
        return self

    def __exit__(self, *args) -> bool:
        if self._simulation_app is not None:
            self._simulation_app.close()
        return False  # do not suppress exceptions

    def __repr__(self) -> str:
        return f"IsaacSession(args={self._args_cli!r})"
