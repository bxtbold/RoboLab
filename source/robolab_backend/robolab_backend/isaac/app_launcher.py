"""Thin wrapper around IsaacLab's AppLauncher."""

from __future__ import annotations

import argparse


def add_launcher_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Append IsaacLab AppLauncher args to *parser* in place and return it."""
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser


def create_launcher(args_cli: argparse.Namespace):
    """Instantiate :class:`isaaclab.app.AppLauncher` from parsed args.

    Returns the launcher instance (call ``.app`` to get the SimulationApp).
    """
    from isaaclab.app import AppLauncher
    return AppLauncher(args_cli)
