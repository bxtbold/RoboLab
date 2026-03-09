"""Installation script for the robolab_backend package."""

from setuptools import find_packages, setup

setup(
    name="robolab_backend",
    version="0.1.0",
    description="Robolab simulator/hardware lifecycle backend (Isaac, MuJoCo, real).",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
)
