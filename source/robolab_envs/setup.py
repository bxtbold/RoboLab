"""Installation script for the robolab_envs package."""

from setuptools import find_packages, setup

setup(
    name="robolab_envs",
    version="0.1.0",
    description="Robolab task/environment definitions for Isaac, MuJoCo, and real robots.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=1.0.0",
    ],
)
