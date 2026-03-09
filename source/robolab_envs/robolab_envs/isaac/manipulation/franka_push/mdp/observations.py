"""Observation functions for Franka push tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, FrameTransformer
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Camera observations
##

def camera_image_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
) -> torch.Tensor:
    """RGB image from a camera sensor.

    Returns:
        Tensor of shape ``(num_envs, H, W, 3)``.
    """
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    rgb = sensor.data.output[data_type]
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    return rgb[..., :3]


##
# Proprioception observations
##

def ee_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector XYZ position relative to robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector linear velocity in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_body_idx = robot.find_bodies("panda_hand")[0][0]
    ee_vel_w = robot.data.body_lin_vel_w[:, ee_body_idx, :]
    return quat_rotate_inverse(robot.data.root_quat_w, ee_vel_w)
