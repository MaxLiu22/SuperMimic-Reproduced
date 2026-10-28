"""HighTorque Pi Plus robot configuration.

The Pi Plus has 22 actuated revolute joints plus a floating base:
- 6 DOF per leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
- 3 DOF waist (yaw/roll/pitch)
- 4 DOF per arm (shoulder pitch/roll/yaw, elbow) — NO wrist DOF
- Arms have ~±10 Nm effort limits

For handstand MVP: head + both hands form a tripod support.

IMPORTANT: MJCF/mesh assets are NOT included in this repository.
Set SUPERMIMIC_PI_PLUS_MJCF environment variable to point to your
local Pi Plus MJCF file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from supermimic.robots.base import RobotConfig, register_robot


PI_PLUS_JOINT_NAMES: tuple[str, ...] = (
    # Left leg (6 DOF)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6 DOF)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3 DOF)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (4 DOF - no wrist)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # Right arm (4 DOF - no wrist)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

PI_PLUS_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "head",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_hand_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_hand_link",
)

PI_PLUS_TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_link",
)

PI_PLUS_ACTION_SCALE: tuple[float, ...] = (
    # Leg joints (same scale as G1 legs)
    0.25, 0.25, 0.25, 0.35, 0.15, 0.15,  # left leg
    0.25, 0.25, 0.25, 0.35, 0.15, 0.15,  # right leg
    # Waist
    0.25, 0.25, 0.25,
    # Arms (scaled for ~10 Nm effort limits)
    0.10, 0.10, 0.10, 0.10,  # left arm (no wrist)
    0.10, 0.10, 0.10, 0.10,  # right arm (no wrist)
)

PI_PLUS_EFFORT_LIMITS: tuple[float, ...] = (
    # Leg joints (higher torque for humanoid locomotion)
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,  # left leg
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,  # right leg
    # Waist
    88.0, 88.0, 88.0,
    # Arms (~10 Nm as per spec)
    10.0, 10.0, 10.0, 10.0,  # left arm
    10.0, 10.0, 10.0, 10.0,  # right arm
)


@register_robot
@dataclass
class PiPlusConfig(RobotConfig):
    """HighTorque Pi Plus humanoid robot configuration.
    
    22 actuated joints, no wrist DOF, ~10 Nm arm effort limits.
    Supports handstand with head+hands tripod contact.
    """
    
    @property
    def name(self) -> str:
        return "pi_plus"
    
    @property
    def display_name(self) -> str:
        return "HighTorque Pi Plus"
    
    @property
    def num_joints(self) -> int:
        return len(PI_PLUS_JOINT_NAMES)  # 23 = 6+6+3+4+4 (legs+waist+arms)
    
    @property
    def joint_names(self) -> tuple[str, ...]:
        return PI_PLUS_JOINT_NAMES
    
    @property
    def body_names(self) -> tuple[str, ...]:
        return PI_PLUS_BODY_NAMES
    
    @property
    def tracked_bodies(self) -> tuple[str, ...]:
        return PI_PLUS_TRACKED_BODIES
    
    @property
    def anchor_body(self) -> str:
        return "torso_link"
    
    @property
    def root_body(self) -> str:
        return "pelvis"
    
    @property
    def foot_bodies(self) -> tuple[str, ...]:
        return ("left_ankle_roll_link", "right_ankle_roll_link")
    
    @property
    def foot_geom_regex(self) -> str:
        return r"^(left|right)_foot.*_collision$"
    
    @property
    def hand_bodies(self) -> tuple[str, ...]:
        return ("left_hand_link", "right_hand_link")
    
    @property
    def head_body(self) -> str:
        return "head"
    
    @property
    def mjcf_path_env_var(self) -> str:
        return "SUPERMIMIC_PI_PLUS_MJCF"
    
    @property
    def default_mjcf_path(self) -> Path | None:
        # No default - must be configured via env var
        return None
    
    @property
    def action_scale(self) -> tuple[float, ...]:
        return PI_PLUS_ACTION_SCALE
    
    @property
    def effort_limits(self) -> tuple[float, ...]:
        return PI_PLUS_EFFORT_LIMITS
    
    @property
    def supports_handstand(self) -> bool:
        return True
    
    # Handstand-specific thresholds (more lenient for inverted poses)
    @property
    def root_height_threshold(self) -> float:
        return 0.35  # Slightly higher tolerance for inverted balance
    
    @property
    def root_tilt_threshold(self) -> float:
        return 0.9  # More lenient for handstand orientation deviations
