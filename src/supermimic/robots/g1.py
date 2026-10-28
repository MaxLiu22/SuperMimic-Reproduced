"""Unitree G1 robot configuration.

The G1 has 29 actuated revolute joints plus a floating base:
- 6 DOF per leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
- 3 DOF waist (yaw/roll/pitch)
- 7 DOF per arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)

MJCF asset path is resolved from mjlab if available, or from the
SUPERMIMIC_G1_MJCF environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from supermimic.robots.base import RobotConfig, register_robot


G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_BODY_NAMES: tuple[str, ...] = (
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
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "imu_in_pelvis",
)

G1_TRACKED_BODIES: tuple[str, ...] = (
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
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

G1_ACTION_SCALE: tuple[float, ...] = (
    0.25, 0.25, 0.25, 0.35, 0.15, 0.15,  # left leg
    0.25, 0.25, 0.25, 0.35, 0.15, 0.15,  # right leg
    0.25, 0.25, 0.25,                     # waist
    0.15, 0.15, 0.15, 0.15, 0.075, 0.075, 0.075,  # left arm
    0.15, 0.15, 0.15, 0.15, 0.075, 0.075, 0.075,  # right arm
)


def _try_mjlab_g1_path() -> Path | None:
    """Try to get G1 MJCF path from mjlab if installed."""
    try:
        import mjlab
        return Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    except ImportError:
        return None


@register_robot
@dataclass
class G1Config(RobotConfig):
    """Unitree G1 humanoid robot configuration."""
    
    @property
    def name(self) -> str:
        return "g1"
    
    @property
    def display_name(self) -> str:
        return "Unitree G1"
    
    @property
    def num_joints(self) -> int:
        return 29
    
    @property
    def joint_names(self) -> tuple[str, ...]:
        return G1_JOINT_NAMES
    
    @property
    def body_names(self) -> tuple[str, ...]:
        return G1_BODY_NAMES
    
    @property
    def tracked_bodies(self) -> tuple[str, ...]:
        return G1_TRACKED_BODIES
    
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
        return r"^(left|right)_foot[1-7]_collision$"
    
    @property
    def hand_bodies(self) -> tuple[str, ...]:
        return ("left_wrist_yaw_link", "right_wrist_yaw_link")
    
    @property
    def head_body(self) -> str:
        return "head"
    
    @property
    def mjcf_path_env_var(self) -> str:
        return "SUPERMIMIC_G1_MJCF"
    
    @property
    def default_mjcf_path(self) -> Path | None:
        return _try_mjlab_g1_path()
    
    @property
    def action_scale(self) -> tuple[float, ...]:
        return G1_ACTION_SCALE
