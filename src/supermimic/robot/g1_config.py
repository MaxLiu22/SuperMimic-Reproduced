"""Unitree G1 robot configuration.

This configuration preserves all the original hardcoded values from the codebase,
now organized in one place for the G1 humanoid robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from supermimic.robot.robot_config import RobotConfig


# G1 joint names in qpos order (after the 7-DOF freejoint)
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

# Bodies tracked for motion imitation
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


@dataclass
class G1Config(RobotConfig):
    """Configuration for the Unitree G1 humanoid robot.
    
    G1 specifications:
    - 29 actuated joints (6 per leg, 3 waist, 7 per arm including wrist)
    - qpos: 36 (7 freejoint + 29 joints)
    - qvel: 35 (6 freejoint + 29 joints)
    """
    
    name: str = field(default="g1", init=False)
    
    @property
    def num_joints(self) -> int:
        return 29
    
    @property
    def joint_names(self) -> tuple[str, ...]:
        return G1_JOINT_NAMES
    
    @property
    def tracked_body_names(self) -> tuple[str, ...]:
        return G1_TRACKED_BODIES
    
    @property
    def anchor_body_name(self) -> str:
        return "torso_link"
    
    @property
    def root_body_name(self) -> str:
        return "pelvis"
    
    @property
    def foot_geom_regex(self) -> str:
        return r"^(left|right)_foot[1-7]_collision$"
    
    @property
    def com_body_name(self) -> str:
        return "torso_link"
    
    @property
    def action_scale(self) -> float:
        # From mjlab.asset_zoo.robots.G1_ACTION_SCALE
        return 0.25
    
    @property
    def xml_path_env_var(self) -> str:
        return "SUPERMIMIC_G1_XML_PATH"
    
    @property
    def xml_path_default(self) -> str | None:
        # G1 has a default path via mjlab
        try:
            from supermimic.robot.g1_config import _get_g1_xml_path_from_mjlab
            return str(_get_g1_xml_path_from_mjlab())
        except ImportError:
            return None
    
    def get_robot_cfg_func(self) -> Callable[[], Any]:
        from mjlab.asset_zoo.robots import get_g1_robot_cfg
        return get_g1_robot_cfg


def _get_g1_xml_path_from_mjlab():
    """Get the G1 XML path from mjlab's asset zoo."""
    import mjlab
    from pathlib import Path
    return Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def get_g1_action_scale() -> float:
    """Get the G1 action scale from mjlab."""
    try:
        from mjlab.asset_zoo.robots import G1_ACTION_SCALE
        return G1_ACTION_SCALE
    except ImportError:
        return 0.25  # Fallback default
