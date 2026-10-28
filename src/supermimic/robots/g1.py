"""Unitree G1 robot configuration.

This preserves the original SuperMimic G1 hardcoding as an explicit configuration.
G1 has 29 actuated revolute joints + floating base (nq=36, nv=35).
"""

from __future__ import annotations

from supermimic.robots.config import ContactConfig, RobotConfig, register_robot_config

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

G1_CONFIG = RobotConfig(
    name="g1",
    num_joints=29,
    nq=36,
    nv=35,
    joint_names=G1_JOINT_NAMES,
    tracked_bodies=G1_TRACKED_BODIES,
    anchor_body="torso_link",
    root_body="pelvis",
    action_scale=1.0,
    foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
    hand_geom_regex=r"^(left|right)_hand_.*$",
    head_geom_regex=r"^head_.*$",
    com_body="torso_link",
    standing_contact=ContactConfig(
        pose_type="standing",
        support_geom_regex=r"^(left|right)_foot[1-7]_collision$",
        height_reference_body="pelvis",
        min_contacts_for_stability=2,
    ),
    inverted_contact=ContactConfig(
        pose_type="inverted",
        support_bodies=("left_wrist_yaw_link", "right_wrist_yaw_link", "head_link"),
        support_geom_regex=r"^((left|right)_hand_.*|head_.*)$",
        height_reference_body="pelvis",
        min_contacts_for_stability=2,
    ),
    mjcf_path_env_var="SUPERMIMIC_G1_MJCF",
    mjcf_relative_path="unitree_g1/xmls/g1.xml",
    height_threshold=0.25,
    tilt_threshold=0.8,
    inverted_tilt_threshold=0.8,
)


def get_g1_action_scale() -> float:
    """Return G1 action scale (for backward compatibility with mjlab imports)."""
    return G1_CONFIG.action_scale


def _get_mjlab_g1_robot_cfg():
    """Lazy import of mjlab's G1 robot config for backward compatibility."""
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    return get_g1_robot_cfg()


G1_CONFIG.mjlab_robot_cfg_fn = _get_mjlab_g1_robot_cfg

register_robot_config(G1_CONFIG)
