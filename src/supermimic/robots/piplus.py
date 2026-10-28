"""HighTorque Pi Plus robot configuration for handstand MVP.

Pi Plus morphology:
- 22 actuated revolute joints + floating base
- qpos dim 29 = freejoint 7 + 22 joints
- No wrist DOF; arms ~±10 Nm torque
- Handstand MVP: head + both_hands tripod contact

IMPORTANT: MJCF/mesh files are NOT in this repository.
Set SUPERMIMIC_PIPLUS_MJCF env var to your local Pi Plus MJCF path.
Example: export SUPERMIMIC_PIPLUS_MJCF=/path/to/piplus/piplus.xml
"""

from __future__ import annotations

from supermimic.robots.config import ContactConfig, RobotConfig, register_robot_config

PIPLUS_JOINT_NAMES: tuple[str, ...] = (
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
    # Torso (3 DOF)
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
    # Head (1 DOF - optional, adjust if your model differs)
    # Note: If your Pi Plus has no head joint, remove this and adjust num_joints to 21
    # "head_pitch_joint",
)

PIPLUS_TRACKED_BODIES: tuple[str, ...] = (
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
    "head_link",
)

PIPLUS_HANDSTAND_SUPPORT_BODIES: tuple[str, ...] = (
    "left_hand_link",
    "right_hand_link",
    "head_link",
)


PIPLUS_CONFIG = RobotConfig(
    name="piplus",
    num_joints=22,
    nq=29,
    nv=28,
    joint_names=PIPLUS_JOINT_NAMES,
    tracked_bodies=PIPLUS_TRACKED_BODIES,
    anchor_body="torso_link",
    root_body="pelvis",
    action_scale=1.0,
    action_scales={
        "left_shoulder_pitch_joint": 0.8,
        "left_shoulder_roll_joint": 0.8,
        "left_shoulder_yaw_joint": 0.8,
        "left_elbow_joint": 0.8,
        "right_shoulder_pitch_joint": 0.8,
        "right_shoulder_roll_joint": 0.8,
        "right_shoulder_yaw_joint": 0.8,
        "right_elbow_joint": 0.8,
    },
    foot_geom_regex=r"^(left|right)_foot.*$",
    hand_geom_regex=r"^(left|right)_hand.*$",
    head_geom_regex=r"^head.*$",
    com_body="torso_link",
    standing_contact=ContactConfig(
        pose_type="standing",
        support_geom_regex=r"^(left|right)_foot.*$",
        height_reference_body="pelvis",
        min_contacts_for_stability=2,
    ),
    inverted_contact=ContactConfig(
        pose_type="inverted",
        support_bodies=PIPLUS_HANDSTAND_SUPPORT_BODIES,
        support_geom_regex=r"^((left|right)_hand.*|head.*)$",
        height_reference_body="pelvis",
        min_contacts_for_stability=2,
    ),
    mjcf_path_env_var="SUPERMIMIC_PIPLUS_MJCF",
    mjcf_relative_path="hightorque_piplus/piplus.xml",
    height_threshold=0.25,
    tilt_threshold=0.8,
    inverted_tilt_threshold=0.8,
)


def validate_piplus_mjcf() -> bool:
    """Check if Pi Plus MJCF is available locally."""
    path = PIPLUS_CONFIG.get_mjcf_path()
    return path is not None and path.exists()


def get_piplus_mjcf_instructions() -> str:
    """Return instructions for setting up Pi Plus MJCF locally."""
    return """
Pi Plus MJCF Setup Instructions
================================

The Pi Plus robot model files are NOT included in this repository.
To use Pi Plus with SuperMimic, you need to:

1. Obtain the Pi Plus URDF/MJCF from HighTorque (not publicly distributed).

2. Place the model files in one of these locations:
   a) Set environment variable:
      export SUPERMIMIC_PIPLUS_MJCF=/path/to/your/piplus.xml

   b) Or place in the default assets directory:
      ~/.supermimic/assets/hightorque_piplus/piplus.xml

3. Ensure all mesh files referenced by the MJCF are in the correct
   relative paths (usually ./meshes/ relative to the XML).

4. Verify your setup:
   python -c "from supermimic.robots.piplus import validate_piplus_mjcf; print(validate_piplus_mjcf())"

For handstand training, you'll also need a handstand motion clip in NPZ format.
See docs/PIPLUS_HANDSTAND.md for motion capture requirements.
"""


register_robot_config(PIPLUS_CONFIG)
