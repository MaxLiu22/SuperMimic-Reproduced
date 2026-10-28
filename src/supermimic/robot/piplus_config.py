"""HighTorque Pi Plus robot configuration.

This configuration provides stubs for the Pi Plus humanoid robot,
designed for handstand (inverted pose) training. Asset paths are
parameterized via environment variables since Pi Plus model files
must NOT be committed to GitHub.

Pi Plus Specifications (from morphology):
- 22 actuated revolute joints + floating base
- qpos dim: 29 = 7 (freejoint) + 22 (joints)
- No wrist DOF
- Arms: ~±10 Nm torque limit
- Handstand MVP: head + both_hands tripod support
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Any

from supermimic.robot.robot_config import RobotConfig


# Pi Plus joint names (22 DOF, no wrist joints)
# ORDER MUST MATCH YOUR LOCAL MJCF - adjust as needed
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
    # Waist (3 DOF)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (4 DOF, no wrist)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    # Right arm (4 DOF, no wrist)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
)

# Bodies tracked for motion imitation (adjust to your MJCF body names)
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
    "right_shoulder_roll_link",
    "right_elbow_link",
    # Include head and hands for handstand contact tracking
    "head_link",
    "left_hand_link",
    "right_hand_link",
)


@dataclass
class PiPlusConfig(RobotConfig):
    """Configuration for the HighTorque Pi Plus humanoid robot.
    
    Pi Plus specifications:
    - 22 actuated joints (no wrist DOF)
    - qpos: 29 (7 freejoint + 22 joints)
    - qvel: 28 (6 freejoint + 22 joints)
    - Arm torque: ~±10 Nm
    
    IMPORTANT: Asset paths must be configured via environment variables.
    Do NOT commit any URDF, MJCF, or mesh files to GitHub.
    """
    
    name: str = field(default="piplus", init=False)
    
    # Allow overriding joint names via env var for local MJCF compatibility
    _joint_names_override: tuple[str, ...] | None = field(default=None, repr=False)
    _tracked_bodies_override: tuple[str, ...] | None = field(default=None, repr=False)
    
    @property
    def num_joints(self) -> int:
        return 22
    
    @property
    def joint_names(self) -> tuple[str, ...]:
        if self._joint_names_override:
            return self._joint_names_override
        # Check for env var override (comma-separated list)
        env_names = os.environ.get("SUPERMIMIC_PIPLUS_JOINT_NAMES")
        if env_names:
            return tuple(n.strip() for n in env_names.split(","))
        return PIPLUS_JOINT_NAMES
    
    @property
    def tracked_body_names(self) -> tuple[str, ...]:
        if self._tracked_bodies_override:
            return self._tracked_bodies_override
        env_names = os.environ.get("SUPERMIMIC_PIPLUS_TRACKED_BODIES")
        if env_names:
            return tuple(n.strip() for n in env_names.split(","))
        return PIPLUS_TRACKED_BODIES
    
    @property
    def anchor_body_name(self) -> str:
        return os.environ.get("SUPERMIMIC_PIPLUS_ANCHOR_BODY", "torso_link")
    
    @property
    def root_body_name(self) -> str:
        return os.environ.get("SUPERMIMIC_PIPLUS_ROOT_BODY", "pelvis")
    
    @property
    def foot_geom_regex(self) -> str:
        # Customize via env var for your MJCF geom naming
        return os.environ.get(
            "SUPERMIMIC_PIPLUS_FOOT_GEOM_REGEX",
            r"^(left|right)_foot.*_collision$"
        )
    
    @property
    def com_body_name(self) -> str:
        return os.environ.get("SUPERMIMIC_PIPLUS_COM_BODY", "torso_link")
    
    # ------------------------------------------------------------------ handstand support
    @property
    def head_body_name(self) -> str | None:
        return os.environ.get("SUPERMIMIC_PIPLUS_HEAD_BODY", "head_link")
    
    @property
    def hand_body_names(self) -> tuple[str, ...]:
        env_names = os.environ.get("SUPERMIMIC_PIPLUS_HAND_BODIES")
        if env_names:
            return tuple(n.strip() for n in env_names.split(","))
        return ("left_hand_link", "right_hand_link")
    
    @property
    def hand_geom_regex(self) -> str | None:
        return os.environ.get(
            "SUPERMIMIC_PIPLUS_HAND_GEOM_REGEX",
            r"^(left|right)_hand.*_collision$"
        )
    
    @property
    def head_geom_regex(self) -> str | None:
        return os.environ.get(
            "SUPERMIMIC_PIPLUS_HEAD_GEOM_REGEX",
            r"^head.*_collision$"
        )
    
    @property
    def supports_inverted_poses(self) -> bool:
        return True
    
    # ------------------------------------------------------------------ action scaling
    @property
    def action_scale(self) -> float:
        # Configurable via env var; default similar to G1
        scale_str = os.environ.get("SUPERMIMIC_PIPLUS_ACTION_SCALE", "0.25")
        return float(scale_str)
    
    @property
    def torque_limits(self) -> tuple[float, ...] | None:
        # Arms have ~±10 Nm limit
        # Return None to use model defaults, or specify per-joint
        limits_str = os.environ.get("SUPERMIMIC_PIPLUS_TORQUE_LIMITS")
        if limits_str:
            return tuple(float(x) for x in limits_str.split(","))
        return None
    
    # ------------------------------------------------------------------ termination thresholds (relaxed for handstand)
    @property
    def root_height_threshold(self) -> float:
        # More permissive for inverted poses
        return float(os.environ.get("SUPERMIMIC_PIPLUS_HEIGHT_THRESH", "0.35"))
    
    @property
    def root_tilt_threshold(self) -> float:
        # More permissive for inverted poses
        return float(os.environ.get("SUPERMIMIC_PIPLUS_TILT_THRESH", "1.2"))
    
    # ------------------------------------------------------------------ asset paths (REQUIRED via env var)
    @property
    def xml_path_env_var(self) -> str:
        return "SUPERMIMIC_PIPLUS_XML_PATH"
    
    @property
    def xml_path_default(self) -> str | None:
        # No default - must be configured locally
        return None
    
    @property
    def meshes_dir_env_var(self) -> str | None:
        return "SUPERMIMIC_PIPLUS_MESHES_DIR"
    
    def get_robot_cfg_func(self) -> Callable[[], Any]:
        """Return a function that creates the mjlab robot entity config.
        
        This requires a local implementation since Pi Plus is not in mjlab's asset zoo.
        Set SUPERMIMIC_PIPLUS_ROBOT_CFG_MODULE to your local module path.
        """
        module_path = os.environ.get("SUPERMIMIC_PIPLUS_ROBOT_CFG_MODULE")
        if module_path:
            import importlib
            module_name, func_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, func_name)
        
        # Fallback: return a stub that raises an informative error
        def _not_configured():
            raise NotImplementedError(
                "Pi Plus robot config not available. "
                "Set SUPERMIMIC_PIPLUS_ROBOT_CFG_MODULE to point to your local "
                "get_piplus_robot_cfg() function, e.g., 'mymodule.piplus:get_piplus_robot_cfg'"
            )
        return _not_configured


def create_piplus_env_template() -> str:
    """Generate a template .env file for Pi Plus configuration.
    
    Returns a string that can be written to a .env file.
    """
    return """\
# ============================================================
# Pi Plus Configuration Template
# ============================================================
# Copy this to your local .env file and configure paths.
# DO NOT commit this file or any robot assets to GitHub!

# REQUIRED: Path to your local Pi Plus MJCF/URDF
SUPERMIMIC_PIPLUS_XML_PATH=/path/to/your/piplus.xml

# Optional: Mesh directory if separate from MJCF
SUPERMIMIC_PIPLUS_MESHES_DIR=/path/to/meshes

# Optional: Custom robot config function (module.path:function_name)
SUPERMIMIC_PIPLUS_ROBOT_CFG_MODULE=mymodule.piplus:get_piplus_robot_cfg

# ============================================================
# Body/Joint Naming (if your MJCF differs from defaults)
# ============================================================
# Comma-separated joint names in qpos order
# SUPERMIMIC_PIPLUS_JOINT_NAMES=joint1,joint2,...

# Comma-separated tracked body names
# SUPERMIMIC_PIPLUS_TRACKED_BODIES=body1,body2,...

# Key body names
SUPERMIMIC_PIPLUS_ANCHOR_BODY=torso_link
SUPERMIMIC_PIPLUS_ROOT_BODY=pelvis
SUPERMIMIC_PIPLUS_COM_BODY=torso_link

# Handstand contact bodies
SUPERMIMIC_PIPLUS_HEAD_BODY=head_link
SUPERMIMIC_PIPLUS_HAND_BODIES=left_hand_link,right_hand_link

# ============================================================
# Geom Patterns (regex for collision geom matching)
# ============================================================
SUPERMIMIC_PIPLUS_FOOT_GEOM_REGEX=^(left|right)_foot.*_collision$
SUPERMIMIC_PIPLUS_HAND_GEOM_REGEX=^(left|right)_hand.*_collision$
SUPERMIMIC_PIPLUS_HEAD_GEOM_REGEX=^head.*_collision$

# ============================================================
# Control Parameters
# ============================================================
SUPERMIMIC_PIPLUS_ACTION_SCALE=0.25
# Per-joint torque limits (comma-separated, 22 values)
# SUPERMIMIC_PIPLUS_TORQUE_LIMITS=50,50,50,80,30,30,...

# ============================================================
# Termination Thresholds (relaxed for handstand)
# ============================================================
SUPERMIMIC_PIPLUS_HEIGHT_THRESH=0.35
SUPERMIMIC_PIPLUS_TILT_THRESH=1.2
"""
