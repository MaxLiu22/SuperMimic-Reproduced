"""Kinematics utilities for humanoid robots.

This module provides:
- HumanoidKinematics: Generic batched differentiable FK for any MuJoCo humanoid
- G1Kinematics: Alias for HumanoidKinematics (backwards compatibility)
- Rotation utilities (quaternions, SO(3), etc.)
"""

from supermimic.kinematics.humanoid_fk import HumanoidKinematics, G1Kinematics
from supermimic.kinematics.rotations import (
    heading_quat,
    projected_gravity,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_angle_axis,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
    yaw_quat,
)

__all__ = [
    "HumanoidKinematics",
    "G1Kinematics",
    "heading_quat",
    "projected_gravity",
    "quat_apply",
    "quat_apply_inverse",
    "quat_canonical",
    "quat_from_angle_axis",
    "quat_from_rotvec",
    "quat_mul",
    "quat_normalize",
    "rot6d_from_quat",
    "rotvec_from_quat",
    "yaw_quat",
]
