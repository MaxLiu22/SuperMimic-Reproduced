"""Differentiable batched forward kinematics for the Unitree G1 (torch).

DEPRECATED: Use HumanoidKinematics from humanoid_fk.py instead.
This module is kept for backwards compatibility.

Built from a compiled ``mujoco.MjModel`` of ``g1.xml`` so the tree, joint axes,
anchors and ``qpos0`` are exactly MuJoCo's. Reproduces, for every body:

* ``xpos`` / ``xquat``                      (link frame origin pose, world)
* link-frame-origin linear velocity, world  (== mjlab ``body_link_lin_vel_w``)
* angular velocity, world                   (== mjlab ``body_link_ang_vel_w``)

given ``qpos`` (36: root pos, root quat wxyz, 29 hinges) and ``qvel`` (35: root
linear velocity in world, root angular velocity in **body** frame, 29 hinge rates),
i.e. MuJoCo's free-joint conventions.
"""

from __future__ import annotations

# Re-export from the generic implementation for backwards compatibility
from supermimic.kinematics.humanoid_fk import HumanoidKinematics as G1Kinematics

__all__ = ["G1Kinematics"]
