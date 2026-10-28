"""Differentiable batched forward kinematics for the Unitree G1 (torch).

BACKWARD COMPATIBILITY: This module re-exports G1Kinematics from the generalized
robot_fk module. For new code, use RobotKinematics from supermimic.kinematics.robot_fk.

Built from a compiled ``mujoco.MjModel`` of ``g1.xml`` so the tree, joint axes,
anchors and ``qpos0`` are exactly MuJoCo's. Reproduces, for every body:

* ``xpos`` / ``xquat``                      (link frame origin pose, world)
* link-frame-origin linear velocity, world  (== mjlab ``body_link_lin_vel_w``)
* angular velocity, world                   (== mjlab ``body_link_ang_vel_w``)

given ``qpos`` (36: root pos, root quat wxyz, 29 hinges) and ``qvel`` (35: root
linear velocity in world, root angular velocity in **body** frame, 29 hinge rates),
i.e. MuJoCo's free-joint conventions.

Quaternion math is element-wise (no 3x3 matmuls) so results are unaffected by TF32.
Position FK follows ``mj_kinematics`` (hinge rotates about its anchor by
``qpos - qpos0``); velocities are propagated down the tree:

    w_b = w_parent + axis_w * qdot
    v_b = v_parent + w_parent x (p_b - p_parent) + (axis_w * qdot) x (p_b - anchor_w)

which equals MuJoCo's ``cvel`` transported to the body origin (``mj_objectVelocity``
with ``flg_local=0``).
"""

from supermimic.kinematics.robot_fk import RobotKinematics

G1Kinematics = RobotKinematics
