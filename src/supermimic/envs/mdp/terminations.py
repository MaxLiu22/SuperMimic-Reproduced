"""Terminations for motion tracking: root height/tilt and inverted-pose support.

Original (standing) terminations: root (pelvis) height deviation and tilt vs reference.

Inverted-pose terminations (handstand MVP):
- For inverted poses, the robot is expected to be upside-down relative to standing.
- We track orientation relative to reference (which may be inverted).
- Contact tracking checks tripod stability (hands + head for handstand).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    """Terminate if root height deviates from reference by more than threshold."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    """Terminate if root tilt deviates from reference by more than threshold.

    This works for both standing and inverted poses because it compares the
    robot's projected gravity against the reference's projected gravity.
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def bad_inverted_orientation(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 0.8,
) -> torch.Tensor:
    """Terminate if orientation deviates from inverted reference.

    For handstand poses, the robot should be inverted (upside-down).
    This checks if the robot's up-vector deviation from reference exceeds threshold.
    Unlike bad_root_tilt, this explicitly expects and handles inverted orientations.
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w

    ref_up_in_ref = quat_apply_inverse(cmd.root_quat_w, -g)
    rob_up_in_rob = quat_apply_inverse(cmd.robot_root_quat_w, -g)

    ref_z = ref_up_in_ref[:, 2]
    rob_z = rob_up_in_rob[:, 2]

    return (ref_z - rob_z).abs() > threshold


def insufficient_contact(
    env: ManagerBasedRlEnv,
    contact_geom_regex: str,
    min_contacts: int = 2,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Terminate if fewer than min_contacts contact points are active.

    For handstand tripod: contact_geom_regex should match hands + head geoms.
    """
    robot = env.scene[entity_name]
    local_ids, _ = robot.find_geoms(contact_geom_regex)
    if len(local_ids) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    global_ids = robot.indexing.geom_ids[local_ids]

    contact_data = env.sim.data.contact
    geom1 = contact_data.geom1
    geom2 = contact_data.geom2

    n_envs = env.num_envs
    active_contacts = torch.zeros(n_envs, dtype=torch.long, device=env.device)

    for gid in global_ids:
        in_contact = ((geom1 == gid) | (geom2 == gid)).any(dim=-1) if geom1.dim() > 1 else ((geom1 == gid) | (geom2 == gid))
        if in_contact.dim() == 0:
            in_contact = in_contact.expand(n_envs)
        active_contacts += in_contact.long()

    return active_contacts < min_contacts


def bad_tripod_stability(
    env: ManagerBasedRlEnv,
    hand_geom_regex: str,
    head_geom_regex: str,
    min_hand_contacts: int = 1,
    require_head_contact: bool = True,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Terminate if tripod stability is lost (for handstand).

    Tripod = two hands + head. Checks:
    - At least min_hand_contacts hand geoms are in contact
    - Head is in contact (if require_head_contact is True)
    """
    robot = env.scene[entity_name]

    hand_ids, _ = robot.find_geoms(hand_geom_regex)
    head_ids, _ = robot.find_geoms(head_geom_regex)

    n_envs = env.num_envs
    device = env.device

    if len(hand_ids) == 0:
        hand_ok = torch.ones(n_envs, dtype=torch.bool, device=device)
    else:
        hand_global = robot.indexing.geom_ids[hand_ids]
        contact_data = env.sim.data.contact
        geom1 = contact_data.geom1
        geom2 = contact_data.geom2

        hand_contacts = torch.zeros(n_envs, dtype=torch.long, device=device)
        for gid in hand_global:
            in_contact = ((geom1 == gid) | (geom2 == gid)).any(dim=-1) if geom1.dim() > 1 else ((geom1 == gid) | (geom2 == gid))
            if in_contact.dim() == 0:
                in_contact = in_contact.expand(n_envs)
            hand_contacts += in_contact.long()
        hand_ok = hand_contacts >= min_hand_contacts

    if not require_head_contact or len(head_ids) == 0:
        head_ok = torch.ones(n_envs, dtype=torch.bool, device=device)
    else:
        head_global = robot.indexing.geom_ids[head_ids]
        contact_data = env.sim.data.contact
        geom1 = contact_data.geom1
        geom2 = contact_data.geom2

        head_in_contact = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for gid in head_global:
            in_contact = ((geom1 == gid) | (geom2 == gid)).any(dim=-1) if geom1.dim() > 1 else ((geom1 == gid) | (geom2 == gid))
            if in_contact.dim() == 0:
                in_contact = in_contact.expand(n_envs)
            head_in_contact |= in_contact
        head_ok = head_in_contact

    return ~(hand_ok & head_ok)


def adaptive_pose_termination(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_threshold: float = 0.25,
    tilt_threshold: float = 0.8,
    inverted_tilt_threshold: float = 0.8,
    inverted_gravity_z_threshold: float = -0.3,
) -> torch.Tensor:
    """Adaptive termination that switches between standing and inverted checks.

    Uses the reference pose's gravity projection to determine if the pose
    is inverted (handstand-like) or upright (standing-like), then applies
    the appropriate termination condition.

    Args:
        inverted_gravity_z_threshold: If reference gravity_z < this, pose is inverted.
            For standing, gravity_z ≈ -1. For handstand, gravity_z ≈ +1.
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w

    ref_grav_z = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    is_inverted = ref_grav_z > inverted_gravity_z_threshold

    height_bad = (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > height_threshold

    ref_proj = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob_proj = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]

    tilt_bad_standing = (ref_proj - rob_proj).abs() > tilt_threshold
    tilt_bad_inverted = (ref_proj - rob_proj).abs() > inverted_tilt_threshold

    tilt_bad = torch.where(is_inverted, tilt_bad_inverted, tilt_bad_standing)

    return height_bad | tilt_bad


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Terminate when clip ends (time-out)."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end
