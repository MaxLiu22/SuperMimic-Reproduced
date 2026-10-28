"""Termination conditions for motion tracking.

Standard terminations: root (pelvis) height deviation and tilt; clip end as time-out.
Inverted terminations: for handstand and other inverted poses.

For handstand MVP, the robot forms a tripod with head + both hands contacting the ground.
Inverted terminations check that the pose deviation from reference is within bounds,
without assuming the robot must be upright.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ============================================================================
# Standard (upright) terminations
# ============================================================================

def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    """Terminate if root height deviates too much from reference (standing assumption)."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    """Terminate if root orientation deviates too much from reference (standing assumption).
    
    Compares the z-component of the gravity vector projected into the root frame.
    For standing poses, both reference and robot should have gravity pointing down
    in their body frames (negative z in world frame -> positive z in body frame).
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Terminate when the motion clip reaches its end."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end


# ============================================================================
# Inverted pose terminations (handstand, etc.)
# ============================================================================

def bad_inverted_contact(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_threshold: float = 0.35,
    tilt_threshold: float = 1.2,
) -> torch.Tensor:
    """Terminate for inverted poses (handstand) if pose deviation is too large.
    
    For handstand MVP with head + both_hands tripod support:
    - Relaxed height threshold (poses can be very different heights when inverted)
    - Relaxed tilt threshold (orientation deviation is compared to reference, not to upright)
    - Does NOT assume the robot must be upright
    
    The key difference from standing terminations is that we compare to the reference
    pose at each timestep, not to an assumed upright configuration.
    
    Args:
        env: The RL environment
        command_name: Name of the motion command
        height_threshold: Max allowed root height deviation from reference (m)
        tilt_threshold: Max allowed root tilt deviation from reference (rad-like metric)
    
    Returns:
        Boolean tensor indicating which envs should terminate
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    
    # Height deviation from reference (not from ground)
    height_dev = (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs()
    
    # Orientation deviation: compare gravity vector projection
    # This works for any pose since we compare robot to reference, not to upright
    g = cmd.robot.data.gravity_vec_w
    ref_g_body = quat_apply_inverse(cmd.root_quat_w, g)
    rob_g_body = quat_apply_inverse(cmd.robot_root_quat_w, g)
    
    # Full gravity vector comparison (not just z-component)
    # This handles arbitrary orientations including inverted
    grav_dev = torch.norm(ref_g_body - rob_g_body, dim=-1)
    
    # Terminate if either deviation exceeds threshold
    return (height_dev > height_threshold) | (grav_dev > tilt_threshold)


def bad_tripod_contact(
    env: ManagerBasedRlEnv,
    command_name: str,
    head_body_name: str = "head_link",
    hand_body_names: tuple[str, str] = ("left_hand_link", "right_hand_link"),
    min_contact_force: float = 1.0,
    require_all_contacts: bool = False,
) -> torch.Tensor:
    """Terminate if tripod contact (head + hands) is lost during handstand.
    
    For handstand MVP, the robot should maintain contact with the ground
    through at least 2 of the 3 support points (head + 2 hands).
    
    Note: This termination requires contact force information from the simulation.
    If contact forces are not available, use bad_inverted_contact instead.
    
    Args:
        env: The RL environment
        command_name: Name of the motion command (for robot reference)
        head_body_name: Name of the head body for contact detection
        hand_body_names: Names of left and right hand bodies
        min_contact_force: Minimum contact force to consider "in contact" (N)
        require_all_contacts: If True, require all 3 contacts. If False, require at least 2.
    
    Returns:
        Boolean tensor indicating which envs should terminate
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    robot = cmd.robot
    
    # Get body indices for contact detection
    try:
        head_idx = robot.body_names.index(head_body_name)
        left_hand_idx = robot.body_names.index(hand_body_names[0])
        right_hand_idx = robot.body_names.index(hand_body_names[1])
    except ValueError as e:
        # Body not found - this termination is not applicable for this robot
        # Return all False (don't terminate)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Check if contact force data is available
    if not hasattr(robot.data, "body_contact_force_w"):
        # No contact data - fall back to not terminating
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get contact forces for the tripod bodies
    contact_forces = robot.data.body_contact_force_w  # (N, num_bodies, 3)
    head_force = torch.norm(contact_forces[:, head_idx], dim=-1)
    left_hand_force = torch.norm(contact_forces[:, left_hand_idx], dim=-1)
    right_hand_force = torch.norm(contact_forces[:, right_hand_idx], dim=-1)
    
    # Count contacts
    head_contact = head_force > min_contact_force
    left_contact = left_hand_force > min_contact_force
    right_contact = right_hand_force > min_contact_force
    num_contacts = head_contact.long() + left_contact.long() + right_contact.long()
    
    if require_all_contacts:
        return num_contacts < 3
    else:
        return num_contacts < 2


def bad_anchor_deviation(
    env: ManagerBasedRlEnv,
    command_name: str,
    pos_threshold: float = 0.5,
    rot_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if anchor body (torso) deviates too much from reference.
    
    This is a general-purpose termination that works for any pose type
    by comparing the anchor body (typically torso) to the reference.
    
    Args:
        env: The RL environment
        command_name: Name of the motion command
        pos_threshold: Max allowed anchor position deviation (m)
        rot_threshold: Max allowed anchor rotation deviation (using quat error metric)
    
    Returns:
        Boolean tensor indicating which envs should terminate
    """
    from mjlab.utils.lab_api.math import quat_error_magnitude
    
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    
    # Position deviation
    pos_dev = torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1)
    
    # Rotation deviation
    rot_dev = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w)
    
    return (pos_dev > pos_threshold) | (rot_dev > rot_threshold)
