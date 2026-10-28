"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out.

Supports both standing (upright) and handstand (inverted) pose modes.
For handstand mode, the termination conditions are adapted:
- Height termination tracks the root body height relative to reference
- Tilt termination uses inverted gravity reference for handstand poses
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 0.25,
    mode: Literal["standing", "handstand", "reference"] = "reference",
) -> torch.Tensor:
    """Terminate if root height deviates too much from reference.
    
    Args:
        env: The environment.
        command_name: Name of the motion command.
        threshold: Maximum allowed height deviation in meters.
        mode: Termination mode:
            - "reference": Compare to reference height (default, works for any pose)
            - "standing": Traditional upright height check (terminates if too low)
            - "handstand": Inverted pose check (terminates if too low relative to reference)
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    
    if mode == "reference":
        # Compare robot root height to reference root height
        return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold
    elif mode == "standing":
        # Traditional standing: terminate if robot is too low
        return cmd.robot_root_pos_w[:, 2] < threshold
    elif mode == "handstand":
        # Handstand: use reference-relative check (allows inverted poses)
        # In handstand, root might be high above ground, so we track relative to reference
        return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold
    else:
        raise ValueError(f"Unknown mode: {mode}")


def bad_root_tilt(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 0.8,
    mode: Literal["standing", "handstand", "reference"] = "reference",
) -> torch.Tensor:
    """Terminate if root orientation deviates too much from reference.
    
    The tilt is measured as the difference in projected gravity z-component
    between the robot's root and the reference root.
    
    Args:
        env: The environment.
        command_name: Name of the motion command.
        threshold: Maximum allowed tilt deviation (cos angle difference).
            For standing: ~0.8 means ~37 degrees from vertical.
            For reference mode: same threshold but relative to reference pose.
        mode: Termination mode:
            - "reference": Compare to reference orientation (default, works for any pose)
            - "standing": Traditional upright check (terminates if tilted from vertical)
            - "handstand": No termination (disabled for inverted poses)
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    
    if mode == "handstand":
        # Disable tilt termination for handstand - let the motion reference guide it
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    g = cmd.robot.data.gravity_vec_w
    
    if mode == "reference":
        # Compare robot's projected gravity to reference's projected gravity
        ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
        rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
        return (ref - rob).abs() > threshold
    elif mode == "standing":
        # Traditional standing: compare to upright (gravity should point down in body frame)
        rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
        # For upright standing, projected gravity z should be close to -1
        return (rob + 1.0).abs() > threshold
    else:
        raise ValueError(f"Unknown mode: {mode}")


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Terminate when the motion clip ends."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end


# =============================================================================
# Handstand-specific terminations
# =============================================================================

def handstand_contact_check(
    env: ManagerBasedRlEnv,
    command_name: str,
    contact_bodies: tuple[str, ...] = ("head", "left_hand_link", "right_hand_link"),
    min_contacts: int = 2,
    max_no_contact_steps: int = 50,
) -> torch.Tensor:
    """Terminate if handstand contact is lost for too long.
    
    For a stable handstand (tripod: head + both hands), we expect at least
    `min_contacts` of the `contact_bodies` to be in contact with the ground.
    
    Args:
        env: The environment.
        command_name: Name of the motion command.
        contact_bodies: Bodies that should be in contact during handstand.
        min_contacts: Minimum number of contact bodies required.
        max_no_contact_steps: Number of steps without proper contact before termination.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    robot = cmd.robot
    
    # Get contact info from the simulation
    # This is a simplified check - in practice you'd check actual contact forces
    contact_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    for body_name in contact_bodies:
        if body_name in robot.body_names:
            body_idx = robot.body_names.index(body_name)
            body_height = robot.data.body_link_pos_w[:, body_idx, 2]
            # Rough contact check: body is close to ground (< 0.1m)
            in_contact = body_height < 0.1
            contact_count += in_contact.long()
    
    has_enough_contacts = contact_count >= min_contacts
    
    # For now, just check instantaneous contacts
    # A more sophisticated version would track contact history
    return ~has_enough_contacts


def inverted_pose_stability(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_height: float = 0.3,
    max_height: float = 2.0,
) -> torch.Tensor:
    """Terminate if the robot is in an unstable inverted pose.
    
    For handstand, the pelvis should be elevated above a minimum height
    but not unrealistically high (indicating a fall or explosion).
    
    Args:
        env: The environment.
        command_name: Name of the motion command.
        min_height: Minimum root height for valid handstand.
        max_height: Maximum root height (sanity check).
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    root_z = cmd.robot_root_pos_w[:, 2]
    
    # Terminate if root is too low (failed handstand) or too high (physics explosion)
    return (root_z < min_height) | (root_z > max_height)
