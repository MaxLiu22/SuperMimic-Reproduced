"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Supports multiple robot configurations (G1, Pi Plus) via the robot abstraction layer.

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.robots import RobotConfig, get_robot_config


class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


def get_robot_cfg_from_mjlab(robot_name: str):
    """Get mjlab robot config for known robots, or raise for custom robots."""
    if robot_name == "g1":
        from mjlab.asset_zoo.robots import get_g1_robot_cfg
        return get_g1_robot_cfg()
    else:
        # For other robots (like pi_plus), the user must provide the MJCF
        # mjlab doesn't have built-in configs for them
        raise ValueError(
            f"Robot '{robot_name}' is not available in mjlab. "
            f"For custom robots, use 'custom' robot_name and provide mjcf_path."
        )


def get_action_scale(robot_cfg: RobotConfig) -> tuple[float, ...]:
    """Get action scale for the robot."""
    return robot_cfg.action_scale


@dataclass
class SuperMimicEnvCfgArgs:
    """Configuration arguments for SuperMimic environment."""
    
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)
    
    # Robot configuration
    robot_name: str = "g1"  # 'g1', 'pi_plus', or 'custom'
    mjcf_path: str | None = None  # Required for custom robots or to override default
    
    # Pose mode for terminations
    pose_mode: Literal["standing", "handstand", "reference"] = "reference"


def _get_tracked_bodies(robot_cfg: RobotConfig) -> tuple[str, ...]:
    """Get tracked bodies from robot config."""
    return robot_cfg.tracked_bodies


def _get_anchor_body(robot_cfg: RobotConfig) -> str:
    """Get anchor body from robot config."""
    return robot_cfg.anchor_body


def _get_root_body(robot_cfg: RobotConfig) -> str:
    """Get root body from robot config."""
    return robot_cfg.root_body


def _get_foot_geom_regex(robot_cfg: RobotConfig) -> str:
    """Get foot geom regex from robot config."""
    return robot_cfg.foot_geom_regex


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    """Create environment configuration.
    
    Args:
        a: Environment configuration arguments.
        
    Returns:
        ManagerBasedRlEnvCfg ready to be used with ManagerBasedRlEnv.
    """
    # Get robot configuration
    robot_cfg = get_robot_config(a.robot_name)
    tracked_bodies = _get_tracked_bodies(robot_cfg)
    anchor_body = _get_anchor_body(robot_cfg)
    root_body = _get_root_body(robot_cfg)
    foot_geom_regex = _get_foot_geom_regex(robot_cfg)
    action_scale = get_action_scale(robot_cfg)
    
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **(
                    {
                        "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),
                        "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
                    }
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=anchor_body,
            root_body_name=root_body,
            body_names=tracked_bodies,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library,
            debug_vis=a.debug_vis,
            **a.extra_command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(anchor_body,)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_regex),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_regex),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    }

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(
            func=sm_term.bad_root_height,
            params={"command_name": "motion", "threshold": a.root_height_threshold, "mode": a.pose_mode},
        ),
        "root_tilt": TerminationTermCfg(
            func=sm_term.bad_root_tilt,
            params={"command_name": "motion", "threshold": a.root_tilt_threshold, "mode": a.pose_mode},
        ),
    }

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": foot_geom_regex, "com_body_name": anchor_body})

    # Get robot scene entity config
    if a.robot_name == "g1":
        from mjlab.asset_zoo.robots import get_g1_robot_cfg
        robot_scene_cfg = get_g1_robot_cfg()
    else:
        # For custom robots, we need to create a minimal config
        # The user is responsible for providing the MJCF path
        from mjlab.entity import ArticulationCfg
        
        mjcf_path = a.mjcf_path or robot_cfg.resolve_mjcf_path()
        robot_scene_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn_type="mjcf",
            mjcf_path=str(mjcf_path),
            actuator_names=(".*",),
        )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": robot_scene_cfg}),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=anchor_body, distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    """Create the SuperMimic tracking environment.
    
    Args:
        a: Environment configuration arguments.
        device: PyTorch device string.
        
    Returns:
        Configured ManagerBasedRlEnv instance.
    """
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)


# =============================================================================
# Convenience functions for handstand MVP
# =============================================================================

def make_handstand_env_cfg(
    npz_files: list[str],
    robot_name: str = "pi_plus",
    mjcf_path: str | None = None,
    num_envs: int = 4096,
    **kwargs,
) -> ManagerBasedRlEnvCfg:
    """Create environment configuration for handstand training.
    
    This is a convenience function that sets up the environment with
    handstand-appropriate defaults:
    - pose_mode="handstand" (disables standing-only terminations)
    - Increased root height threshold
    - Motion reference-relative terminations
    
    Args:
        npz_files: List of handstand motion npz files.
        robot_name: Robot config name (default: 'pi_plus').
        mjcf_path: Optional MJCF path override.
        num_envs: Number of parallel environments.
        **kwargs: Additional arguments passed to SuperMimicEnvCfgArgs.
        
    Returns:
        ManagerBasedRlEnvCfg configured for handstand training.
    """
    # Get robot config for thresholds
    robot_cfg = get_robot_config(robot_name)
    
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        robot_name=robot_name,
        mjcf_path=mjcf_path,
        pose_mode="handstand",
        root_height_threshold=robot_cfg.root_height_threshold,
        root_tilt_threshold=robot_cfg.root_tilt_threshold,
        **kwargs,
    )
    return make_env_cfg(args)


def make_handstand_env(
    npz_files: list[str],
    robot_name: str = "pi_plus",
    mjcf_path: str | None = None,
    device: str = "cuda:0",
    **kwargs,
) -> ManagerBasedRlEnv:
    """Create environment for handstand training.
    
    See make_handstand_env_cfg for details.
    """
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    cfg = make_handstand_env_cfg(npz_files, robot_name, mjcf_path, **kwargs)
    return ManagerBasedRlEnv(cfg, device=device)
