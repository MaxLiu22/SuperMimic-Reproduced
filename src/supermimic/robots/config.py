"""Robot configuration dataclass and registry.

IMPORTANT: Robot model files (URDF, MJCF, meshes, textures) are NOT committed
to this repository. All asset paths must be parameterized via environment
variables or configuration files that point to local installations.

Environment variables:
- SUPERMIMIC_ROBOT_ASSETS: Base directory for robot assets (default: ~/.supermimic/assets)
- SUPERMIMIC_<ROBOT>_MJCF: Override MJCF path for specific robot (e.g., SUPERMIMIC_PIPLUS_MJCF)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


@dataclass
class ContactConfig:
    """Contact configuration for pose-specific termination conditions.

    For standing poses: feet are the primary contact bodies.
    For handstands: hands + head form a tripod contact.
    """

    pose_type: Literal["standing", "inverted", "custom"] = "standing"

    support_bodies: tuple[str, ...] = ()
    support_geom_regex: str | None = None
    height_reference_body: str = "pelvis"
    min_contacts_for_stability: int = 2


@dataclass
class RobotConfig:
    """Complete robot configuration for SuperMimic.

    This dataclass encapsulates all robot-specific parameters needed by:
    - Forward kinematics (FK)
    - MDP observations/rewards/terminations
    - Motion conversion and retargeting
    - Action spaces and limits
    """

    name: str
    num_joints: int
    nq: int
    nv: int

    joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    anchor_body: str
    root_body: str = "pelvis"

    action_scale: float = 1.0
    action_scales: dict[str, float] = field(default_factory=dict)

    foot_geom_regex: str | None = None
    hand_geom_regex: str | None = None
    head_geom_regex: str | None = None

    com_body: str = "torso_link"

    standing_contact: ContactConfig = field(default_factory=lambda: ContactConfig(
        pose_type="standing",
        support_geom_regex=r"^(left|right)_foot.*$",
        min_contacts_for_stability=2,
    ))
    inverted_contact: ContactConfig | None = None

    mjcf_path_env_var: str | None = None
    mjcf_relative_path: str | None = None

    mjlab_robot_cfg_fn: Callable | None = None

    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    inverted_tilt_threshold: float = 0.8

    def __post_init__(self):
        assert len(self.joint_names) == self.num_joints, \
            f"joint_names length {len(self.joint_names)} != num_joints {self.num_joints}"
        assert self.nq == 7 + self.num_joints, \
            f"nq {self.nq} should be 7 (freejoint) + {self.num_joints} joints"
        assert self.nv == 6 + self.num_joints, \
            f"nv {self.nv} should be 6 (freejoint) + {self.num_joints} joints"

    @property
    def state_dim(self) -> int:
        return 3 + 4 + self.num_joints + 3 + 3 + self.num_joints

    @property
    def feat_dim(self) -> int:
        return 1 + 6 + 3 + 3 + self.num_joints + self.num_joints

    @property
    def delta_dim(self) -> int:
        return 3 + 3 + self.num_joints + 3 + 3 + self.num_joints

    @property
    def xi_dim(self) -> int:
        return 1 + self.num_joints + 3

    def get_mjcf_path(self) -> Path | None:
        """Resolve MJCF path from environment or default locations.

        Priority:
        1. Robot-specific env var (e.g., SUPERMIMIC_PIPLUS_MJCF)
        2. Base assets dir + relative path
        3. None (caller should handle missing asset)
        """
        if self.mjcf_path_env_var:
            env_path = os.environ.get(self.mjcf_path_env_var)
            if env_path:
                p = Path(env_path)
                if p.exists():
                    return p

        base_dir = os.environ.get("SUPERMIMIC_ROBOT_ASSETS", str(Path.home() / ".supermimic" / "assets"))
        if self.mjcf_relative_path:
            p = Path(base_dir) / self.mjcf_relative_path
            if p.exists():
                return p

        return None

    def get_action_scale(self, joint_name: str | None = None) -> float:
        """Get action scale for a joint (or default)."""
        if joint_name and joint_name in self.action_scales:
            return self.action_scales[joint_name]
        return self.action_scale


_ROBOT_REGISTRY: dict[str, RobotConfig] = {}


def register_robot_config(config: RobotConfig) -> None:
    """Register a robot configuration."""
    _ROBOT_REGISTRY[config.name] = config


def get_robot_config(name: str) -> RobotConfig:
    """Get a registered robot configuration by name."""
    if name not in _ROBOT_REGISTRY:
        available = list(_ROBOT_REGISTRY.keys())
        raise KeyError(f"Robot '{name}' not registered. Available: {available}")
    return _ROBOT_REGISTRY[name]


def list_robots() -> list[str]:
    """List all registered robot names."""
    return list(_ROBOT_REGISTRY.keys())
