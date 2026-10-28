"""Base robot configuration class and registry.

Robot-specific parameters are abstracted here to allow the same training
pipeline to work with different humanoids. Asset paths (MJCF, URDF, meshes)
are NOT embedded; they are resolved via environment variables or config
files so proprietary robot geometry stays local to the user.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar


@dataclass
class RobotConfig(ABC):
    """Base class for robot-specific configuration.
    
    Subclasses define joint names, body names, dimensions, and how to
    resolve asset paths. No mesh/URDF/MJCF files are embedded.
    """
    
    # Registry of all robot configs (populated by @register_robot)
    _registry: ClassVar[dict[str, type["RobotConfig"]]] = {}
    
    # ------------------------------------------------------------------ identity
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this robot (e.g. 'g1', 'pi_plus')."""
        ...
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g. 'Unitree G1', 'HighTorque Pi Plus')."""
        ...
    
    # ------------------------------------------------------------------ dimensions
    
    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of actuated revolute joints (excludes floating base)."""
        ...
    
    @property
    def nq(self) -> int:
        """Total qpos dimension: freejoint (7) + num_joints."""
        return 7 + self.num_joints
    
    @property
    def nv(self) -> int:
        """Total qvel dimension: freejoint (6) + num_joints."""
        return 6 + self.num_joints
    
    # ------------------------------------------------------------------ joint/body names
    
    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]:
        """Ordered tuple of actuated joint names (depth-first traversal order)."""
        ...
    
    @property
    @abstractmethod
    def body_names(self) -> tuple[str, ...]:
        """Ordered tuple of body/link names (excluding world)."""
        ...
    
    @property
    @abstractmethod
    def tracked_bodies(self) -> tuple[str, ...]:
        """Bodies tracked for motion imitation rewards."""
        ...
    
    @property
    @abstractmethod
    def anchor_body(self) -> str:
        """Body used as the anchor for relative pose computations (typically torso)."""
        ...
    
    @property
    @abstractmethod
    def root_body(self) -> str:
        """Root body of the kinematic tree (typically pelvis)."""
        ...
    
    # ------------------------------------------------------------------ contact bodies (task-specific)
    
    @property
    @abstractmethod
    def foot_bodies(self) -> tuple[str, ...]:
        """Bodies that contact ground in normal standing/walking."""
        ...
    
    @property
    def foot_geom_regex(self) -> str:
        """Regex pattern matching foot collision geoms for friction randomization."""
        return r"^(left|right)_foot.*_collision$"
    
    @property
    def hand_bodies(self) -> tuple[str, ...]:
        """Bodies that contact ground in handstand (hands/wrists)."""
        return ()  # Override in subclass if handstand is supported
    
    @property
    def head_body(self) -> str | None:
        """Head body for handstand tripod contact (head + both hands)."""
        return None  # Override in subclass if handstand is supported
    
    # ------------------------------------------------------------------ asset paths
    
    @property
    @abstractmethod
    def mjcf_path_env_var(self) -> str:
        """Environment variable name for the MJCF/XML asset path."""
        ...
    
    @property
    def default_mjcf_path(self) -> Path | None:
        """Default MJCF path if env var is not set (None = require env var)."""
        return None
    
    def resolve_mjcf_path(self) -> Path:
        """Resolve the MJCF asset path from env var or default.
        
        Raises:
            FileNotFoundError: If path is not configured or doesn't exist.
        """
        env_path = os.environ.get(self.mjcf_path_env_var)
        if env_path:
            p = Path(env_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"{self.mjcf_path_env_var}={env_path} does not exist"
                )
            return p
        if self.default_mjcf_path is not None:
            if not self.default_mjcf_path.exists():
                raise FileNotFoundError(
                    f"Default MJCF path {self.default_mjcf_path} does not exist. "
                    f"Set {self.mjcf_path_env_var} to override."
                )
            return self.default_mjcf_path
        raise FileNotFoundError(
            f"MJCF path not configured. Set environment variable {self.mjcf_path_env_var}"
        )
    
    # ------------------------------------------------------------------ action scaling
    
    @property
    @abstractmethod
    def action_scale(self) -> tuple[float, ...]:
        """Per-joint action scaling factors (effort/kp ratio)."""
        ...
    
    # ------------------------------------------------------------------ effort limits
    
    @property
    def effort_limits(self) -> tuple[float, ...] | None:
        """Per-joint effort (torque) limits in Nm, or None to use MJCF defaults."""
        return None
    
    # ------------------------------------------------------------------ termination thresholds
    
    @property
    def root_height_threshold(self) -> float:
        """Max deviation from reference root height before termination (meters)."""
        return 0.25
    
    @property
    def root_tilt_threshold(self) -> float:
        """Max tilt deviation (cos(angle) difference) before termination."""
        return 0.8
    
    # ------------------------------------------------------------------ handstand mode
    
    @property
    def supports_handstand(self) -> bool:
        """Whether this robot config supports inverted (handstand) poses."""
        return len(self.hand_bodies) > 0 and self.head_body is not None
    
    @property
    def handstand_contact_bodies(self) -> tuple[str, ...]:
        """Bodies expected to contact ground in handstand (head + hands tripod)."""
        if self.head_body is None:
            return self.hand_bodies
        return (self.head_body,) + self.hand_bodies


def register_robot(cls: type[RobotConfig]) -> type[RobotConfig]:
    """Class decorator to register a robot config in the global registry."""
    instance = cls()
    RobotConfig._registry[instance.name] = cls
    return cls


def get_robot_config(name: str) -> RobotConfig:
    """Get a robot config instance by name.
    
    Args:
        name: Robot identifier (e.g. 'g1', 'pi_plus')
        
    Returns:
        RobotConfig instance
        
    Raises:
        KeyError: If robot name is not registered
    """
    if name not in RobotConfig._registry:
        available = ", ".join(sorted(RobotConfig._registry.keys()))
        raise KeyError(f"Unknown robot '{name}'. Available: {available}")
    return RobotConfig._registry[name]()


def list_robots() -> list[str]:
    """List all registered robot names."""
    return sorted(RobotConfig._registry.keys())
