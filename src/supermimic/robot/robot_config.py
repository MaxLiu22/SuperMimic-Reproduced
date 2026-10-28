"""Base robot configuration class.

All robot-specific parameters that affect MDP dimensions, kinematics,
termination conditions, and asset loading are defined here.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any


@dataclass
class RobotConfig(ABC):
    """Abstract base class for robot configurations.
    
    Defines all robot-specific parameters needed by SuperMimic:
    - Kinematic dimensions (joints, qpos/qvel sizes)
    - Body/joint naming conventions
    - Asset paths (parameterized via env vars for private assets)
    - Action scaling and limits
    - Termination thresholds
    """
    
    # ------------------------------------------------------------------ identification
    name: str = field(init=False)
    
    # ------------------------------------------------------------------ kinematic dimensions
    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of actuated joints (hinges)."""
        ...
    
    @property
    def nq(self) -> int:
        """Generalized position dimension: 7 (freejoint) + num_joints."""
        return 7 + self.num_joints
    
    @property
    def nv(self) -> int:
        """Generalized velocity dimension: 6 (freejoint) + num_joints."""
        return 6 + self.num_joints
    
    @property
    def state_dim(self) -> int:
        """Full state dimension: nq + nv."""
        return self.nq + self.nv
    
    @property
    def feat_dim(self) -> int:
        """Canonical feature dimension (same as state_dim for humanoids)."""
        return self.state_dim
    
    @property
    def delta_dim(self) -> int:
        """Network output dimension: state_dim - 1 (quaternion -> rotvec)."""
        return self.state_dim - 1
    
    @property
    def xi_dim(self) -> int:
        """Simulation context dimension: 1 (friction) + num_joints (encoder bias) + 3 (COM offset)."""
        return 1 + self.num_joints + 3
    
    @property
    def ref_dim(self) -> int:
        """Reference state dimension (same as state_dim)."""
        return self.state_dim
    
    @property
    def goal_feat_dim(self) -> int:
        """Goal feature dimension: joint_pos + joint_vel + grav(3) + ang_b(3) + lin_h(3) + height(1)."""
        return self.num_joints * 2 + 10
    
    # ------------------------------------------------------------------ body/joint names
    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]:
        """Ordered joint names matching qpos layout."""
        ...
    
    @property
    @abstractmethod
    def tracked_body_names(self) -> tuple[str, ...]:
        """Body names tracked for motion imitation rewards."""
        ...
    
    @property
    @abstractmethod
    def anchor_body_name(self) -> str:
        """Anchor body for relative pose computations (usually torso)."""
        ...
    
    @property
    @abstractmethod
    def root_body_name(self) -> str:
        """Root body (pelvis/base) for termination checks."""
        ...
    
    @property
    @abstractmethod
    def foot_geom_regex(self) -> str:
        """Regex matching foot collision geoms for friction randomization."""
        ...
    
    @property
    @abstractmethod
    def com_body_name(self) -> str:
        """Body used for COM offset randomization."""
        ...
    
    # ------------------------------------------------------------------ handstand support
    @property
    def head_body_name(self) -> str | None:
        """Head body name for handstand contact detection. None if not applicable."""
        return None
    
    @property
    def hand_body_names(self) -> tuple[str, ...]:
        """Hand/wrist body names for handstand contact detection."""
        return ()
    
    @property
    def hand_geom_regex(self) -> str | None:
        """Regex matching hand collision geoms. None if not defined."""
        return None
    
    @property
    def head_geom_regex(self) -> str | None:
        """Regex matching head collision geoms. None if not defined."""
        return None
    
    # ------------------------------------------------------------------ action scaling
    @property
    @abstractmethod
    def action_scale(self) -> float:
        """Action scale factor for joint position control."""
        ...
    
    @property
    def torque_limits(self) -> tuple[float, ...] | None:
        """Per-joint torque limits (Nm). None uses model defaults."""
        return None
    
    # ------------------------------------------------------------------ termination thresholds
    @property
    def root_height_threshold(self) -> float:
        """Maximum allowed root height deviation from reference (m)."""
        return 0.25
    
    @property
    def root_tilt_threshold(self) -> float:
        """Maximum allowed root tilt deviation from reference (cos angle diff)."""
        return 0.8
    
    @property
    def supports_inverted_poses(self) -> bool:
        """Whether this robot config supports inverted (handstand) poses."""
        return False
    
    # ------------------------------------------------------------------ asset paths
    @property
    @abstractmethod
    def xml_path_env_var(self) -> str:
        """Environment variable name for the MJCF/URDF path."""
        ...
    
    @property
    def xml_path_default(self) -> str | None:
        """Default asset path (None means env var is required)."""
        return None
    
    def get_xml_path(self) -> Path:
        """Resolve the XML/MJCF asset path from env var or default."""
        path = os.environ.get(self.xml_path_env_var)
        if path:
            return Path(path)
        if self.xml_path_default:
            return Path(self.xml_path_default)
        raise ValueError(
            f"Robot asset path not configured. Set {self.xml_path_env_var} environment variable "
            f"to point to your local MJCF/URDF file for {self.name}."
        )
    
    @property
    def meshes_dir_env_var(self) -> str | None:
        """Environment variable for mesh directory (if separate from MJCF)."""
        return None
    
    # ------------------------------------------------------------------ mjlab integration
    @abstractmethod
    def get_robot_cfg_func(self) -> Callable[[], Any]:
        """Return a callable that produces the mjlab robot entity config."""
        ...
    
    # ------------------------------------------------------------------ state slicing helpers
    def slice_qpos_root(self) -> slice:
        """Slice for root position in qpos [0:3]."""
        return slice(0, 3)
    
    def slice_qpos_quat(self) -> slice:
        """Slice for root quaternion in qpos [3:7]."""
        return slice(3, 7)
    
    def slice_qpos_joints(self) -> slice:
        """Slice for joint positions in qpos [7:7+nj]."""
        return slice(7, 7 + self.num_joints)
    
    def slice_qvel_lin(self) -> slice:
        """Slice for root linear velocity in qvel [0:3]."""
        return slice(0, 3)
    
    def slice_qvel_ang(self) -> slice:
        """Slice for root angular velocity in qvel [3:6]."""
        return slice(3, 6)
    
    def slice_qvel_joints(self) -> slice:
        """Slice for joint velocities in qvel [6:6+nj]."""
        return slice(6, 6 + self.num_joints)
    
    def slice_state_qpos(self) -> slice:
        """Slice for qpos in full state [0:nq]."""
        return slice(0, self.nq)
    
    def slice_state_qvel(self) -> slice:
        """Slice for qvel in full state [nq:nq+nv]."""
        return slice(self.nq, self.nq + self.nv)
    
    # ------------------------------------------------------------------ reference state slicing
    def slice_ref_root_pos(self) -> slice:
        """Slice for root position in reference state."""
        return slice(0, 3)
    
    def slice_ref_root_quat(self) -> slice:
        """Slice for root quaternion in reference state."""
        return slice(3, 7)
    
    def slice_ref_joint_pos(self) -> slice:
        """Slice for joint positions in reference state."""
        return slice(7, 7 + self.num_joints)
    
    def slice_ref_root_lin_vel(self) -> slice:
        """Slice for root linear velocity in reference state."""
        start = 7 + self.num_joints
        return slice(start, start + 3)
    
    def slice_ref_root_ang_vel(self) -> slice:
        """Slice for root angular velocity in reference state."""
        start = 7 + self.num_joints + 3
        return slice(start, start + 3)
    
    def slice_ref_joint_vel(self) -> slice:
        """Slice for joint velocities in reference state."""
        start = 7 + self.num_joints + 6
        return slice(start, start + self.num_joints)
