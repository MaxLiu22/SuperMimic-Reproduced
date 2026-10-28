"""State representation for the world model (plan D1–D3).

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(nj), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(nj)]            (state_dim, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, feat_dim):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(nj), joint_vel(nj)]

Network output ``delta`` (delta_dim) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(nj)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(nj)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
delta_dim error vector [dp(3), rotvec(3), dv(3), dw(3), dq(nj), dqd(nj)].

Robot dimensions are now parameterized via RobotConfig for multi-robot support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

if TYPE_CHECKING:
    from supermimic.robot.robot_config import RobotConfig

# Legacy constants for backward compatibility (G1 defaults)
STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70
NJ = 29


def get_dims_for_robot(robot_cfg: "RobotConfig | None" = None) -> tuple[int, int, int, int]:
    """Get (state_dim, feat_dim, delta_dim, nj) for a robot config."""
    if robot_cfg is None:
        return STATE_DIM, FEAT_DIM, DELTA_DIM, NJ
    return robot_cfg.state_dim, robot_cfg.feat_dim, robot_cfg.delta_dim, robot_cfg.num_joints


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset.
    # These are computed dynamically based on num_joints if None.
    xi_offset: tuple[float, ...] | None = None
    xi_scale: tuple[float, ...] | None = None
    
    def get_xi_offset(self, nj: int) -> tuple[float, ...]:
        """Get xi offset, computing default based on num_joints if not set."""
        if self.xi_offset is not None:
            return self.xi_offset
        return (0.75,) + (0.0,) * nj + (0.0,) * 3
    
    def get_xi_scale(self, nj: int) -> tuple[float, ...]:
        """Get xi scale, computing default based on num_joints if not set."""
        if self.xi_scale is not None:
            return self.xi_scale
        return (0.45,) + (0.01,) * nj + (0.05,) * 3


class StateSpec:
    """State specification with robot-parameterized dimensions.
    
    Args:
        cfg: StateSpecCfg configuration
        device: torch device
        dtype: torch dtype
        robot_cfg: Optional RobotConfig for dimension parameterization.
                   If None, uses G1 defaults for backward compatibility.
    """
    
    def __init__(
        self,
        cfg: StateSpecCfg,
        device,
        dtype=torch.float32,
        robot_cfg: "RobotConfig | None" = None,
    ):
        self.cfg = cfg
        self.dt = cfg.dt
        self.robot_cfg = robot_cfg
        
        # Get dimensions from robot config or use defaults
        state_dim, feat_dim, delta_dim, nj = get_dims_for_robot(robot_cfg)
        self.state_dim = state_dim
        self.feat_dim = feat_dim
        self.delta_dim = delta_dim
        self.nj = nj
        
        # Compute slice indices for state decomposition
        self._root_pos_end = 3
        self._root_quat_end = 7
        self._joint_pos_end = 7 + nj
        self._root_lin_vel_end = 7 + nj + 3
        self._root_ang_vel_end = 7 + nj + 6
        self._joint_vel_end = 7 + nj + 6 + nj  # == state_dim
        
        # Compute slice indices for delta decomposition
        self._dqd_end = 6 + nj
        self._dp_h_end = 6 + nj + 3
        self._dth_b_end = 6 + nj + 6
        self._dq_end = 6 + nj + 6 + nj  # == delta_dim
        
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, nj, nj)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, nj, 3, 3, nj)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, nj, nj)).to(device=device, dtype=dtype)
        
        xi_offset = cfg.get_xi_offset(nj)
        xi_scale = cfg.get_xi_scale(nj)
        self.xi_offset = torch.tensor(xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(xi_scale, device=device, dtype=dtype)

    # ------------------------------------------------------------------ split / features

    def split(self, s: torch.Tensor):
        """Split state tensor into components: (root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel)."""
        return (
            s[..., :self._root_pos_end],
            s[..., self._root_pos_end:self._root_quat_end],
            s[..., self._root_quat_end:self._joint_pos_end],
            s[..., self._joint_pos_end:self._root_lin_vel_end],
            s[..., self._root_lin_vel_end:self._root_ang_vel_end],
            s[..., self._root_ang_vel_end:self._joint_vel_end],
        )
    
    @staticmethod
    def split_g1(s: torch.Tensor):
        """Legacy static split for G1 (29 joints). Use split() method for robot-agnostic code."""
        return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, feat_dim)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, delta_dim) to state ``s`` (…, state_dim)."""
        d = delta * self.out_scale
        dv_h = d[..., 0:3]
        dw_b = d[..., 3:6]
        dqd = d[..., 6:self._dqd_end]
        dp_h = d[..., self._dqd_end:self._dp_h_end]
        dth_b = d[..., self._dp_h_end:self._dth_b_end]
        dq = d[..., self._dth_b_end:self._dq_end]
        
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
