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
error vector [dp(3), rotvec(3), dv(3), dw(3), dq(nj), dqd(nj)].

The dimensions are parameterized by RobotConfig.num_joints (nj):
- G1: nj=29, state_dim=71, feat_dim=71, delta_dim=70
- Pi Plus: nj=22, state_dim=57, feat_dim=57, delta_dim=56
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
    from supermimic.robots.config import RobotConfig


def compute_state_dim(nj: int) -> int:
    """State dim: root_pos(3) + root_quat(4) + joint_pos(nj) + root_lin_vel(3) + root_ang_vel(3) + joint_vel(nj)"""
    return 3 + 4 + nj + 3 + 3 + nj


def compute_feat_dim(nj: int) -> int:
    """Feature dim: root_z(1) + rot6d(6) + lin_vel_h(3) + ang_vel_b(3) + joint_pos(nj) + joint_vel(nj)"""
    return 1 + 6 + 3 + 3 + nj + nj


def compute_delta_dim(nj: int) -> int:
    """Delta dim: dv_h(3) + dw_b(3) + dqd(nj) + dp_h(3) + dth_b(3) + dq(nj)"""
    return 3 + 3 + nj + 3 + 3 + nj


def compute_xi_dim(nj: int) -> int:
    """Xi dim: friction(1) + encoder_bias(nj) + com_offset(3)"""
    return 1 + nj + 3


NJ_G1 = 29
STATE_DIM = compute_state_dim(NJ_G1)
FEAT_DIM = compute_feat_dim(NJ_G1)
DELTA_DIM = compute_delta_dim(NJ_G1)
NJ = NJ_G1


@dataclass
class StateSpecCfg:
    """State specification config, parameterized by num_joints.

    For G1 (29 joints), use default values.
    For Pi Plus (22 joints), set num_joints=22.
    """

    num_joints: int = NJ_G1
    dt: float = 0.02
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    xi_friction_offset: float = 0.75
    xi_friction_scale: float = 0.45
    xi_encoder_offset: float = 0.0
    xi_encoder_scale: float = 0.01
    xi_com_offset: float = 0.0
    xi_com_scale: float = 0.05

    @property
    def state_dim(self) -> int:
        return compute_state_dim(self.num_joints)

    @property
    def feat_dim(self) -> int:
        return compute_feat_dim(self.num_joints)

    @property
    def delta_dim(self) -> int:
        return compute_delta_dim(self.num_joints)

    @property
    def xi_dim(self) -> int:
        return compute_xi_dim(self.num_joints)

    @classmethod
    def from_robot_config(cls, robot_config: "RobotConfig", **overrides) -> "StateSpecCfg":
        """Create StateSpecCfg from RobotConfig."""
        return cls(num_joints=robot_config.num_joints, **overrides)


class StateSpec:
    """State specification for world model, parameterized by num_joints."""

    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        self.nj = cfg.num_joints
        self.state_dim = cfg.state_dim
        self.feat_dim = cfg.feat_dim
        self.delta_dim = cfg.delta_dim
        self.xi_dim = cfg.xi_dim

        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, self.nj, self.nj)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, self.nj, 3, 3, self.nj)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, self.nj, self.nj)).to(device=device, dtype=dtype)

        xi_offset = (cfg.xi_friction_offset,) + (cfg.xi_encoder_offset,) * self.nj + (cfg.xi_com_offset,) * 3
        xi_scale = (cfg.xi_friction_scale,) + (cfg.xi_encoder_scale,) * self.nj + (cfg.xi_com_scale,) * 3
        self.xi_offset = torch.tensor(xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(xi_scale, device=device, dtype=dtype)

        self._joint_pos_end = 7 + self.nj
        self._vel_start = self._joint_pos_end
        self._vel_end = self._vel_start + 3
        self._ang_start = self._vel_end
        self._ang_end = self._ang_start + 3
        self._jvel_start = self._ang_end
        self._jvel_end = self._jvel_start + self.nj

    def split(self, s: torch.Tensor):
        """Split state into components: (root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel)"""
        return (
            s[..., 0:3],
            s[..., 3:7],
            s[..., 7:self._joint_pos_end],
            s[..., self._vel_start:self._vel_end],
            s[..., self._ang_start:self._ang_end],
            s[..., self._jvel_start:self._jvel_end],
        )

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` to state ``s``."""
        d = delta * self.out_scale
        nj = self.nj
        dv_h = d[..., 0:3]
        dw_b = d[..., 3:6]
        dqd = d[..., 6:6 + nj]
        dp_h = d[..., 6 + nj:9 + nj]
        dth_b = d[..., 9 + nj:12 + nj]
        dq = d[..., 12 + nj:12 + 2 * nj]

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

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised error vector [dp, rotvec, dv, dw, dq, dqd] / sigma."""
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
