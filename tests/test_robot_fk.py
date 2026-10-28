"""Tests for robot-agnostic forward kinematics (RobotKinematics).

This module tests that:
1. RobotKinematics produces the same results as the original G1Kinematics
2. The FK is differentiable
3. It works with different robot configurations
"""

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.robot_fk import RobotKinematics
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_rotvec, rotvec_from_quat


def _random_state(model, rng, n):
    qpos = np.zeros((n, model.nq))
    qvel = rng.normal(size=(n, model.nv)) * np.array([1.0] * 3 + [2.0] * 3 + [3.0] * (model.nv - 6))
    qpos[:, :3] = rng.normal(size=(n, 3))
    q = rng.normal(size=(n, 4))
    qpos[:, 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    qpos[:, 7:] = rng.uniform(lo, hi, size=(n, model.nq - 7))
    return qpos, qvel


def _mujoco_reference(model, qpos, qvel):
    data = mujoco.MjData(model)
    n = qpos.shape[0]
    nb = model.nbody - 1
    pos = np.zeros((n, nb, 3))
    quat = np.zeros((n, nb, 4))
    lin = np.zeros((n, nb, 3))
    ang = np.zeros((n, nb, 3))
    res = np.zeros(6)
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        pos[i] = data.xpos[1:]
        quat[i] = data.xquat[1:]
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
            ang[i, b - 1] = res[:3]
            lin[i, b - 1] = res[3:]
    return pos, quat, lin, ang


def _signed_quat_err(q, q_ref):
    sign = np.sign(np.sum(q * q_ref, axis=-1, keepdims=True))
    sign[sign == 0] = 1
    return np.abs(q * sign - q_ref).max()


class TestRobotKinematics:
    """Tests for the generalized RobotKinematics class."""

    @pytest.mark.parametrize("dtype,atol_pos,atol_vel", [(torch.float64, 1e-10, 1e-9), (torch.float32, 2e-5, 2e-4)])
    def test_fk_matches_mujoco(self, dtype, atol_pos, atol_vel):
        """RobotKinematics should match MuJoCo's FK exactly."""
        torch.backends.cuda.matmul.allow_tf32 = False
        model = load_g1_model()
        rng = np.random.default_rng(0)
        qpos, qvel = _random_state(model, rng, 64)
        pos_ref, quat_ref, lin_ref, ang_ref = _mujoco_reference(model, qpos, qvel)

        fk = RobotKinematics(model, device="cpu", dtype=dtype)
        out = fk(torch.as_tensor(qpos, dtype=dtype), torch.as_tensor(qvel, dtype=dtype))
        
        assert out["pos"].shape == (64, 30, 3) and out["quat"].shape == (64, 30, 4)
        np.testing.assert_allclose(out["pos"].numpy(), pos_ref, atol=atol_pos)
        assert _signed_quat_err(out["quat"].numpy(), quat_ref) < atol_pos * 10
        np.testing.assert_allclose(out["ang_vel"].numpy(), ang_ref, atol=atol_vel)
        np.testing.assert_allclose(out["lin_vel"].numpy(), lin_ref, atol=atol_vel)

    def test_g1kinematics_alias(self):
        """G1Kinematics should be an alias for RobotKinematics."""
        assert G1Kinematics is RobotKinematics

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_fk_cuda_matches_cpu(self):
        """CUDA FK should match CPU FK."""
        torch.backends.cuda.matmul.allow_tf32 = False
        model = load_g1_model()
        rng = np.random.default_rng(1)
        qpos, qvel = _random_state(model, rng, 256)
        fk_cpu = RobotKinematics(model, "cpu", torch.float32)
        fk_gpu = RobotKinematics(model, "cuda", torch.float32)
        a = fk_cpu(torch.as_tensor(qpos, dtype=torch.float32), torch.as_tensor(qvel, dtype=torch.float32))
        b = fk_gpu(torch.as_tensor(qpos, dtype=torch.float32, device="cuda"), torch.as_tensor(qvel, dtype=torch.float32, device="cuda"))
        for k in a:
            np.testing.assert_allclose(a[k].numpy(), b[k].cpu().numpy(), atol=5e-5)

    def test_fk_is_differentiable(self):
        """FK should support autograd."""
        model = load_g1_model()
        rng = np.random.default_rng(2)
        qpos, qvel = _random_state(model, rng, 4)
        fk = RobotKinematics(model, "cpu", torch.float64)
        qp = torch.tensor(qpos, dtype=torch.float64, requires_grad=True)
        qv = torch.tensor(qvel, dtype=torch.float64, requires_grad=True)
        out = fk(qp, qv)
        loss = out["pos"].square().sum() + out["lin_vel"].square().sum() + out["quat"].square().sum()
        loss.backward()
        assert torch.isfinite(qp.grad).all() and torch.isfinite(qv.grad).all()
        assert qp.grad.abs().sum() > 0 and qv.grad.abs().sum() > 0

    def test_body_name_lookup(self):
        """Body name lookup should work correctly."""
        model = load_g1_model()
        fk = RobotKinematics(model, "cpu")
        
        pelvis_idx = fk.body_index("pelvis")
        assert pelvis_idx == 0
        
        indices = fk.body_indices(["pelvis", "torso_link"])
        assert len(indices) == 2
        
    def test_num_joints(self):
        """FK should correctly report the number of joints."""
        model = load_g1_model()
        fk = RobotKinematics(model, "cpu")
        assert fk.num_joints == 29
        assert fk.nq == 36
        assert fk.nv == 35


class TestRobotConfig:
    """Tests for robot configuration."""

    def test_g1_config_loads(self):
        """G1 config should load correctly."""
        from supermimic.robots import G1_CONFIG
        
        assert G1_CONFIG.name == "g1"
        assert G1_CONFIG.num_joints == 29
        assert G1_CONFIG.nq == 36
        assert G1_CONFIG.nv == 35
        assert len(G1_CONFIG.joint_names) == 29
        assert G1_CONFIG.state_dim == 71
        assert G1_CONFIG.xi_dim == 33

    def test_piplus_config_loads(self):
        """Pi Plus config should load correctly."""
        from supermimic.robots import PIPLUS_CONFIG
        
        assert PIPLUS_CONFIG.name == "piplus"
        assert PIPLUS_CONFIG.num_joints == 22
        assert PIPLUS_CONFIG.nq == 29
        assert PIPLUS_CONFIG.nv == 28
        assert len(PIPLUS_CONFIG.joint_names) == 22
        assert PIPLUS_CONFIG.state_dim == 57
        assert PIPLUS_CONFIG.xi_dim == 26

    def test_get_robot_config(self):
        """get_robot_config should return the correct config."""
        from supermimic.robots import get_robot_config
        
        g1 = get_robot_config("g1")
        assert g1.name == "g1"
        
        piplus = get_robot_config("piplus")
        assert piplus.name == "piplus"
        
        with pytest.raises(KeyError):
            get_robot_config("nonexistent")


class TestStateSpec:
    """Tests for parameterized state specification."""

    def test_state_spec_g1_dimensions(self):
        """StateSpec should have correct dimensions for G1."""
        from supermimic.world_model.state_spec import StateSpecCfg, StateSpec
        
        cfg = StateSpecCfg(num_joints=29)
        spec = StateSpec(cfg, device="cpu")
        
        assert spec.state_dim == 71
        assert spec.feat_dim == 71
        assert spec.delta_dim == 70
        assert spec.xi_dim == 33

    def test_state_spec_piplus_dimensions(self):
        """StateSpec should have correct dimensions for Pi Plus."""
        from supermimic.world_model.state_spec import StateSpecCfg, StateSpec
        
        cfg = StateSpecCfg(num_joints=22)
        spec = StateSpec(cfg, device="cpu")
        
        assert spec.state_dim == 57
        assert spec.feat_dim == 57
        assert spec.delta_dim == 56
        assert spec.xi_dim == 26

    def test_state_spec_split(self):
        """split() should correctly partition state vectors."""
        from supermimic.world_model.state_spec import StateSpecCfg, StateSpec
        import torch
        
        for nj in [22, 29]:
            cfg = StateSpecCfg(num_joints=nj)
            spec = StateSpec(cfg, device="cpu")
            
            state_dim = spec.state_dim
            s = torch.randn(4, state_dim)
            p, q, qj, v, w, qd = spec.split(s)
            
            assert p.shape == (4, 3)
            assert q.shape == (4, 4)
            assert qj.shape == (4, nj)
            assert v.shape == (4, 3)
            assert w.shape == (4, 3)
            assert qd.shape == (4, nj)

    def test_state_spec_from_robot_config(self):
        """StateSpecCfg.from_robot_config should work."""
        from supermimic.world_model.state_spec import StateSpecCfg
        from supermimic.robots import get_robot_config
        
        piplus = get_robot_config("piplus")
        cfg = StateSpecCfg.from_robot_config(piplus)
        
        assert cfg.num_joints == 22
        assert cfg.state_dim == 57


def test_rotvec_round_trip():
    """Rotation vector round trip should be stable."""
    v = torch.randn(1000, 3, dtype=torch.float64) * 2.5
    v[0] = 0.0
    v[1] = torch.tensor([1e-9, 0, 0])
    q = quat_from_rotvec(v)
    np.testing.assert_allclose(q.norm(dim=-1).numpy(), 1.0, atol=1e-12)
    v2 = rotvec_from_quat(q)
    n = v.norm(dim=-1, keepdim=True)
    ang = torch.remainder(n, 2 * np.pi)
    ang = torch.where(ang > np.pi, ang - 2 * np.pi, ang)
    wrapped = torch.where(n > 0, v / n.clamp_min(1e-30) * ang, v)
    np.testing.assert_allclose(v2.numpy(), wrapped.numpy(), atol=1e-9)
    v0 = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    quat_from_rotvec(v0).sum().backward()
    assert torch.isfinite(v0.grad).all()
