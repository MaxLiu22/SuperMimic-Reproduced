"""Read / write the full simulator state (qpos | qvel) as one state_dim tensor.

Dimensions are now parameterized via RobotConfig for multi-robot support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from supermimic.robot.robot_config import RobotConfig

# Legacy constants for backward compatibility (G1 defaults)
STATE_DIM = 36 + 35
NQ = 36
NV = 35


def get_state_dims(robot_cfg: "RobotConfig | None" = None) -> tuple[int, int, int]:
    """Get (state_dim, nq, nv) for a robot config."""
    if robot_cfg is None:
        return STATE_DIM, NQ, NV
    return robot_cfg.state_dim, robot_cfg.nq, robot_cfg.nv


class StateIO:
    """Read/write simulator state with robot-parameterized dimensions.
    
    Args:
        env: The mjlab environment
        robot_cfg: Optional RobotConfig for dimension parameterization.
                   If None, uses G1 defaults for backward compatibility.
    """
    
    def __init__(self, env, robot_cfg: "RobotConfig | None" = None):
        self.env = env
        self.sim = env.sim
        self.robot_cfg = robot_cfg
        
        # Get dimensions from robot config or use defaults
        self.state_dim, self.nq, self.nv = get_state_dims(robot_cfg)
        
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        
        actual_nq = self.sim.data.qpos.shape[-1]
        actual_nv = self.sim.data.qvel.shape[-1]
        if actual_nq != self.nq or actual_nv != self.nv:
            raise ValueError(
                f"Simulation dimensions (nq={actual_nq}, nv={actual_nv}) don't match "
                f"robot config (nq={self.nq}, nv={self.nv}). Check your robot configuration."
            )

    def read(self) -> torch.Tensor:
        """(N, state_dim) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        self.sim.data.qpos[env_ids] = state[:, :self.nq]
        self.sim.data.qvel[env_ids] = state[:, self.nq:]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
