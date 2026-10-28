"""Per-environment simulation context xi (plan: xi_dim = foot friction 1 + encoder bias nj + torso COM offset 3).

Supports variable joint counts:
- G1: xi_dim = 33 (1 + 29 + 3)
- Pi Plus: xi_dim = 26 (1 + 22 + 3)
"""

from __future__ import annotations

import numpy as np
import torch


def compute_xi_dim(num_joints: int) -> int:
    """Compute xi dimension: 1 (friction) + nj (encoder bias) + 3 (COM offset)."""
    return 1 + num_joints + 3


# Default for G1 (29 joints) - kept for backwards compatibility
XI_DIM = 33


class SimContext:
    """Simulation context reader for domain randomization state.
    
    Extracts per-environment context vector xi containing:
    - Foot friction coefficient (1)
    - Encoder bias per joint (num_joints)
    - Center of mass offset (3)
    
    Args:
        env: The environment instance.
        foot_geom_regex: Regex pattern for foot geoms.
        com_body_name: Body name for COM offset tracking.
        entity_name: Entity name in the scene.
    """
    
    def __init__(
        self,
        env,
        foot_geom_regex: str,
        com_body_name: str = "torso_link",
        entity_name: str = "robot",
    ):
        self.env = env
        self.robot = env.scene[entity_name]
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )
        
        # Compute xi dimension based on number of joints
        self.num_joints = self.robot.data.joint_pos.shape[-1]
        self.xi_dim = compute_xi_dim(self.num_joints)

    def read(self) -> torch.Tensor:
        """Read the current simulation context.
        
        Returns:
            Tensor of shape (num_envs, xi_dim) containing:
            - friction[0]: foot friction coefficient
            - encoder_bias[1:1+nj]: per-joint encoder bias
            - com_offset[1+nj:]: COM offset from nominal
        """
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        xi = torch.cat([friction.reshape(n, 1), bias, com], dim=-1)
        assert xi.shape == (n, self.xi_dim), f"Expected xi shape ({n}, {self.xi_dim}), got {xi.shape}"
        return xi.clone()
