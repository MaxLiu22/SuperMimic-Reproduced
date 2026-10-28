"""Per-environment simulation context xi.

Xi layout: foot friction (1) + encoder bias (num_joints) + torso COM offset (3).
For G1 (29 joints): XI_DIM = 33.
For Pi Plus (22 joints): XI_DIM = 26.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from supermimic.robots.config import RobotConfig


def compute_xi_dim(num_joints: int) -> int:
    """Compute xi dimension: friction(1) + encoder_bias(nj) + com_offset(3)."""
    return 1 + num_joints + 3


XI_DIM_G1 = 33
XI_DIM = XI_DIM_G1


class SimContext:
    """Simulation context reader for domain randomization parameters."""

    def __init__(
        self,
        env,
        foot_geom_regex: str,
        com_body_name: str = "torso_link",
        entity_name: str = "robot",
        robot_config: "RobotConfig | None" = None,
    ):
        self.env = env
        self.robot = env.scene[entity_name]
        self.robot_config = robot_config

        if robot_config is not None:
            self.xi_dim = compute_xi_dim(robot_config.num_joints)
        else:
            self.xi_dim = XI_DIM_G1

        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )

    def read(self) -> torch.Tensor:
        """Read current simulation context parameters.

        Returns:
            xi tensor of shape (num_envs, xi_dim) containing:
            - friction (1): foot geom friction coefficient
            - encoder_bias (num_joints): joint encoder biases
            - com_offset (3): COM offset from nominal
        """
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:
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
        assert xi.shape == (n, self.xi_dim), f"xi shape {xi.shape} != (n={n}, xi_dim={self.xi_dim})"
        return xi.clone()
