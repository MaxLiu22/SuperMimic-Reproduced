"""Per-environment simulation context xi.

xi = [foot friction(1), encoder bias(nj), torso COM offset(3)]

Dimensions are now parameterized via RobotConfig for multi-robot support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from supermimic.robot.robot_config import RobotConfig

# Legacy constant for backward compatibility (G1: 1 + 29 + 3 = 33)
XI_DIM = 33


def get_xi_dim(robot_cfg: "RobotConfig | None" = None) -> int:
    """Get xi dimension for a robot config."""
    if robot_cfg is None:
        return XI_DIM
    return robot_cfg.xi_dim


class SimContext:
    """Per-environment simulation context with robot-parameterized dimensions.
    
    Args:
        env: The mjlab environment
        foot_geom_regex: Regex pattern matching foot collision geoms
        com_body_name: Body name for COM offset tracking
        entity_name: Name of the robot entity in the scene
        robot_cfg: Optional RobotConfig for dimension parameterization.
                   If None, uses G1 defaults for backward compatibility.
    """
    
    def __init__(
        self,
        env,
        foot_geom_regex: str,
        com_body_name: str = "torso_link",
        entity_name: str = "robot",
        robot_cfg: "RobotConfig | None" = None,
    ):
        self.env = env
        self.robot = env.scene[entity_name]
        self.robot_cfg = robot_cfg
        self.xi_dim = get_xi_dim(robot_cfg)
        
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )

    def read(self) -> torch.Tensor:
        """Read simulation context: (N, xi_dim) = [friction(1), encoder_bias(nj), com_offset(3)]."""
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
        
        expected_dim = self.xi_dim
        if xi.shape[-1] != expected_dim:
            raise ValueError(
                f"xi dimension mismatch: got {xi.shape[-1]}, expected {expected_dim}. "
                f"Check that robot config num_joints ({self.robot_cfg.num_joints if self.robot_cfg else 29}) "
                f"matches the simulation."
            )
        return xi.clone()
