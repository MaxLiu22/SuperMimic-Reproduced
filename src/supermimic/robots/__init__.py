"""Robot configuration abstraction for multi-robot SuperMimic support.

This module provides a unified interface for different humanoid robots,
abstracting away robot-specific dimensions, joint names, body names,
and asset paths. Currently supports:
  - Unitree G1 (29 actuated joints)
  - HighTorque Pi Plus (22 actuated joints, no wrist DOF)
"""

from supermimic.robots.base import (
    RobotConfig,
    get_robot_config,
    register_robot,
    list_robots,
)
from supermimic.robots.g1 import G1Config
from supermimic.robots.pi_plus import PiPlusConfig

__all__ = [
    "RobotConfig",
    "get_robot_config",
    "register_robot",
    "list_robots",
    "G1Config",
    "PiPlusConfig",
]
