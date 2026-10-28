"""Robot abstraction layer for multi-robot SuperMimic support.

This module provides a robot-agnostic interface that parameterizes:
- Joint/body topology (names, DOFs, kinematic structure)
- MJCF/URDF asset paths (local, not committed to git)
- Action scales and limits
- Tracked bodies for motion tracking
- Contact bodies for termination (standing, handstand, etc.)
"""

from supermimic.robots.config import (
    ContactConfig,
    RobotConfig,
    get_robot_config,
    register_robot_config,
)
from supermimic.robots.g1 import G1_CONFIG
from supermimic.robots.piplus import PIPLUS_CONFIG

__all__ = [
    "RobotConfig",
    "ContactConfig",
    "get_robot_config",
    "register_robot_config",
    "G1_CONFIG",
    "PIPLUS_CONFIG",
]
