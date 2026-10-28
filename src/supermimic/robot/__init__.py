"""Robot abstraction layer for multi-robot support.

This package provides a clean abstraction for robot-specific configurations,
allowing SuperMimic to work with different robots (G1, Pi Plus, etc.) without
hardcoding robot-specific dimensions and parameters throughout the codebase.
"""

from supermimic.robot.robot_config import RobotConfig
from supermimic.robot.g1_config import G1Config
from supermimic.robot.piplus_config import PiPlusConfig
from supermimic.robot.factory import get_robot_config, get_default_robot_config, set_default_robot

__all__ = [
    "RobotConfig",
    "G1Config",
    "PiPlusConfig",
    "get_robot_config",
    "get_default_robot_config",
    "set_default_robot",
]
