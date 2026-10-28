"""Robot configuration factory and registry.

Provides a simple interface to get robot configurations by name
and manage the default robot for the session.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supermimic.robot.robot_config import RobotConfig

# Registry of known robot configurations
_ROBOT_REGISTRY: dict[str, type["RobotConfig"]] = {}

# Default robot for the session (can be overridden)
_DEFAULT_ROBOT: str | None = None


def _ensure_registry():
    """Lazy-load the registry to avoid circular imports."""
    if not _ROBOT_REGISTRY:
        from supermimic.robot.g1_config import G1Config
        from supermimic.robot.piplus_config import PiPlusConfig
        
        _ROBOT_REGISTRY["g1"] = G1Config
        _ROBOT_REGISTRY["piplus"] = PiPlusConfig


def register_robot(name: str, config_class: type["RobotConfig"]) -> None:
    """Register a custom robot configuration.
    
    Args:
        name: Identifier for the robot (e.g., "my_robot")
        config_class: RobotConfig subclass
    """
    _ROBOT_REGISTRY[name.lower()] = config_class


def get_robot_config(name: str | None = None, **kwargs) -> "RobotConfig":
    """Get a robot configuration by name.
    
    Args:
        name: Robot identifier ("g1", "piplus", etc.). 
              If None, uses default from env var or "g1".
        **kwargs: Additional arguments passed to the config constructor.
    
    Returns:
        Instantiated RobotConfig subclass.
    
    Raises:
        ValueError: If the robot name is not registered.
    """
    _ensure_registry()
    
    if name is None:
        name = _get_default_robot_name()
    
    name = name.lower()
    if name not in _ROBOT_REGISTRY:
        available = ", ".join(sorted(_ROBOT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown robot '{name}'. Available robots: {available}. "
            f"Use register_robot() to add custom configurations."
        )
    
    return _ROBOT_REGISTRY[name](**kwargs)


def get_default_robot_config(**kwargs) -> "RobotConfig":
    """Get the default robot configuration for this session.
    
    The default is determined by:
    1. set_default_robot() if called
    2. SUPERMIMIC_ROBOT environment variable
    3. "g1" as the fallback default
    """
    return get_robot_config(None, **kwargs)


def set_default_robot(name: str) -> None:
    """Set the default robot for this session.
    
    Args:
        name: Robot identifier (e.g., "g1", "piplus")
    """
    global _DEFAULT_ROBOT
    _ensure_registry()
    name = name.lower()
    if name not in _ROBOT_REGISTRY:
        available = ", ".join(sorted(_ROBOT_REGISTRY.keys()))
        raise ValueError(f"Unknown robot '{name}'. Available: {available}")
    _DEFAULT_ROBOT = name


def _get_default_robot_name() -> str:
    """Get the default robot name from session state or environment."""
    if _DEFAULT_ROBOT:
        return _DEFAULT_ROBOT
    return os.environ.get("SUPERMIMIC_ROBOT", "g1")


def list_robots() -> list[str]:
    """List all registered robot configurations."""
    _ensure_registry()
    return sorted(_ROBOT_REGISTRY.keys())
