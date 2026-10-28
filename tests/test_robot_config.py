"""Tests for robot configuration abstraction."""

import pytest

from supermimic.robots import (
    G1Config,
    PiPlusConfig,
    RobotConfig,
    get_robot_config,
    list_robots,
    register_robot,
)
from supermimic.robots.base import compute_goal_feat_dim, compute_ref_dim, compute_state_dim


def test_g1_config():
    """Test G1 robot configuration."""
    cfg = G1Config()
    assert cfg.name == "g1"
    assert cfg.num_joints == 29
    assert cfg.nq == 36  # 7 + 29
    assert cfg.nv == 35  # 6 + 29
    assert len(cfg.joint_names) == 29
    assert len(cfg.action_scale) == 29
    assert cfg.root_body == "pelvis"
    assert cfg.anchor_body == "torso_link"


def test_pi_plus_config():
    """Test Pi Plus robot configuration."""
    cfg = PiPlusConfig()
    assert cfg.name == "pi_plus"
    assert cfg.num_joints == 22
    assert cfg.nq == 29  # 7 + 22
    assert cfg.nv == 28  # 6 + 22
    assert len(cfg.joint_names) == 22
    assert len(cfg.action_scale) == 22
    assert len(cfg.effort_limits) == 22
    assert cfg.supports_handstand
    assert cfg.head_body == "head"
    assert "left_hand_link" in cfg.hand_bodies
    assert "right_hand_link" in cfg.hand_bodies


def test_pi_plus_no_wrist():
    """Pi Plus should have no wrist joints."""
    cfg = PiPlusConfig()
    joint_names = cfg.joint_names
    assert not any("wrist" in j for j in joint_names)


def test_pi_plus_arm_effort_limits():
    """Pi Plus arm joints should have ~10 Nm effort limit."""
    cfg = PiPlusConfig()
    # Arm joints are at indices 15-18 (left) and 19-22 (right)
    arm_limits = cfg.effort_limits[15:23]
    assert all(lim == 10.0 for lim in arm_limits)


def test_get_robot_config():
    """Test registry lookup."""
    g1 = get_robot_config("g1")
    assert isinstance(g1, G1Config)
    
    pi_plus = get_robot_config("pi_plus")
    assert isinstance(pi_plus, PiPlusConfig)


def test_get_unknown_robot():
    """Test error for unknown robot."""
    with pytest.raises(KeyError):
        get_robot_config("unknown_robot")


def test_list_robots():
    """Test listing available robots."""
    robots = list_robots()
    assert "g1" in robots
    assert "pi_plus" in robots


def test_handstand_contact_bodies():
    """Test handstand contact body configuration."""
    pi = PiPlusConfig()
    contacts = pi.handstand_contact_bodies
    assert "head" in contacts
    assert "left_hand_link" in contacts
    assert "right_hand_link" in contacts
    assert len(contacts) == 3


def test_dimension_computations():
    """Test dimension computation functions."""
    # G1: 29 joints
    assert compute_goal_feat_dim(29) == 68  # 2*29 + 10
    assert compute_ref_dim(29) == 71  # 13 + 2*29
    assert compute_state_dim(29) == 71  # 7 + 2*29
    
    # Pi Plus: 22 joints
    assert compute_goal_feat_dim(22) == 54  # 2*22 + 10
    assert compute_ref_dim(22) == 57  # 13 + 2*22
    assert compute_state_dim(22) == 57  # 7 + 2*22


# Import dimension helpers
from supermimic.envs.mdp.motion_library import compute_goal_feat_dim, compute_ref_dim
from supermimic.world_model.state_spec import compute_state_dim
