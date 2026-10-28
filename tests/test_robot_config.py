"""Tests for the robot abstraction layer."""

import pytest


def test_g1_config_properties():
    """G1Config should have correct dimensions."""
    from supermimic.robot.g1_config import G1Config
    
    cfg = G1Config()
    assert cfg.name == "g1"
    assert cfg.num_joints == 29
    assert cfg.nq == 36
    assert cfg.nv == 35
    assert cfg.state_dim == 71
    assert cfg.delta_dim == 70
    assert cfg.xi_dim == 33  # 1 + 29 + 3
    assert cfg.ref_dim == 71
    assert cfg.goal_feat_dim == 68  # 29*2 + 10
    
    assert len(cfg.joint_names) == 29
    assert cfg.anchor_body_name == "torso_link"
    assert cfg.root_body_name == "pelvis"


def test_piplus_config_properties():
    """PiPlusConfig should have correct dimensions."""
    from supermimic.robot.piplus_config import PiPlusConfig
    
    cfg = PiPlusConfig()
    assert cfg.name == "piplus"
    assert cfg.num_joints == 22
    assert cfg.nq == 29
    assert cfg.nv == 28
    assert cfg.state_dim == 57
    assert cfg.delta_dim == 56
    assert cfg.xi_dim == 26  # 1 + 22 + 3
    assert cfg.ref_dim == 57
    assert cfg.goal_feat_dim == 54  # 22*2 + 10
    
    assert len(cfg.joint_names) == 22
    assert cfg.supports_inverted_poses is True


def test_robot_factory():
    """Factory should return correct configs."""
    from supermimic.robot import get_robot_config, list_robots
    
    robots = list_robots()
    assert "g1" in robots
    assert "piplus" in robots
    
    g1 = get_robot_config("g1")
    assert g1.name == "g1"
    assert g1.num_joints == 29
    
    piplus = get_robot_config("piplus")
    assert piplus.name == "piplus"
    assert piplus.num_joints == 22
    
    # Default should be G1
    default = get_robot_config()
    assert default.name == "g1"


def test_robot_factory_unknown_robot():
    """Factory should raise for unknown robots."""
    from supermimic.robot import get_robot_config
    
    with pytest.raises(ValueError, match="Unknown robot"):
        get_robot_config("unknown_robot")


def test_piplus_config_env_override(monkeypatch):
    """PiPlusConfig should respect environment variable overrides."""
    from supermimic.robot.piplus_config import PiPlusConfig
    
    # Test action scale override
    monkeypatch.setenv("SUPERMIMIC_PIPLUS_ACTION_SCALE", "0.5")
    cfg = PiPlusConfig()
    assert cfg.action_scale == 0.5
    
    # Test body name overrides
    monkeypatch.setenv("SUPERMIMIC_PIPLUS_HEAD_BODY", "custom_head")
    cfg = PiPlusConfig()
    assert cfg.head_body_name == "custom_head"


def test_robot_slice_helpers():
    """RobotConfig slice helpers should return correct indices."""
    from supermimic.robot.g1_config import G1Config
    from supermimic.robot.piplus_config import PiPlusConfig
    
    g1 = G1Config()
    assert g1.slice_qpos_root() == slice(0, 3)
    assert g1.slice_qpos_quat() == slice(3, 7)
    assert g1.slice_qpos_joints() == slice(7, 36)
    
    piplus = PiPlusConfig()
    assert piplus.slice_qpos_root() == slice(0, 3)
    assert piplus.slice_qpos_quat() == slice(3, 7)
    assert piplus.slice_qpos_joints() == slice(7, 29)


def test_state_spec_with_robot_config():
    """StateSpec should work with different robot configs."""
    import torch
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
    from supermimic.robot.g1_config import G1Config
    from supermimic.robot.piplus_config import PiPlusConfig
    
    cfg = StateSpecCfg()
    
    # G1
    g1_spec = StateSpec(cfg, "cpu", robot_cfg=G1Config())
    assert g1_spec.state_dim == 71
    assert g1_spec.nj == 29
    
    # Pi Plus
    piplus_spec = StateSpec(cfg, "cpu", robot_cfg=PiPlusConfig())
    assert piplus_spec.state_dim == 57
    assert piplus_spec.nj == 22


def test_piplus_env_template():
    """PiPlus env template should be valid."""
    from supermimic.robot.piplus_config import create_piplus_env_template
    
    template = create_piplus_env_template()
    assert "SUPERMIMIC_PIPLUS_XML_PATH" in template
    assert "SUPERMIMIC_PIPLUS_JOINT_NAMES" in template
    assert "SUPERMIMIC_PIPLUS_HEAD_BODY" in template
