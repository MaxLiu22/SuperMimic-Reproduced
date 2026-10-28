# Pi Plus Robot Setup for SuperMimic

This document describes how to configure SuperMimic for the HighTorque Pi Plus humanoid robot, including handstand (倒立) training.

## IMPORTANT: Asset Privacy

**Pi Plus URDF/MJCF/mesh files must NOT be committed to GitHub.**

All robot model files should remain on your local machine (AutoDL, personal workstation, etc.). SuperMimic uses environment variables to locate these files at runtime.

## Pi Plus Specifications

| Property | Value |
|----------|-------|
| Actuated joints | 22 revolute |
| Floating base | 7-DOF freejoint |
| qpos dimension | 29 (7 + 22) |
| qvel dimension | 28 (6 + 22) |
| Wrist DOF | None |
| Arm torque | ~±10 Nm |

### Joint Configuration (22 DOF)

Default joint ordering (customize via environment variable if your MJCF differs):

```
# Legs (12 DOF)
left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint,
right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint,
right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint,

# Waist (3 DOF)
waist_yaw_joint, waist_roll_joint, waist_pitch_joint,

# Arms (7 DOF, no wrist)
left_shoulder_pitch_joint, left_shoulder_roll_joint,
left_elbow_pitch_joint, left_elbow_roll_joint,
right_shoulder_pitch_joint, right_shoulder_roll_joint,
right_elbow_pitch_joint, right_elbow_roll_joint
```

## Environment Setup

### Step 1: Create a `.env` file

Create a `.env` file in your project root (NOT tracked by git):

```bash
# Required: Path to your local Pi Plus MJCF
SUPERMIMIC_PIPLUS_XML_PATH=/path/to/your/piplus.xml

# Optional: Mesh directory if separate from MJCF
SUPERMIMIC_PIPLUS_MESHES_DIR=/path/to/meshes

# Set default robot to Pi Plus
SUPERMIMIC_ROBOT=piplus
```

### Step 2: Configure body/joint names (if needed)

If your MJCF uses different naming conventions:

```bash
# Comma-separated joint names in qpos order
SUPERMIMIC_PIPLUS_JOINT_NAMES=joint1,joint2,...

# Key body names
SUPERMIMIC_PIPLUS_ANCHOR_BODY=torso_link
SUPERMIMIC_PIPLUS_ROOT_BODY=pelvis
SUPERMIMIC_PIPLUS_COM_BODY=torso_link

# Handstand contact bodies
SUPERMIMIC_PIPLUS_HEAD_BODY=head_link
SUPERMIMIC_PIPLUS_HAND_BODIES=left_hand_link,right_hand_link
```

### Step 3: Configure collision geom patterns

```bash
SUPERMIMIC_PIPLUS_FOOT_GEOM_REGEX=^(left|right)_foot.*_collision$
SUPERMIMIC_PIPLUS_HAND_GEOM_REGEX=^(left|right)_hand.*_collision$
SUPERMIMIC_PIPLUS_HEAD_GEOM_REGEX=^head.*_collision$
```

### Step 4: Create a robot config module (for mjlab integration)

Create a local Python module that provides the mjlab entity config:

```python
# mymodule/piplus.py

def get_piplus_robot_cfg():
    """Return mjlab robot entity config for Pi Plus."""
    from mjlab.managers.scene_entity_config import ArticulationCfg
    import os
    
    xml_path = os.environ.get("SUPERMIMIC_PIPLUS_XML_PATH")
    if not xml_path:
        raise ValueError("SUPERMIMIC_PIPLUS_XML_PATH not set")
    
    return ArticulationCfg(
        spawn=ArticulationCfg.SpawnCfg(
            source_type="file",
            source_path=xml_path,
        ),
        # Add actuator configs, collision settings, etc.
    )
```

Then set:

```bash
SUPERMIMIC_PIPLUS_ROBOT_CFG_MODULE=mymodule.piplus:get_piplus_robot_cfg
```

## Handstand MVP Training

### Motion Data Preparation

1. Create/retarget handstand motion data to Pi Plus joint format
2. Convert to NPZ format:

```python
from supermimic.data.motion_convert import convert_csv_to_npz
from supermimic.robot import get_robot_config

# Load Pi Plus config
piplus_cfg = get_robot_config("piplus")

# Convert handstand motion
convert_csv_to_npz(
    csv_path="handstand.csv",
    out_path="data/npz/piplus_handstand/handstand.npz",
    input_fps=60.0,
    output_fps=50.0,
    robot_cfg=piplus_cfg,
)
```

### Training Configuration

```python
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env
from supermimic.robot import get_robot_config

# Load Pi Plus config
piplus_cfg = get_robot_config("piplus")

# Create handstand training config
args = SuperMimicEnvCfgArgs(
    npz_files=["data/npz/piplus_handstand/handstand.npz"],
    robot_cfg=piplus_cfg,
    inverted_mode=True,  # Enable handstand-friendly terminations
    num_envs=4096,
    # Relaxed thresholds for inverted poses
    root_height_threshold=0.35,
    root_tilt_threshold=1.2,
)

# Create environment
env = make_env(args, device="cuda:0")
```

### Key Differences for Handstand Training

1. **Inverted Mode**: Set `inverted_mode=True` to use handstand-compatible terminations
2. **Relaxed Thresholds**: Inverted poses have more variability, so thresholds are relaxed
3. **Tripod Contact**: The robot forms a tripod with head + both hands
4. **No Standing Assumption**: Terminations compare to reference, not to upright

### Termination Conditions (Inverted Mode)

- **bad_inverted_contact**: Terminates if pose deviation from reference exceeds thresholds
- **bad_tripod_contact**: (Optional) Terminates if fewer than 2 contact points

## Hardware Requirements

Recommended for training:

- **GPU**: RTX PRO 6000 / RTX 5090 or equivalent
- **CUDA**: 13.x
- **VRAM**: 24+ GB for 4096 environments
- **RAM**: 64+ GB

## Troubleshooting

### "Robot asset path not configured"

Set `SUPERMIMIC_PIPLUS_XML_PATH` to your local MJCF file.

### "Joint order mismatch"

Your MJCF joint order doesn't match the expected order. Either:
1. Reorder joints in your MJCF, or
2. Set `SUPERMIMIC_PIPLUS_JOINT_NAMES` to match your MJCF order

### "Model dimensions mismatch"

Check that your MJCF has exactly:
- 1 freejoint (7 qpos DOF)
- 22 hinge joints

### Contact detection not working

Ensure your MJCF has properly named collision geoms matching the regex patterns.

## Code Example: Full Training Script

```python
#!/usr/bin/env python3
"""Handstand MVP training for Pi Plus."""

import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from supermimic.config import TrainConfig, DataCfg, EnvCfg
from supermimic.robot import get_robot_config
from supermimic.scripts.train import train

def main():
    piplus_cfg = get_robot_config("piplus")
    
    cfg = TrainConfig(
        run_name="piplus_handstand_mvp",
        device="cuda:0",
        data=DataCfg(
            npz_dir=Path("data/npz/piplus_handstand"),
        ),
        env=EnvCfg(
            num_envs=4096,
        ),
    )
    
    # Train with inverted mode
    train(cfg, robot_cfg=piplus_cfg, inverted_mode=True)

if __name__ == "__main__":
    main()
```
