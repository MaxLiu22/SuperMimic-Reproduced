# Pi Plus Handstand MVP

This document describes how to train SuperMimic for handstand (倒立) on the HighTorque Pi Plus humanoid robot.

## Overview

The Pi Plus is a humanoid robot with:
- **22 actuated revolute joints** + floating base
- **qpos dimension**: 29 (freejoint 7 + 22 joints)
- **No wrist DOF**; arms ~±10 Nm torque
- **Handstand pose**: head + both hands tripod contact

## Hardware Requirements

- **GPU**: RTX PRO 6000 or RTX 5090 recommended
- **CUDA**: Version 13.x
- **VRAM**: 16GB+ recommended for large batch sizes

## IMPORTANT: Local Asset Setup

**Pi Plus robot model files (URDF, MJCF, meshes, textures) are NOT included in this repository.**

You must set up the robot assets locally:

### Option 1: Environment Variable (Recommended)

```bash
export SUPERMIMIC_PIPLUS_MJCF=/path/to/your/piplus.xml
```

### Option 2: Default Assets Directory

Place files in:
```
~/.supermimic/assets/hightorque_piplus/
├── piplus.xml          # Main MJCF file
└── meshes/             # Mesh files referenced by MJCF
    ├── pelvis.stl
    ├── torso.stl
    └── ...
```

### Verify Setup

```python
from supermimic.robots.piplus import validate_piplus_mjcf
print(validate_piplus_mjcf())  # Should print True
```

## Motion Data Preparation

### 1. Capture Handstand Motion

Record a handstand motion clip with:
- Head and both hands in contact with ground (tripod)
- Legs extended upward
- Smooth transitions (if including entry/exit)

### 2. Convert to NPZ Format

```python
from supermimic.data.motion_convert import convert_csv_to_npz
from supermimic.robots import get_robot_config

# Get Pi Plus config
piplus_config = get_robot_config("piplus")

# Convert CSV to NPZ
convert_csv_to_npz(
    csv_path="handstand_motion.csv",
    out_path="data/npz/piplus_handstand/clip_001.npz",
    input_fps=120.0,  # Your capture frame rate
    output_fps=50.0,
    robot_config=piplus_config,
)
```

### CSV Format

The CSV should have columns (no header):
- Columns 0-2: Root position (x, y, z) in meters
- Columns 3-6: Root quaternion (x, y, z, w)
- Columns 7-28: 22 joint angles in radians (see `PIPLUS_JOINT_NAMES` for order)

## Training Configuration

### Environment Setup

```python
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env

args = SuperMimicEnvCfgArgs(
    npz_files=["data/npz/piplus_handstand/clip_001.npz"],
    robot_name="piplus",
    pose_type="inverted",  # Key: enables inverted-pose terminations
    
    # Inverted pose specific settings
    inverted_tilt_threshold=0.8,
    require_tripod_contact=True,  # Enable tripod stability checks
    min_hand_contacts=1,
    require_head_contact=True,
    
    # Standard settings
    num_envs=4096,
    episode_length_s=10.0,
    domain_randomization=True,
)

env = make_env(args, device="cuda:0")
```

### World Model Configuration

```python
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.robots import get_robot_config

piplus_config = get_robot_config("piplus")

state_spec_cfg = StateSpecCfg.from_robot_config(piplus_config)
# state_spec_cfg.num_joints == 22
# state_spec_cfg.state_dim == 57 (vs 71 for G1)
# state_spec_cfg.xi_dim == 26 (vs 33 for G1)
```

### Training Script

```bash
python -m supermimic.scripts.train \
    --run_name piplus_handstand_v1 \
    --data.npz_dir data/npz/piplus_handstand \
    --env.num_envs 4096 \
    --schedule.total_iterations 320000 \
    --device cuda:0
```

## Termination Conditions

For handstand (inverted) poses, the following terminations are active:

1. **Root Height**: Terminates if root deviates >0.25m from reference height
2. **Inverted Orientation**: Terminates if orientation deviates from inverted reference
3. **Tripod Stability** (optional): Terminates if hand/head contacts are lost

The key difference from standing is that the tilt check expects the robot to be upside-down relative to gravity.

## Pi Plus Joint Configuration

```python
PIPLUS_JOINT_NAMES = (
    # Left leg (6 DOF)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6 DOF)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Torso (3 DOF)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (4 DOF - no wrist)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # Right arm (4 DOF - no wrist)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
```

## Customizing Pi Plus Configuration

If your Pi Plus MJCF has different joint/body names, edit `src/supermimic/robots/piplus.py`:

```python
# Modify these tuples to match your MJCF
PIPLUS_JOINT_NAMES = (...)
PIPLUS_TRACKED_BODIES = (...)

# Update geom regexes if needed
PIPLUS_CONFIG = RobotConfig(
    ...
    hand_geom_regex=r"^(left|right)_hand.*$",
    head_geom_regex=r"^head.*$",
    ...
)
```

## Troubleshooting

### "MJCF not found" Error

Set the environment variable:
```bash
export SUPERMIMIC_PIPLUS_MJCF=/absolute/path/to/piplus.xml
```

### Joint Mismatch Error

Ensure your MJCF joint names match `PIPLUS_JOINT_NAMES` exactly, in order.

### Tripod Contact Not Detected

1. Verify your MJCF has contact-enabled geoms for hands and head
2. Adjust the geom regex patterns in `PIPLUS_CONFIG`
3. Check that geom `contype` and `conaffinity` allow contact

### Poor Handstand Stability

1. Increase `min_hand_contacts` for stricter stability
2. Adjust `inverted_tilt_threshold` (lower = stricter)
3. Ensure motion data is clean and stable

## Differences from G1

| Feature | G1 | Pi Plus |
|---------|-----|---------|
| Joints | 29 | 22 |
| qpos dim | 36 | 29 |
| qvel dim | 35 | 28 |
| State dim | 71 | 57 |
| Xi dim | 33 | 26 |
| Wrist DOF | Yes | No |
| Arm torque | Higher | ~±10 Nm |

## Citation

If you use this code for Pi Plus handstand research, please cite both SuperMimic and this repository.
