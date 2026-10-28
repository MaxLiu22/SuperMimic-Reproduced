# Pi Plus Handstand MVP Setup Guide

This document describes how to configure SuperMimic for HighTorque Pi Plus handstand training.

## Overview

The Pi Plus adaptation adds:
- Robot abstraction layer supporting multiple humanoid robots
- Pi Plus configuration (22 actuated joints, no wrist DOF)
- Handstand-aware terminations (inverted pose support)
- Parameterized asset paths (no proprietary meshes in repo)

## Hardware Requirements

- 1× NVIDIA RTX PRO 6000 or RTX 5090
- CUDA 13+ driver
- ~24GB VRAM for full training

## Environment Variables

### Required for Pi Plus

```bash
# Path to your local Pi Plus MJCF/XML file (with meshes resolved)
export SUPERMIMIC_PI_PLUS_MJCF="/path/to/pi_plus/mjcf/pi_plus.xml"

# Default robot for training
export SUPERMIMIC_ROBOT="pi_plus"
```

### Optional for G1 (if not using mjlab default)

```bash
export SUPERMIMIC_G1_MJCF="/path/to/g1/mjcf/g1.xml"
```

## Motion Reference Data

Motion reference files (`.npz`) must be provided locally. They are NOT included in this repository.

### Expected NPZ Format

Each motion file should contain:
- `fps`: Frame rate (typically 50 Hz)
- `joint_pos`: (T, 22) joint positions for Pi Plus
- `joint_vel`: (T, 22) joint velocities
- `body_pos_w`: (T, nb, 3) body positions in world frame
- `body_quat_w`: (T, nb, 4) body quaternions (wxyz)
- `body_lin_vel_w`: (T, nb, 3) body linear velocities
- `body_ang_vel_w`: (T, nb, 3) body angular velocities
- `joint_names`: Array of joint name strings
- `body_names`: Array of body name strings

### Directory Structure

```
/your/local/data/
├── handstand_motions/
│   ├── handstand_clean_01.npz
│   ├── handstand_clean_02.npz
│   └── ...
└── pi_plus_mjcf/
    ├── pi_plus.xml
    └── meshes/
        └── ...
```

## Running Handstand Training

### Basic Training Command

```bash
cd /workspace

# Set environment variables
export SUPERMIMIC_PI_PLUS_MJCF="/your/path/pi_plus.xml"
export SUPERMIMIC_ROBOT="pi_plus"

# Run training
python -m supermimic.scripts.train \
    --run-name pi_plus_handstand_v1 \
    --data.npz-dir /your/path/handstand_motions \
    --env.num-envs 4096
```

### Using the Handstand Environment Factory

```python
from supermimic.envs.tracking_env_cfg import make_handstand_env

# Create handstand environment with Pi Plus
env = make_handstand_env(
    npz_files=[
        "/your/path/handstand_clean_01.npz",
        "/your/path/handstand_clean_02.npz",
    ],
    robot_name="pi_plus",
    mjcf_path="/your/path/pi_plus.xml",  # Optional if env var is set
    device="cuda:0",
    num_envs=4096,
)
```

### Custom Training Configuration

```python
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env

args = SuperMimicEnvCfgArgs(
    npz_files=["path/to/handstand_01.npz"],
    num_envs=4096,
    robot_name="pi_plus",
    mjcf_path="/your/path/pi_plus.xml",
    
    # Handstand-specific settings
    pose_mode="handstand",  # Disables standing-only terminations
    root_height_threshold=0.35,  # Relaxed for inverted balance
    root_tilt_threshold=0.9,  # More lenient for handstand orientation
    
    # Training settings
    episode_length_s=10.0,
    domain_randomization=True,
)

env = make_env(args, device="cuda:0")
```

## Pi Plus Robot Specification

### Joint Configuration (22 DOF)

| Group | Joints | DOF |
|-------|--------|-----|
| Left Leg | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll | 6 |
| Right Leg | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll | 6 |
| Waist | yaw, roll, pitch | 3 |
| Left Arm | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 4 |
| Right Arm | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 4 |

**Note:** Pi Plus has NO wrist DOF (unlike G1's 7 DOF arms).

### Effort Limits

- Leg joints: 50-139 Nm
- Waist joints: 88 Nm
- Arm joints: **~10 Nm** (limited torque)

### State Dimensions

| Dimension | G1 (29 joints) | Pi Plus (22 joints) |
|-----------|----------------|---------------------|
| qpos | 36 (7 + 29) | 29 (7 + 22) |
| qvel | 35 (6 + 29) | 28 (6 + 22) |
| State | 71 | 57 |
| Goal feat | 68 | 54 |

### Handstand Contact Bodies

For tripod support:
- Head: `head`
- Left hand: `left_hand_link`
- Right hand: `right_hand_link`

## Converting Motion Data

If you have CSV motion data, convert it to NPZ format:

```python
from supermimic.data.motion_convert import convert_csv_to_npz, load_robot_model

# Load Pi Plus model for FK
model = load_robot_model(robot_name="pi_plus")

# Convert CSV to NPZ
convert_csv_to_npz(
    csv_path="path/to/handstand.csv",
    out_path="path/to/handstand.npz",
    input_fps=120.0,  # Your recording FPS
    output_fps=50.0,  # Standard training FPS
    model=model,
)
```

## Troubleshooting

### "MJCF path not configured"

Set the environment variable:
```bash
export SUPERMIMIC_PI_PLUS_MJCF="/path/to/your/pi_plus.xml"
```

### "Joint dimension mismatch"

Ensure your motion NPZ files have 22 joints (not 29). The motion data must match the Pi Plus kinematic structure.

### "Out of memory"

Reduce `num_envs`:
```bash
python -m supermimic.scripts.train --env.num-envs 2048
```

### "Contact bodies not found"

Verify your Pi Plus MJCF has bodies named:
- `head`
- `left_hand_link`
- `right_hand_link`
- `pelvis`
- `torso_link`

If your MJCF uses different names, update `supermimic/robots/pi_plus.py`.

## Architecture Notes

### Robot Abstraction

```
supermimic/robots/
├── __init__.py      # Exports
├── base.py          # RobotConfig base class, registry
├── g1.py            # Unitree G1 config
└── pi_plus.py       # HighTorque Pi Plus config
```

### Key Abstractions

1. **RobotConfig**: Base class defining robot interface
   - Joint names, body names, dimensions
   - Asset path resolution via env vars
   - Action scaling, effort limits

2. **HumanoidKinematics**: Generic batched FK
   - Works with any MuJoCo humanoid model
   - No hardcoded robot-specific code

3. **Pose Mode**: Termination handling
   - `"reference"`: Track reference pose (default)
   - `"standing"`: Traditional upright terminations
   - `"handstand"`: Inverted pose support

## Next Steps

1. Prepare clean handstand motion reference clips
2. Ensure MJCF resolves all mesh paths correctly
3. Run single-clip MVP training
4. Evaluate tracking performance
5. Iterate on hyperparameters if needed
