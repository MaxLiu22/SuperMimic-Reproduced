"""Environment layer: mjlab ManagerBasedRlEnv configured for SuperMimic.

Supports multiple robot configurations (G1, Pi Plus) and pose modes
(standing, handstand, reference-tracking).
"""

from supermimic.envs.tracking_env_cfg import (
    SuperMimicEnvCfgArgs,
    make_env,
    make_env_cfg,
    make_handstand_env,
    make_handstand_env_cfg,
)

__all__ = [
    "SuperMimicEnvCfgArgs",
    "make_env",
    "make_env_cfg",
    "make_handstand_env",
    "make_handstand_env_cfg",
]
