"""Multi-clip motion library held as flat GPU tensors.

All clips are concatenated along a single global frame axis.  A per-frame
``clip_id`` and per-clip ``start`` / ``length`` tables let a batch of
environments index arbitrary clips and frames with plain gathers.

Two storage modes:

* **full** (small sets, e.g. LAFAN1): per frame we keep the reference joints, the pelvis
  state and the world-frame link states of the *tracked* bodies, plus precomputed goal
  features. ~1 KB/frame.
* **compact** (BONES-SEED, 47.5M frames): per frame only the reference state
  ``[root_pos(3), root_quat(4), joint_pos(nj), root_lin_vel_w(3), root_ang_vel_w(3),
  joint_vel(nj)]`` in fp16/fp32; tracked-body states and goal features are computed
  on the fly with the differentiable FK (plan 1.3).

Goal features (per frame, plan D5):
  [joint_pos - q0 (nj), joint_vel (nj), projected gravity of the ref root (3),
   ref root angular velocity in the root frame (3),
   ref root linear velocity in the ref heading frame (3), ref root height (1)]
  Total dim = 2*nj + 10 (68 for G1 with 29 joints, 54 for Pi Plus with 22 joints)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse, yaw_quat


def compute_goal_feat_dim(num_joints: int) -> int:
    """Compute goal feature dimension: 2*nj + 10."""
    return 2 * num_joints + 10


def compute_ref_dim(num_joints: int) -> int:
    """Compute reference state dimension: 3 + 4 + nj + 3 + 3 + nj = 13 + 2*nj."""
    return 13 + 2 * num_joints


# Default values for G1 (29 joints) - kept for backwards compatibility
GOAL_FEAT_DIM = 68  # 2*29 + 10
REF_DIM = 71  # 13 + 2*29


@dataclass
class ClipInfo:
    name: str
    start: int
    length: int

    @property
    def end(self) -> int:  # exclusive
        return self.start + self.length


@dataclass
class RefState:
    root_pos_w: torch.Tensor  # (N,3)
    root_quat_w: torch.Tensor  # (N,4) wxyz
    joint_pos: torch.Tensor  # (N,nj)
    root_lin_vel_w: torch.Tensor  # (N,3)
    root_ang_vel_w: torch.Tensor  # (N,3)
    joint_vel: torch.Tensor  # (N,nj)


@dataclass
class RefBodies:
    pos_w: torch.Tensor  # (N,B,3)
    quat_w: torch.Tensor  # (N,B,4)
    lin_vel_w: torch.Tensor  # (N,B,3)
    ang_vel_w: torch.Tensor  # (N,B,3)


def goal_features(ref: RefState, default_joint_pos: torch.Tensor) -> torch.Tensor:
    q = ref.root_quat_w
    grav = projected_gravity(q)
    ang_b = quat_apply_inverse(q, ref.root_ang_vel_w)
    lin_h = quat_apply_inverse(yaw_quat(q), ref.root_lin_vel_w)
    height = ref.root_pos_w[..., 2:3]
    return torch.cat([ref.joint_pos - default_joint_pos, ref.joint_vel, grav, ang_b, lin_h, height], dim=-1)


class MotionLibrary:
    """Motion library for multi-clip motion reference data.
    
    Supports both G1 (29 joints) and Pi Plus (22 joints) robots.
    Joint dimensions are inferred from the motion data.
    """
    
    def __init__(
        self,
        npz_files: list[str | Path],
        body_names: tuple[str, ...],
        default_joint_pos: torch.Tensor | np.ndarray,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        bin_frames: int = 50,
        compact: bool = False,
        store_dtype: torch.dtype = torch.float16,
        robot_name: str | None = None,
        mjcf_path: str | Path | None = None,
    ):
        """Initialize motion library.
        
        Args:
            npz_files: List of motion npz files or a single .pt library file.
            body_names: Names of bodies to track.
            default_joint_pos: Default joint positions for goal feature computation.
            device: PyTorch device.
            dtype: Data type for computations.
            bin_frames: Number of frames per bin for return-based sampling.
            compact: Use compact storage (compute FK on the fly).
            store_dtype: Storage dtype for compact mode.
            robot_name: Robot config name for compact mode FK (e.g. 'g1', 'pi_plus').
                       Falls back to SUPERMIMIC_ROBOT env var or 'g1'.
            mjcf_path: Explicit MJCF path for compact mode FK. Overrides robot_name.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.body_names = tuple(body_names)
        self.bin_frames = int(bin_frames)
        self.compact = compact
        self.robot_name = robot_name or os.environ.get("SUPERMIMIC_ROBOT", "g1")
        
        if len(npz_files) == 0:
            raise ValueError("MotionLibrary needs at least one clip")
        if isinstance(default_joint_pos, torch.Tensor):
            q0 = default_joint_pos.detach().to(device=self.device, dtype=dtype).reshape(-1)
        else:
            q0 = torch.as_tensor(np.asarray(default_joint_pos), dtype=dtype, device=self.device).reshape(-1)
        self.default_joint_pos = q0
        self.num_joints = q0.shape[0]
        
        # Compute dimensions based on number of joints
        self.goal_feat_dim = compute_goal_feat_dim(self.num_joints)
        self.ref_dim = compute_ref_dim(self.num_joints)

        clips: list[ClipInfo] = []
        start = 0
        body_idx = root_idx = None
        ref_chunks: list[np.ndarray] = []
        body_chunks = {"pos": [], "quat": [], "lin": [], "ang": []}
        
        # A single packed cache (build_library.py) can stand in for the npz list (compact mode only).
        if len(npz_files) == 1 and str(npz_files[0]).endswith(".pt"):
            assert compact, "packed .pt libraries are compact-only"
            pack = torch.load(npz_files[0], map_location="cpu", weights_only=False)
            self._ref = pack["ref"].to(device=self.device, dtype=store_dtype)
            clips = [ClipInfo(n, int(st), int(ln)) for n, st, ln in zip(pack["names"], pack["starts"], pack["lengths"])]
            start = int(self._ref.shape[0])
            # Infer num_joints from loaded ref data
            loaded_ref_dim = self._ref.shape[1]
            self.num_joints = (loaded_ref_dim - 13) // 2
            self.goal_feat_dim = compute_goal_feat_dim(self.num_joints)
            self.ref_dim = loaded_ref_dim
            ref_chunks = None
            
        for f in (npz_files if ref_chunks is not None else []):
            with np.load(f) as d:
                if body_idx is None:
                    names = [str(n) for n in d["body_names"]]
                    body_idx = np.array([names.index(n) for n in self.body_names])
                    root_idx = names.index("pelvis")
                n = int(d["joint_pos"].shape[0])
                nj = d["joint_pos"].shape[1]
                # Verify joint dimensions match
                if nj != self.num_joints:
                    raise ValueError(
                        f"Motion file {f} has {nj} joints but expected {self.num_joints}. "
                        f"Make sure motion data matches the robot configuration."
                    )
                bp, bq, blv, bav = d["body_pos_w"], d["body_quat_w"], d["body_lin_vel_w"], d["body_ang_vel_w"]
                ref = np.concatenate(
                    [bp[:, root_idx], bq[:, root_idx], d["joint_pos"], blv[:, root_idx], bav[:, root_idx], d["joint_vel"]], axis=1
                ).astype(np.float32)
                ref_chunks.append(ref)
                if not compact:
                    body_chunks["pos"].append(bp[:, body_idx])
                    body_chunks["quat"].append(bq[:, body_idx])
                    body_chunks["lin"].append(blv[:, body_idx])
                    body_chunks["ang"].append(bav[:, body_idx])
            clips.append(ClipInfo(Path(f).stem, start, n))
            start += n
        self.clips = clips
        self.num_clips = len(clips)
        self.num_frames = start

        if ref_chunks is not None:
            ref_np = np.concatenate(ref_chunks, 0)
            assert ref_np.shape[1] == self.ref_dim, f"Expected ref_dim {self.ref_dim}, got {ref_np.shape[1]}"
            self._ref = torch.as_tensor(ref_np, device=self.device, dtype=store_dtype if compact else dtype)
            del ref_np, ref_chunks
        if not compact:
            t = lambda arrs: torch.as_tensor(np.concatenate(arrs, 0), dtype=dtype, device=self.device)  # noqa: E731
            self.body_pos_w = t(body_chunks["pos"])
            self.body_quat_w = t(body_chunks["quat"])
            self.body_lin_vel_w = t(body_chunks["lin"])
            self.body_ang_vel_w = t(body_chunks["ang"])
            del body_chunks
            self.goal_feat = goal_features(self._split(self._ref), q0)
        else:
            # Initialize FK for compact mode
            self._init_compact_fk(mjcf_path)

        self.clip_start = torch.tensor([c.start for c in clips], device=self.device, dtype=torch.long)
        self.clip_len = torch.tensor([c.length for c in clips], device=self.device, dtype=torch.long)
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), self.clip_len)
        # Bins of `bin_frames` frames inside each clip (plan D14), numbered globally.
        bins_per_clip = (self.clip_len + self.bin_frames - 1) // self.bin_frames
        self.clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.num_bins = int(bins_per_clip.sum().item())
        bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), bins_per_clip)
        local_bin = torch.arange(self.num_bins, device=self.device) - self.clip_bin_offset[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local_bin * self.bin_frames
        self.bin_end = torch.minimum(self.bin_start + self.bin_frames, self.clip_start[bin_clip] + self.clip_len[bin_clip])
        self.bin_clip = bin_clip

    def _init_compact_fk(self, mjcf_path: str | Path | None = None) -> None:
        """Initialize FK for compact mode."""
        from supermimic.kinematics.humanoid_fk import HumanoidKinematics
        
        if mjcf_path is not None:
            self._fk = HumanoidKinematics.from_mjcf(mjcf_path, device=self.device, dtype=self.dtype)
        else:
            self._fk = HumanoidKinematics.from_robot_config(self.robot_name, device=self.device, dtype=self.dtype)
        self._tracked_idx = self._fk.body_indices(self.body_names)

    # ------------------------------------------------------------------ per-frame access

    def _split(self, r: torch.Tensor) -> RefState:
        """Split reference tensor into RefState components."""
        nj = self.num_joints
        return RefState(
            r[..., 0:3],                      # root_pos_w
            r[..., 3:7],                      # root_quat_w
            r[..., 7:7+nj],                   # joint_pos
            r[..., 7+nj:10+nj],               # root_lin_vel_w
            r[..., 10+nj:13+nj],              # root_ang_vel_w
            r[..., 13+nj:13+2*nj],            # joint_vel
        )

    def ref(self, frame: torch.Tensor) -> RefState:
        return self._split(self._ref[frame].to(self.dtype))

    def ref_raw(self, frame: torch.Tensor) -> torch.Tensor:
        """Raw reference tensor (formerly ref71 for G1)."""
        return self._ref[frame].to(self.dtype)

    def bodies(self, frame: torch.Tensor) -> RefBodies:
        """Tracked-body link states (world) at the given frames."""
        if not self.compact:
            return RefBodies(self.body_pos_w[frame], self.body_quat_w[frame], self.body_lin_vel_w[frame], self.body_ang_vel_w[frame])
        qpos, qvel = self.ref_qpos_qvel(frame)
        with torch.no_grad():
            out = self._fk(qpos, qvel)
        i = self._tracked_idx
        return RefBodies(out["pos"][..., i, :], out["quat"][..., i, :], out["lin_vel"][..., i, :], out["ang_vel"][..., i, :])

    # Convenience column accessors (full fp32 copies; prefer ``ref()`` in hot paths).
    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref[:, 0:3].to(self.dtype)

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref[:, 3:7].to(self.dtype)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref[:, 7:7+self.num_joints].to(self.dtype)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        nj = self.num_joints
        return self._ref[:, 7+nj:10+nj].to(self.dtype)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        nj = self.num_joints
        return self._ref[:, 10+nj:13+nj].to(self.dtype)

    @property
    def joint_vel(self) -> torch.Tensor:
        nj = self.num_joints
        return self._ref[:, 13+nj:13+2*nj].to(self.dtype)

    # ------------------------------------------------------------------ queries

    def bin_of_frame(self, frame: torch.Tensor) -> torch.Tensor:
        clip = self.frame_clip[frame]
        return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return self.clip_last[self.frame_clip[frame]]

    def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
        """(N, k*goal_feat_dim) goal features of frames frame..frame+k-1, clamped at the clip end."""
        last = self.last_frame_of(frame)
        offs = torch.arange(k, device=self.device)
        idx = torch.minimum(frame[:, None] + offs[None, :], last[:, None])  # (N,k)
        if not self.compact:
            return self.goal_feat[idx].reshape(frame.shape[0], k * self.goal_feat_dim)
        feat = goal_features(self.ref(idx.reshape(-1)), self.default_joint_pos)
        return feat.reshape(frame.shape[0], k * self.goal_feat_dim)

    def ref_qpos_qvel(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MuJoCo-layout reference state: qpos (N, nq) and qvel (N, nv; root ang vel in body frame)."""
        r = self.ref(frame)
        qpos = torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos], dim=-1)
        ang_b = quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w)
        qvel = torch.cat([r.root_lin_vel_w, ang_b, r.joint_vel], dim=-1)
        return qpos, qvel

    def summary(self) -> str:
        hours = self.num_frames / 50.0 / 3600.0
        mem = self._ref.numel() * self._ref.element_size() / 1e9
        if not self.compact:
            mem += sum(t.numel() * t.element_size() for t in (self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w, self.goal_feat)) / 1e9
        return f"MotionLibrary({'compact' if self.compact else 'full'}): {self.num_clips} clips, {self.num_frames} frames ({hours:.3f} h @50Hz), {self.num_bins} bins, {mem:.2f} GB on {self.device}, {self.num_joints} joints"

    # Backwards compatibility alias
    def ref71(self, frame: torch.Tensor) -> torch.Tensor:
        """DEPRECATED: Use ref_raw() instead."""
        return self.ref_raw(frame)


def discover_npz(directory: str | Path, limit: int | None = None, names: list[str] | None = None, manifest: str | Path | None = None) -> list[Path]:
    """List npz clips; ``manifest`` (csv with name,keep columns) restricts to keep==1."""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if manifest is not None:
        import csv

        with open(manifest) as f:
            keep = {row["name"] for row in csv.DictReader(f) if row.get("keep", "1") in ("1", "True", "true")}
        files = [f for f in files if f.stem in keep]
    if names is not None:
        wanted = set(names)
        files = [f for f in files if f.stem in wanted]
    if limit is not None:
        files = files[:limit]
    return files
