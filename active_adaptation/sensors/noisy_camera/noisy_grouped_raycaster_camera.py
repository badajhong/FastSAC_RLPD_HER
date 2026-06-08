from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera import CameraData

from ..grouped_ray_caster import GroupedRayCasterCamera
from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_grouped_raycaster_camera_cfg import NoisyGroupedRayCasterCameraCfg


def _euler_xyz_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert roll/pitch/yaw (rad) tensors of shape [N] to quaternion [N, 4] (w, x, y, z)."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


class NoisyGroupedRayCasterCamera(NoisyCameraMixin, GroupedRayCasterCamera):
    cfg: NoisyGroupedRayCasterCameraCfg

    def _initialize_impl(self):
        super()._initialize_impl()  # type: ignore
        self.build_noise_pipeline()
        self.build_history_buffers()
        self._build_camera_dr_buffers()

    def _build_camera_dr_buffers(self):
        """Store nominal offset and intrinsic matrices for per-episode DR."""
        self._nominal_offset_pos = self._offset_pos.clone()    # [N, 3]
        self._nominal_offset_quat = self._offset_quat.clone()  # [N, 4]
        self._nominal_intrinsic_matrices = self._data.intrinsic_matrices.clone()  # [N, 3, 3]

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the sensor and noise pipeline."""
        super().reset(env_ids)
        self.reset_noise_pipeline(env_ids)
        self.reset_history_buffers(env_ids)
        self._resample_camera_dr(env_ids)

    def _resample_camera_dr(self, env_ids: Sequence[int] | None = None):
        """Resample per-episode extrinsics and intrinsics DR for given env_ids."""
        if env_ids is None:
            env_ids = self._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self._device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        n = len(env_ids)

        # ── Extrinsics DR ────────────────────────────────────────────────────
        pos_std = self.cfg.extrinsics_pos_noise_std
        rot_std = self.cfg.extrinsics_rot_noise_std

        if any(s > 0.0 for s in pos_std):
            std_t = torch.tensor(pos_std, device=self._device, dtype=torch.float32)
            pos_noise = torch.randn(n, 3, device=self._device) * std_t
            self._offset_pos[env_ids] = self._nominal_offset_pos[env_ids] + pos_noise

        if any(s > 0.0 for s in rot_std):
            std_t = torch.tensor(rot_std, device=self._device, dtype=torch.float32)
            rpy = torch.randn(n, 3, device=self._device) * std_t
            noise_quat = _euler_xyz_to_quat(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            self._offset_quat[env_ids] = math_utils.quat_mul(
                self._nominal_offset_quat[env_ids], noise_quat
            )

        # ── Intrinsics DR ────────────────────────────────────────────────────
        fl_std = self.cfg.intrinsics_focal_length_noise_std
        ap_std = self.cfg.intrinsics_aperture_noise_std

        if fl_std > 0.0 or ap_std > 0.0:
            perturbed_K = self._nominal_intrinsic_matrices[env_ids].clone()  # [n, 3, 3]

            if fl_std > 0.0:
                nominal_fl = self.cfg.pattern_cfg.focal_length
                delta_fl = torch.randn(n, device=self._device) * fl_std
                scale = (nominal_fl + delta_fl) / nominal_fl  # [n]
                perturbed_K[:, 0, 0] = perturbed_K[:, 0, 0] * scale
                perturbed_K[:, 1, 1] = perturbed_K[:, 1, 1] * scale

            if ap_std > 0.0:
                nominal_hap = self.cfg.pattern_cfg.horizontal_aperture
                nominal_vap = self.cfg.pattern_cfg.vertical_aperture
                delta_hap = torch.randn(n, device=self._device) * ap_std
                # keep aspect ratio: vertical aperture noise proportional to horizontal
                delta_vap = delta_hap * (nominal_vap / nominal_hap)
                scale_x = nominal_hap / (nominal_hap + delta_hap)
                scale_y = nominal_vap / (nominal_vap + delta_vap)
                perturbed_K[:, 0, 0] = perturbed_K[:, 0, 0] * scale_x
                perturbed_K[:, 1, 1] = perturbed_K[:, 1, 1] * scale_y

            self._data.intrinsic_matrices[env_ids] = perturbed_K

            # recompute ray_starts / ray_directions for all envs (pattern func is batched)
            ray_starts_all, ray_dirs_all = self.cfg.pattern_cfg.func(
                self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
            )
            self.ray_starts[env_ids] = ray_starts_all[env_ids]
            self.ray_directions[env_ids] = ray_dirs_all[env_ids]

    """
    Implementation
    """

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        super()._update_buffers_impl(env_ids)
        self.apply_noise_pipeline_to_all_data_types(env_ids)
        self.update_history_buffers(env_ids)
