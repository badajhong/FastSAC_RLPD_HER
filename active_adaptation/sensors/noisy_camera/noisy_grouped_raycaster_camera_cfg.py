from isaaclab.utils import configclass

from ..grouped_ray_caster import GroupedRayCasterCameraCfg
from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera


@configclass
class NoisyGroupedRayCasterCameraCfg(NoisyCameraCfgMixin, GroupedRayCasterCameraCfg):
    """
    Configuration class for the NoisyGroupedRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type = NoisyGroupedRayCasterCamera

    # ── Extrinsics Domain Randomization ──────────────────────────────────────
    extrinsics_pos_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Per-episode Gaussian noise std (m) on camera mount position (x, y, z) relative to parent link.
    Resampled at every env reset. Set to (0,0,0) to disable."""

    extrinsics_rot_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Per-episode Gaussian noise std (rad) on camera mount orientation (roll, pitch, yaw) relative to nominal.
    Resampled at every env reset. Set to (0,0,0) to disable."""

    # ── Intrinsics Domain Randomization ──────────────────────────────────────
    intrinsics_focal_length_noise_std: float = 0.0
    """Per-episode Gaussian noise std on focal_length (same units as pattern_cfg.focal_length).
    Resampled at every env reset. Set to 0.0 to disable."""

    intrinsics_aperture_noise_std: float = 0.0
    """Per-episode Gaussian noise std on horizontal_aperture (same units as pattern_cfg.horizontal_aperture).
    Resampled at every env reset. Set to 0.0 to disable."""
