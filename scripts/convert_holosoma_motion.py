#!/usr/bin/env python3
"""Convert a self-describing Holosoma motion archive to VAIC's motion format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_R1_POSE_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "active_adaptation"
    / "assets"
    / "r1_default_pose.npz"
)

VAIC_CONTACT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(quaternions.dtype).eps):
        raise ValueError("cannot normalize a zero-length quaternion")
    return quaternions / norms


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply WXYZ quaternions with NumPy broadcasting."""
    left = np.asarray(left)
    right = np.asarray(right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate_wxyz(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by WXYZ quaternions with NumPy broadcasting."""
    quaternion = _normalize_quaternions(np.asarray(quaternion))
    vectors = np.asarray(vectors)
    vector_part = quaternion[..., 1:]
    twice_cross = 2.0 * np.cross(vector_part, vectors)
    return (
        vectors + quaternion[..., :1] * twice_cross + np.cross(vector_part, twice_cross)
    )


def _yaw_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return the yaw-only component of one WXYZ quaternion."""
    w, x, y, z = _normalize_quaternions(np.asarray(quaternion)).tolist()
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32
    )


def _slerp_wxyz(start: np.ndarray, end: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Shortest-arc SLERP for a time sequence of WXYZ quaternions."""
    raw_start = np.asarray(start)
    raw_end = np.asarray(end)
    identical = np.all(raw_start == raw_end, axis=-1)
    start = _normalize_quaternions(raw_start)
    end = _normalize_quaternions(raw_end)
    alphas = np.asarray(alphas, dtype=start.dtype)
    if alphas.size == 0:
        return np.empty((0, *start.shape), dtype=start.dtype)

    dot = np.sum(start * end, axis=-1)
    end = np.where((dot < 0.0)[..., None], -end, end)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    close = sin_theta < 1.0e-6
    safe_sin_theta = np.where(close, 1.0, sin_theta)

    alpha_shape = (alphas.shape[0],) + (1,) * start.ndim
    alpha = alphas.reshape(alpha_shape)
    theta_t = theta[None, ..., None]
    denominator = safe_sin_theta[None, ..., None]
    spherical = (
        np.sin((1.0 - alpha) * theta_t) / denominator * start[None]
        + np.sin(alpha * theta_t) / denominator * end[None]
    )
    linear = (1.0 - alpha) * start[None] + alpha * end[None]
    result = np.where(close[None, ..., None], linear, spherical)
    result = _normalize_quaternions(result).astype(start.dtype, copy=False)
    # Object start/target quaternions are the exact same boundary sample.
    # Preserve that stored sample bit-for-bit instead of renormalizing it.
    return np.where(identical[None, ..., None], raw_start[None], result)


def _load_default_pose_reference(path: Path) -> dict[str, Any]:
    """Load a simulator-FK reference pose used to anchor synthetic transitions."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "body_names",
            "body_pos_w",
            "body_quat_w",
            "joint_names",
            "joint_pos",
            "root_body_name",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"default-pose reference is missing fields: {missing}")
        reference: dict[str, Any] = {key: archive[key] for key in archive.files}

    reference["body_names"] = [str(name) for name in reference["body_names"].tolist()]
    reference["joint_names"] = [str(name) for name in reference["joint_names"].tolist()]
    reference["root_body_name"] = str(np.asarray(reference["root_body_name"]).item())
    body_count = len(reference["body_names"])
    joint_count = len(reference["joint_names"])
    if np.asarray(reference["body_pos_w"]).shape != (body_count, 3):
        raise ValueError("default body_pos_w does not match reference body_names")
    if np.asarray(reference["body_quat_w"]).shape != (body_count, 4):
        raise ValueError("default body_quat_w does not match reference body_names")
    if np.asarray(reference["joint_pos"]).shape != (joint_count,):
        raise ValueError("default joint_pos does not match reference joint_names")
    if reference["root_body_name"] not in reference["body_names"]:
        raise ValueError("default root_body_name is absent from reference body_names")

    reference["body_pos_w"] = np.asarray(reference["body_pos_w"], dtype=np.float32)
    reference["body_quat_w"] = _normalize_quaternions(
        np.asarray(reference["body_quat_w"], dtype=np.float32)
    )
    reference["joint_pos"] = np.asarray(reference["joint_pos"], dtype=np.float32)
    return reference


def _anchored_default_state(
    motion: dict[str, np.ndarray],
    body_names: list[str],
    joint_names: list[str],
    reference: dict[str, Any],
    boundary_index: int,
) -> dict[str, np.ndarray]:
    """Anchor the canonical default pose to one motion boundary like Holosoma."""
    reference_joint_names = reference["joint_names"]
    if set(reference_joint_names) != set(joint_names):
        missing = sorted(set(joint_names).difference(reference_joint_names))
        extra = sorted(set(reference_joint_names).difference(joint_names))
        raise ValueError(
            "default-pose joint names do not match the motion "
            f"(missing={missing}, extra={extra})"
        )

    root_name = reference["root_body_name"]
    if root_name not in body_names:
        raise ValueError(
            f"motion does not contain default-pose root body {root_name!r}"
        )
    root_motion_index = body_names.index(root_name)
    root_reference_index = reference["body_names"].index(root_name)
    boundary_root_pos = motion["body_pos_w"][boundary_index, root_motion_index]
    boundary_root_quat = motion["body_quat_w"][boundary_index, root_motion_index]
    boundary_yaw_quat = _yaw_quaternion_wxyz(boundary_root_quat)

    reference_root_pos = reference["body_pos_w"][root_reference_index]
    reference_yaw_quat = _yaw_quaternion_wxyz(
        reference["body_quat_w"][root_reference_index]
    )
    reference_yaw_inverse = reference_yaw_quat.copy()
    reference_yaw_inverse[1:] *= -1.0
    yaw_delta = _quat_multiply_wxyz(boundary_yaw_quat, reference_yaw_inverse)
    anchored_root_pos = np.asarray(
        [boundary_root_pos[0], boundary_root_pos[1], reference_root_pos[2]],
        dtype=np.float32,
    )

    reference_relative_pos = reference["body_pos_w"] - reference_root_pos
    anchored_reference_pos = anchored_root_pos + _quat_rotate_wxyz(
        yaw_delta, reference_relative_pos
    )
    anchored_reference_quat = _normalize_quaternions(
        _quat_multiply_wxyz(yaw_delta, reference["body_quat_w"])
    ).astype(np.float32)

    # Unknown bodies are objects, not robot links. Holosoma holds each object's
    # complete boundary state constant throughout the synthetic transition.
    default_state = {
        key: motion[key][boundary_index].copy()
        for key in (
            "body_pos_w",
            "body_lin_vel_w",
            "body_quat_w",
            "body_ang_vel_w",
            "joint_pos",
            "joint_vel",
        )
    }
    for reference_index, name in enumerate(reference["body_names"]):
        if name not in body_names:
            continue
        motion_index = body_names.index(name)
        default_state["body_pos_w"][motion_index] = anchored_reference_pos[
            reference_index
        ]
        default_state["body_quat_w"][motion_index] = anchored_reference_quat[
            reference_index
        ]
        default_state["body_lin_vel_w"][motion_index] = 0.0
        default_state["body_ang_vel_w"][motion_index] = 0.0

    reference_joint_index = {name: i for i, name in enumerate(reference_joint_names)}
    default_state["joint_pos"] = np.asarray(
        [reference["joint_pos"][reference_joint_index[name]] for name in joint_names],
        dtype=np.float32,
    )
    default_state["joint_vel"] = np.zeros_like(default_state["joint_pos"])
    return default_state


def _transition_steps(duration_s: float, fps: float) -> int:
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError(
            f"transition duration must be finite and non-negative, got {duration_s}"
        )
    steps = round(duration_s * fps)
    # Match Holosoma, which skips an interpolation segment of zero or one step.
    return steps if steps > 1 else 0


def _transition_segment(
    start: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    segment: dict[str, np.ndarray] = {}
    for key in (
        "body_pos_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_pos",
        "joint_vel",
    ):
        alpha = alphas.reshape((len(alphas),) + (1,) * start[key].ndim)
        segment[key] = (
            start[key][None] + alpha * (target[key] - start[key])[None]
        ).astype(np.float32)
    segment["body_quat_w"] = _slerp_wxyz(
        start["body_quat_w"], target["body_quat_w"], alphas
    ).astype(np.float32)
    return segment


def augment_default_pose_transitions(
    motion: dict[str, np.ndarray],
    body_names: list[str],
    joint_names: list[str],
    reference: dict[str, Any] | None,
    *,
    fps: float,
    prepend_duration_s: float = 2.0,
    append_duration_s: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bake Holosoma's per-clip default-pose interpolation into VAIC arrays."""
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    original_frames = int(motion["joint_pos"].shape[0])
    if original_frames == 0:
        raise ValueError("cannot augment an empty motion")

    prepend_steps = _transition_steps(prepend_duration_s, fps)
    append_steps = _transition_steps(append_duration_s, fps)
    if (prepend_steps or append_steps) and reference is None:
        raise ValueError(
            "a default-pose reference is required when transitions are enabled"
        )
    original = {key: value.copy() for key, value in motion.items()}

    segments_before: dict[str, np.ndarray] | None = None
    if prepend_steps:
        assert reference is not None
        default_start = _anchored_default_state(
            original, body_names, joint_names, reference, boundary_index=0
        )
        motion_start = {
            key: original[key][0]
            for key in (
                "body_pos_w",
                "body_lin_vel_w",
                "body_quat_w",
                "body_ang_vel_w",
                "joint_pos",
                "joint_vel",
            )
        }
        prepend_alphas = np.linspace(0.0, 1.0, prepend_steps + 1, dtype=np.float32)[:-1]
        segments_before = _transition_segment(
            default_start, motion_start, prepend_alphas
        )

    segments_after: dict[str, np.ndarray] | None = None
    if append_steps:
        assert reference is not None
        motion_end = {
            key: original[key][-1]
            for key in (
                "body_pos_w",
                "body_lin_vel_w",
                "body_quat_w",
                "body_ang_vel_w",
                "joint_pos",
                "joint_vel",
            )
        }
        default_end = _anchored_default_state(
            original, body_names, joint_names, reference, boundary_index=-1
        )
        append_alphas = np.linspace(0.0, 1.0, append_steps + 1, dtype=np.float32)[1:]
        segments_after = _transition_segment(motion_end, default_end, append_alphas)

    augmented: dict[str, np.ndarray] = {}
    time_major_keys = (
        "body_pos_w",
        "body_lin_vel_w",
        "body_quat_w",
        "body_ang_vel_w",
        "joint_pos",
        "joint_vel",
    )
    for key in time_major_keys:
        chunks = []
        if segments_before is not None:
            chunks.append(segments_before[key])
        chunks.append(original[key])
        if segments_after is not None:
            chunks.append(segments_after[key])
        augmented[key] = np.concatenate(chunks, axis=0)

    contact_chunks = []
    if prepend_steps:
        contact_chunks.append(
            np.zeros((prepend_steps, original["body_contact"].shape[1]), dtype=bool)
        )
    contact_chunks.append(original["body_contact"])
    if append_steps:
        contact_chunks.append(
            np.zeros((append_steps, original["body_contact"].shape[1]), dtype=bool)
        )
    augmented["body_contact"] = np.concatenate(contact_chunks, axis=0)

    transition_meta: dict[str, Any] = {
        "semantics": "holosoma_wbt_default_pose",
        "prepend_duration_s": prepend_duration_s,
        "append_duration_s": append_duration_s,
        "prepend_frames": prepend_steps,
        "append_frames": append_steps,
        "original_frames": original_frames,
        "output_frames": original_frames + prepend_steps + append_steps,
        "real_motion_range": [prepend_steps, prepend_steps + original_frames],
        "object_behavior": "hold_boundary_state",
        "object_contact_behavior": "false_during_synthetic_transitions",
        "feet_contact_behavior": "recomputed_from_augmented_body_positions",
    }
    return augmented, transition_meta


def evenly_spaced_points(points: np.ndarray, count: int = 128) -> np.ndarray:
    """Use Holosoma's deterministic linspace-style point reduction."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"object points must have shape (N, 3), got {points.shape}")
    if len(points) < count:
        raise ValueError(f"need at least {count} object points, got {len(points)}")
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _feet_contact(
    body_pos_w: np.ndarray, body_names: list[str], threshold: float
) -> np.ndarray:
    contact = []
    for side in ("left", "right"):
        names = [
            f"{side}_foot_front_outer_link",
            f"{side}_foot_front_inner_link",
            f"{side}_foot_rear_outer_link",
            f"{side}_foot_rear_inner_link",
        ]
        missing = [name for name in names if name not in body_names]
        if missing:
            raise ValueError(
                f"cannot estimate {side} foot contact; missing bodies: {missing}"
            )
        indices = [body_names.index(name) for name in names]
        contact.append(body_pos_w[:, indices, 2].min(axis=1) <= threshold)
    return np.stack(contact, axis=1)


def convert_motion(
    source: Path,
    object_points_path: Path,
    *,
    foot_height_threshold: float = 0.015,
    default_pose_reference: Path | None = None,
    default_pose_prepend_duration_s: float = 2.0,
    default_pose_append_duration_s: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Normalize one Holosoma archive entirely in memory."""
    with np.load(source, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}

    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
        "contact_object_label",
        "contact_object_names",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise KeyError(f"Holosoma archive is missing required fields: {missing}")

    joint_names = [str(name) for name in raw["joint_names"].tolist()]
    body_names = [str(name) for name in raw["body_names"].tolist()]
    frame_count = int(raw["body_pos_w"].shape[0])
    if raw["joint_pos"].shape != (frame_count, len(joint_names) + 7):
        raise ValueError(
            "joint_pos must contain root xyz+wxyz followed by named joints; "
            f"got {raw['joint_pos'].shape} for {len(joint_names)} names"
        )
    if raw["joint_vel"].shape != (frame_count, len(joint_names) + 6):
        raise ValueError(
            "joint_vel must contain root linear+angular velocity followed by named joints; "
            f"got {raw['joint_vel'].shape} for {len(joint_names)} names"
        )
    if not body_names or body_names[0] != "world":
        raise ValueError("expected the first Holosoma body column to be 'world'")

    normalized_body_names = body_names[1:]
    body_pos_w = raw["body_pos_w"][:, 1:]
    body_quat_w = raw["body_quat_w"][:, 1:]
    body_lin_vel_w = raw["body_lin_vel_w"][:, 1:]
    body_ang_vel_w = raw["body_ang_vel_w"][:, 1:]
    expected_body_shape = (frame_count, len(normalized_body_names))
    for key, array, width in (
        ("body_pos_w", body_pos_w, 3),
        ("body_quat_w", body_quat_w, 4),
        ("body_lin_vel_w", body_lin_vel_w, 3),
        ("body_ang_vel_w", body_ang_vel_w, 3),
    ):
        if array.shape != (*expected_body_shape, width):
            raise ValueError(f"{key} has unexpected shape {array.shape}")

    source_contact_names = [str(name) for name in raw["contact_object_names"].tolist()]
    source_contact = np.asarray(raw["contact_object_label"], dtype=bool)
    if source_contact.shape != (frame_count, len(source_contact_names)):
        raise ValueError(
            "contact_object_label width does not match contact_object_names: "
            f"{source_contact.shape} vs {len(source_contact_names)}"
        )

    # Preserve direct ankle/object contacts for motions such as plastic-box
    # pushing. These slots can be selected as contact EEFs by the task config.
    body_contact = np.zeros((frame_count, len(VAIC_CONTACT_BODY_NAMES)), dtype=bool)
    for side, destination in (("left", 0), ("right", 1)):
        source_name = f"{side}_ankle_roll_link"
        if source_name in source_contact_names:
            body_contact[:, destination] = source_contact[
                :, source_contact_names.index(source_name)
            ]

    # Holosoma's virtual hand markers are fixed +0.1 m along each wrist-roll
    # link. Use those labels but place them in VAIC's wrist-roll contact slots.
    for side, destination in (("left", 2), ("right", 5)):
        candidates = [f"{side}_hand_contact_link", f"{side}_wrist_roll_link"]
        source_name = next(
            (name for name in candidates if name in source_contact_names), None
        )
        if source_name is None:
            raise ValueError(f"missing a Holosoma {side}-hand contact label")
        body_contact[:, destination] = source_contact[
            :, source_contact_names.index(source_name)
        ]

    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    motion = {
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        # Both Holosoma and VAIC store quaternions in WXYZ order.
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
        "joint_pos": raw["joint_pos"][:, 7:].astype(np.float32),
        "joint_vel": raw["joint_vel"][:, 6:].astype(np.float32),
        "body_contact": body_contact,
    }

    prepend_steps = _transition_steps(default_pose_prepend_duration_s, fps)
    append_steps = _transition_steps(default_pose_append_duration_s, fps)
    reference_path: Path | None = None
    reference: dict[str, Any] | None = None
    if prepend_steps or append_steps:
        reference_path = (
            default_pose_reference.expanduser().resolve()
            if default_pose_reference is not None
            else DEFAULT_R1_POSE_REFERENCE
        )
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"default-pose reference does not exist: {reference_path}"
            )
        reference = _load_default_pose_reference(reference_path)

    motion, transition_meta = augment_default_pose_transitions(
        motion,
        normalized_body_names,
        joint_names,
        reference,
        fps=fps,
        prepend_duration_s=default_pose_prepend_duration_s,
        append_duration_s=default_pose_append_duration_s,
    )
    transition_meta["reference"] = (
        str(reference_path) if reference_path is not None else None
    )

    object_points = evenly_spaced_points(
        np.load(object_points_path, allow_pickle=False)
    )
    motion["object_contact"] = motion["body_contact"].any(axis=1, keepdims=True)
    motion["feet_contact"] = _feet_contact(
        motion["body_pos_w"], normalized_body_names, foot_height_threshold
    )
    motion["object_points"] = object_points[None]
    meta: dict[str, object] = {
        "body_names": normalized_body_names,
        "joint_names": joint_names,
        "fps": fps,
        "source": str(source.resolve()),
        "conversion": {
            "format": "holosoma_to_vaic_v2",
            "quaternion_order": "wxyz",
            "foot_height_threshold": foot_height_threshold,
            "object_points_source": str(object_points_path.resolve()),
            "object_points_selection": "linspace_128",
            "default_pose_transition": transition_meta,
        },
    }
    return motion, meta


def write_motion(
    output_dir: Path, motion: dict[str, np.ndarray], meta: dict[str, object]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "motion.npz", **motion)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
        stream.write("\n")


def _default_object_points(source: Path) -> Path:
    # .../train_r1/rl/<object>/<clip>.npz -> .../train_r1/objects/<object>/sample_points.npy
    if len(source.parents) < 4:
        raise ValueError(f"cannot infer object-points path from {source}")
    object_name = source.parent.name
    primary = source.parents[2] / "objects" / object_name / "sample_points.npy"

    # Some copied train_r1 clouds are stale. Suitcase, for example, has source
    # contact indices up to 595 but its copied cloud contains only 340 points.
    # Prefer the canonical retargeting cloud when the source index range or
    # recorded target coordinates reject the nearby copy.
    required_point_count = 0
    source_indices: np.ndarray | None = None
    source_target_points: np.ndarray | None = None
    source_target_valid: np.ndarray | None = None
    with np.load(source, allow_pickle=False) as archive:
        if "contact_object_target_indices" in archive:
            source_indices = np.asarray(archive["contact_object_target_indices"])
            valid_indices = source_indices[source_indices >= 0]
            if valid_indices.size:
                required_point_count = int(valid_indices.max()) + 1
        if "contact_object_target_points_obj" in archive:
            source_target_points = np.asarray(
                archive["contact_object_target_points_obj"]
            )
        if "contact_object_target_valid" in archive:
            source_target_valid = np.asarray(
                archive["contact_object_target_valid"], dtype=bool
            )

    def is_compatible(candidate: Path) -> bool:
        if not candidate.is_file():
            return False
        points = np.asarray(np.load(candidate, allow_pickle=False))
        if points.ndim != 2 or points.shape[1] != 3:
            return False
        if points.shape[0] < required_point_count:
            return False
        if source_indices is None or source_target_points is None:
            return True
        if source_target_points.shape != (*source_indices.shape, 3):
            return False

        valid = source_indices >= 0
        if source_target_valid is not None:
            expected_valid_shape = source_indices.shape[:-1]
            if source_target_valid.shape != expected_valid_shape:
                return False
            valid &= source_target_valid[..., None]
        if not valid.any():
            return True
        return bool(
            np.allclose(
                points[source_indices[valid]],
                source_target_points[valid],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        )

    if is_compatible(primary):
        return primary

    hoi_root = source.parents[3]
    canonical = (
        hoi_root
        / "src/holosoma_retargeting/holosoma_retargeting/models/objects"
        / object_name
        / "sample_points.npy"
    )
    if is_compatible(canonical):
        return canonical

    raise FileNotFoundError(
        "could not infer an object point cloud compatible with source contact "
        "indices/target coordinates "
        f"(need at least {required_point_count} points); pass --object-points"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Holosoma source .npz")
    parser.add_argument("output_dir", type=Path, help="VAIC motion directory to create")
    parser.add_argument(
        "--object-points",
        type=Path,
        default=None,
        help="object-local sample_points.npy (inferred from the source path by default)",
    )
    parser.add_argument("--foot-height-threshold", type=float, default=0.015)
    parser.add_argument(
        "--default-pose-reference",
        type=Path,
        default=None,
        help=(f"canonical FK default-pose .npz (default: {DEFAULT_R1_POSE_REFERENCE})"),
    )
    parser.add_argument(
        "--default-pose-prepend-duration-s",
        type=float,
        default=2.0,
        help="seconds of default-to-motion interpolation (default: 2.0)",
    )
    parser.add_argument(
        "--default-pose-append-duration-s",
        type=float,
        default=2.0,
        help="seconds of motion-to-default interpolation (default: 2.0)",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    object_points = (
        args.object_points.expanduser().resolve()
        if args.object_points is not None
        else _default_object_points(source)
    )
    motion, meta = convert_motion(
        source,
        object_points,
        foot_height_threshold=args.foot_height_threshold,
        default_pose_reference=args.default_pose_reference,
        default_pose_prepend_duration_s=args.default_pose_prepend_duration_s,
        default_pose_append_duration_s=args.default_pose_append_duration_s,
    )
    write_motion(args.output_dir, motion, meta)
    print(
        f"Converted {source} -> {args.output_dir.resolve()} "
        f"({motion['joint_pos'].shape[0]} frames, {len(meta['joint_names'])} joints)"
    )


if __name__ == "__main__":
    main()
