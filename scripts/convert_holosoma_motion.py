#!/usr/bin/env python3
"""Convert a self-describing Holosoma motion archive to VAIC's motion format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def evenly_spaced_points(points: np.ndarray, count: int = 128) -> np.ndarray:
    """Use Holosoma's deterministic linspace-style point reduction."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"object points must have shape (N, 3), got {points.shape}")
    if len(points) < count:
        raise ValueError(f"need at least {count} object points, got {len(points)}")
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _feet_contact(body_pos_w: np.ndarray, body_names: list[str], threshold: float) -> np.ndarray:
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
            raise ValueError(f"cannot estimate {side} foot contact; missing bodies: {missing}")
        indices = [body_names.index(name) for name in names]
        contact.append(body_pos_w[:, indices, 2].min(axis=1) <= threshold)
    return np.stack(contact, axis=1)


def convert_motion(
    source: Path,
    object_points_path: Path,
    *,
    foot_height_threshold: float = 0.015,
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

    # Holosoma's virtual hand markers are fixed +0.1 m along each wrist-roll
    # link.  Use those labels but place them in VAIC's wrist-roll contact slots.
    body_contact = np.zeros((frame_count, len(VAIC_CONTACT_BODY_NAMES)), dtype=bool)
    for side, destination in (("left", 2), ("right", 5)):
        candidates = [f"{side}_hand_contact_link", f"{side}_wrist_roll_link"]
        source_name = next((name for name in candidates if name in source_contact_names), None)
        if source_name is None:
            raise ValueError(f"missing a Holosoma {side}-hand contact label")
        body_contact[:, destination] = source_contact[:, source_contact_names.index(source_name)]

    object_points = evenly_spaced_points(np.load(object_points_path, allow_pickle=False))
    motion = {
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        # Both Holosoma and VAIC store quaternions in WXYZ order.
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
        "joint_pos": raw["joint_pos"][:, 7:].astype(np.float32),
        "joint_vel": raw["joint_vel"][:, 6:].astype(np.float32),
        "object_contact": body_contact.any(axis=1, keepdims=True),
        "body_contact": body_contact,
        "feet_contact": _feet_contact(
            body_pos_w, normalized_body_names, foot_height_threshold
        ),
        "object_points": object_points[None],
    }
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    meta: dict[str, object] = {
        "body_names": normalized_body_names,
        "joint_names": joint_names,
        "fps": fps,
        "source": str(source.resolve()),
        "conversion": {
            "format": "holosoma_to_vaic_v1",
            "quaternion_order": "wxyz",
            "foot_height_threshold": foot_height_threshold,
            "object_points_source": str(object_points_path.resolve()),
            "object_points_selection": "linspace_128",
        },
    }
    return motion, meta


def write_motion(output_dir: Path, motion: dict[str, np.ndarray], meta: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "motion.npz", **motion)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
        stream.write("\n")


def _default_object_points(source: Path) -> Path:
    # .../train_r1/rl/<object>/<clip>.npz -> .../train_r1/objects/<object>/sample_points.npy
    if len(source.parents) < 3:
        raise ValueError(f"cannot infer object-points path from {source}")
    return source.parents[2] / "objects" / source.parent.name / "sample_points.npy"


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
    )
    write_motion(args.output_dir, motion, meta)
    print(
        f"Converted {source} -> {args.output_dir.resolve()} "
        f"({motion['joint_pos'].shape[0]} frames, {len(meta['joint_names'])} joints)"
    )


if __name__ == "__main__":
    main()

