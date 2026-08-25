import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "convert_holosoma_motion.py"
SPEC = importlib.util.spec_from_file_location("convert_holosoma_motion", MODULE_PATH)
CONVERTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONVERTER)


def _synthetic_archive(path: Path, frame_count: int = 4) -> tuple[list[str], list[str]]:
    joint_names = ["joint_a", "joint_b"]
    body_names = [
        "world",
        "pelvis_link",
        "left_foot_front_outer_link",
        "left_foot_front_inner_link",
        "left_foot_rear_outer_link",
        "left_foot_rear_inner_link",
        "right_foot_front_outer_link",
        "right_foot_front_inner_link",
        "right_foot_rear_outer_link",
        "right_foot_rear_inner_link",
        "largebox_link",
    ]
    body_pos = np.zeros((frame_count, len(body_names), 3), dtype=np.float64)
    body_pos[:, 1:, 2] = 0.1
    for name in body_names:
        if "foot_" in name:
            body_pos[:, body_names.index(name), 2] = [0.0, 0.01, 0.02, 0.03]
    body_quat = np.zeros((frame_count, len(body_names), 4), dtype=np.float64)
    body_quat[..., 0] = 1.0
    contact_names = np.array(["left_hand_contact_link", "right_hand_contact_link"])
    contact = np.array(
        [[False, False], [True, False], [False, True], [True, True]], dtype=bool
    )
    np.savez(
        path,
        fps=np.array([50]),
        joint_pos=np.arange(frame_count * 9).reshape(frame_count, 9),
        joint_vel=np.arange(frame_count * 8).reshape(frame_count, 8),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        contact_object_label=contact,
        contact_object_names=contact_names,
    )
    return joint_names, body_names


def test_converts_root_layout_contacts_and_geometry(tmp_path):
    source = tmp_path / "clip.npz"
    joint_names, body_names = _synthetic_archive(source)
    point_path = tmp_path / "sample_points.npy"
    points = np.arange(340 * 3, dtype=np.float64).reshape(340, 3)
    np.save(point_path, points)

    motion, meta = CONVERTER.convert_motion(source, point_path)

    assert motion["joint_pos"].shape == (4, 2)
    assert motion["joint_vel"].shape == (4, 2)
    np.testing.assert_array_equal(motion["joint_pos"], np.load(source)["joint_pos"][:, 7:])
    np.testing.assert_array_equal(motion["joint_vel"], np.load(source)["joint_vel"][:, 6:])
    assert motion["body_pos_w"].shape == (4, len(body_names) - 1, 3)
    assert meta["body_names"] == body_names[1:]
    assert meta["joint_names"] == joint_names
    assert meta["fps"] == 50.0

    assert motion["body_contact"].shape == (4, 8)
    np.testing.assert_array_equal(motion["body_contact"][:, 2], [False, True, False, True])
    np.testing.assert_array_equal(motion["body_contact"][:, 5], [False, False, True, True])
    np.testing.assert_array_equal(motion["object_contact"][:, 0], [False, True, True, True])
    assert motion["feet_contact"].shape == (4, 2)
    np.testing.assert_array_equal(motion["feet_contact"][:, 0], [True, True, False, False])
    assert motion["object_points"].shape == (1, 128, 3)
    np.testing.assert_array_equal(
        motion["object_points"][0],
        points[np.linspace(0, 339, 128, dtype=np.int64)].astype(np.float32),
    )


def test_preserves_wxyz_quaternions(tmp_path):
    source = tmp_path / "clip.npz"
    _, body_names = _synthetic_archive(source)
    with np.load(source) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["body_quat_w"][0, 1] = [0.5, 0.1, 0.2, 0.3]
    np.savez(source, **payload)
    point_path = tmp_path / "sample_points.npy"
    np.save(point_path, np.zeros((340, 3)))

    motion, _ = CONVERTER.convert_motion(source, point_path)

    np.testing.assert_allclose(motion["body_quat_w"][0, 0], [0.5, 0.1, 0.2, 0.3])
    assert motion["body_quat_w"].shape[1] == len(body_names) - 1

