import importlib.util
from pathlib import Path

import numpy as np
import pytest


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


def _synthetic_default_pose(
    path: Path, joint_names: list[str], body_names: list[str]
) -> Path:
    reference_body_names = [name for name in body_names[1:] if name != "largebox_link"]
    body_pos = np.zeros((len(reference_body_names), 3), dtype=np.float32)
    body_pos[reference_body_names.index("pelvis_link"), 2] = 0.76
    body_quat = np.zeros((len(reference_body_names), 4), dtype=np.float32)
    body_quat[:, 0] = 1.0
    np.savez_compressed(
        path,
        body_names=np.asarray(reference_body_names),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        joint_names=np.asarray(joint_names),
        joint_pos=np.asarray([-1.0, 1.0], dtype=np.float32),
        root_body_name=np.asarray("pelvis_link"),
    )
    return path


def test_converts_root_layout_contacts_and_geometry(tmp_path):
    source = tmp_path / "clip.npz"
    joint_names, body_names = _synthetic_archive(source)
    point_path = tmp_path / "sample_points.npy"
    points = np.arange(340 * 3, dtype=np.float64).reshape(340, 3)
    np.save(point_path, points)

    motion, meta = CONVERTER.convert_motion(
        source,
        point_path,
        default_pose_prepend_duration_s=0.0,
        default_pose_append_duration_s=0.0,
    )

    assert motion["joint_pos"].shape == (4, 2)
    assert motion["joint_vel"].shape == (4, 2)
    np.testing.assert_array_equal(
        motion["joint_pos"], np.load(source)["joint_pos"][:, 7:]
    )
    np.testing.assert_array_equal(
        motion["joint_vel"], np.load(source)["joint_vel"][:, 6:]
    )
    assert motion["body_pos_w"].shape == (4, len(body_names) - 1, 3)
    assert meta["body_names"] == body_names[1:]
    assert meta["joint_names"] == joint_names
    assert meta["fps"] == 50.0

    assert motion["body_contact"].shape == (4, 8)
    np.testing.assert_array_equal(
        motion["body_contact"][:, 2], [False, True, False, True]
    )
    np.testing.assert_array_equal(
        motion["body_contact"][:, 5], [False, False, True, True]
    )
    np.testing.assert_array_equal(
        motion["object_contact"][:, 0], [False, True, True, True]
    )
    assert motion["feet_contact"].shape == (4, 2)
    np.testing.assert_array_equal(
        motion["feet_contact"][:, 0], [True, True, False, False]
    )
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

    motion, _ = CONVERTER.convert_motion(
        source,
        point_path,
        default_pose_prepend_duration_s=0.0,
        default_pose_append_duration_s=0.0,
    )

    np.testing.assert_allclose(motion["body_quat_w"][0, 0], [0.5, 0.1, 0.2, 0.3])
    assert motion["body_quat_w"].shape[1] == len(body_names) - 1


def test_preserves_direct_ankle_object_contacts(tmp_path):
    source = tmp_path / "clip.npz"
    _synthetic_archive(source)
    with np.load(source) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["contact_object_names"] = np.asarray(
        [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_hand_contact_link",
            "right_hand_contact_link",
        ]
    )
    payload["contact_object_label"] = np.asarray(
        [
            [True, False, False, False],
            [False, True, True, False],
            [False, False, False, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )
    np.savez(source, **payload)
    point_path = tmp_path / "sample_points.npy"
    np.save(point_path, np.zeros((340, 3)))

    motion, _ = CONVERTER.convert_motion(
        source,
        point_path,
        default_pose_prepend_duration_s=0.0,
        default_pose_append_duration_s=0.0,
    )

    np.testing.assert_array_equal(
        motion["body_contact"][:, 0], [True, False, False, False]
    )
    np.testing.assert_array_equal(
        motion["body_contact"][:, 1], [False, True, False, False]
    )
    np.testing.assert_array_equal(
        motion["body_contact"][:, 2], [False, True, False, True]
    )
    np.testing.assert_array_equal(
        motion["body_contact"][:, 5], [False, False, True, True]
    )
    assert motion["object_contact"].all()


def test_object_point_inference_rejects_stale_nearby_cloud(tmp_path):
    hoi_root = tmp_path / "HOI"
    source = hoi_root / "train_r1/rl/suitcase/clip.npz"
    source.parent.mkdir(parents=True)
    np.savez(source, contact_object_target_indices=np.asarray([0, 595]))

    nearby = hoi_root / "train_r1/objects/suitcase/sample_points.npy"
    nearby.parent.mkdir(parents=True)
    np.save(nearby, np.zeros((340, 3)))
    canonical = (
        hoi_root
        / "src/holosoma_retargeting/holosoma_retargeting/models/objects"
        / "suitcase/sample_points.npy"
    )
    canonical.parent.mkdir(parents=True)
    np.save(canonical, np.zeros((596, 3)))

    assert CONVERTER._default_object_points(source) == canonical


def test_object_point_inference_rejects_misordered_nearby_cloud(tmp_path):
    hoi_root = tmp_path / "HOI"
    source = hoi_root / "train_r1/rl/suitcase/clip.npz"
    source.parent.mkdir(parents=True)
    expected_points = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.savez(
        source,
        contact_object_target_indices=np.asarray([[[0, 2]]]),
        contact_object_target_points_obj=np.asarray(
            [[[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]]
        ),
        contact_object_target_valid=np.asarray([[True]]),
    )

    nearby = hoi_root / "train_r1/objects/suitcase/sample_points.npy"
    nearby.parent.mkdir(parents=True)
    np.save(nearby, expected_points[::-1])
    canonical = (
        hoi_root
        / "src/holosoma_retargeting/holosoma_retargeting/models/objects"
        / "suitcase/sample_points.npy"
    )
    canonical.parent.mkdir(parents=True)
    np.save(canonical, expected_points)

    assert CONVERTER._default_object_points(source) == canonical


def test_bakes_holosoma_default_pose_boundaries_without_duplicate_frames(tmp_path):
    source = tmp_path / "clip.npz"
    joint_names, body_names = _synthetic_archive(source)
    point_path = tmp_path / "sample_points.npy"
    np.save(point_path, np.zeros((340, 3)))
    reference_path = _synthetic_default_pose(
        tmp_path / "default_pose.npz", joint_names, body_names
    )

    original, _ = CONVERTER.convert_motion(
        source,
        point_path,
        default_pose_prepend_duration_s=0.0,
        default_pose_append_duration_s=0.0,
    )
    motion, meta = CONVERTER.convert_motion(
        source,
        point_path,
        default_pose_reference=reference_path,
        default_pose_prepend_duration_s=0.04,
        default_pose_append_duration_s=0.04,
    )

    # At 50 Hz, 0.04 seconds gives two frames on each side. The original
    # first/last frames appear once, at indices 2 and 5 respectively.
    assert motion["joint_pos"].shape == (8, 2)
    for key in (
        "body_pos_w",
        "body_lin_vel_w",
        "body_quat_w",
        "body_ang_vel_w",
        "joint_pos",
        "joint_vel",
        "body_contact",
        "object_contact",
    ):
        np.testing.assert_array_equal(motion[key][2:6], original[key])

    default_joint_pos = np.asarray([-1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(motion["joint_pos"][0], default_joint_pos)
    np.testing.assert_allclose(
        motion["joint_pos"][1], (default_joint_pos + original["joint_pos"][0]) / 2.0
    )
    np.testing.assert_allclose(
        motion["joint_pos"][6], (original["joint_pos"][-1] + default_joint_pos) / 2.0
    )
    np.testing.assert_allclose(motion["joint_pos"][7], default_joint_pos)
    np.testing.assert_array_equal(motion["joint_vel"][0], 0.0)
    np.testing.assert_array_equal(motion["joint_vel"][-1], 0.0)

    normalized_body_names = body_names[1:]
    root_index = normalized_body_names.index("pelvis_link")
    object_index = normalized_body_names.index("largebox_link")
    assert motion["body_pos_w"][0, root_index, 2] == pytest.approx(0.76)
    np.testing.assert_array_equal(
        motion["body_pos_w"][:2, object_index],
        np.repeat(original["body_pos_w"][:1, object_index], 2, axis=0),
    )
    np.testing.assert_array_equal(
        motion["body_pos_w"][-2:, object_index],
        np.repeat(original["body_pos_w"][-1:, object_index], 2, axis=0),
    )
    assert not motion["body_contact"][:2].any()
    assert not motion["body_contact"][-2:].any()
    assert not motion["object_contact"][:2].any()
    assert not motion["object_contact"][-2:].any()
    assert motion["feet_contact"][0].all()
    assert motion["feet_contact"][-1].all()

    transition = meta["conversion"]["default_pose_transition"]
    assert transition["prepend_frames"] == 2
    assert transition["append_frames"] == 2
    assert transition["real_motion_range"] == [2, 6]
    assert transition["output_frames"] == 8


def test_wxyz_slerp_uses_shortest_arc_and_stays_normalized():
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    yaw_180 = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    result = CONVERTER._slerp_wxyz(
        identity, yaw_180, np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(result[0], identity, atol=1.0e-6)
    np.testing.assert_allclose(
        result[1], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], atol=1.0e-6
    )
    np.testing.assert_allclose(result[2], yaw_180, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1.0, atol=1.0e-6)

    antipodal = CONVERTER._slerp_wxyz(
        identity, -identity, np.asarray([0.25, 0.75], dtype=np.float32)
    )
    np.testing.assert_allclose(antipodal, np.repeat(identity[None], 2, axis=0))


HOI_ROOT = Path(__file__).resolve().parents[2] / "HOI"
REAL_SOURCE = HOI_ROOT / "train_r1/rl/largebox/sub3_largebox_003.npz"
REAL_POINTS = HOI_ROOT / "train_r1/objects/largebox/sample_points.npy"


@pytest.mark.skipif(
    not (REAL_SOURCE.is_file() and REAL_POINTS.is_file()),
    reason="the neighboring HOI R1 dataset is unavailable",
)
def test_real_r1_clip_has_100_frame_transitions_and_preserves_real_slice():
    original, _ = CONVERTER.convert_motion(
        REAL_SOURCE,
        REAL_POINTS,
        default_pose_prepend_duration_s=0.0,
        default_pose_append_duration_s=0.0,
    )
    motion, meta = CONVERTER.convert_motion(REAL_SOURCE, REAL_POINTS)

    assert original["joint_pos"].shape[0] == 325
    assert motion["joint_pos"].shape[0] == 525
    for key in (
        "body_pos_w",
        "body_lin_vel_w",
        "body_quat_w",
        "body_ang_vel_w",
        "joint_pos",
        "joint_vel",
        "body_contact",
        "object_contact",
    ):
        np.testing.assert_array_equal(motion[key][100:425], original[key])

    root_index = meta["body_names"].index("pelvis_link")
    object_index = meta["body_names"].index("largebox_link")
    np.testing.assert_allclose(
        motion["body_pos_w"][0, root_index],
        [0.60275596, 0.73412222, 0.76],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        motion["body_pos_w"][-1, root_index],
        [0.52934426, -0.71597064, 0.76],
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        motion["body_pos_w"][:100, object_index],
        np.repeat(original["body_pos_w"][:1, object_index], 100, axis=0),
    )
    np.testing.assert_array_equal(
        motion["body_pos_w"][-100:, object_index],
        np.repeat(original["body_pos_w"][-1:, object_index], 100, axis=0),
    )
    assert not motion["body_contact"][:100].any()
    assert not motion["body_contact"][-100:].any()
    np.testing.assert_allclose(
        np.linalg.norm(motion["body_quat_w"], axis=-1), 1.0, atol=1.0e-5
    )

    transition = meta["conversion"]["default_pose_transition"]
    assert transition["prepend_frames"] == 100
    assert transition["append_frames"] == 100
    assert transition["real_motion_range"] == [100, 425]
    assert transition["output_frames"] == 525
