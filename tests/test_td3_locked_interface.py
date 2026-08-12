"""Isaac-free regression locks for the skateboard TD3/DAgger interface.

These tests intentionally inspect configuration and normalized Python ASTs
instead of constructing the simulator.  They protect the environment-facing
contract while keeping the focused TD3 test suite fast enough to run on CPU.
Comments and source formatting do not affect any fingerprint.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_TASK_PATH = Path("cfg/task/base/hdmi-base.yaml")
SKATEBOARD_TASK_PATH = Path("cfg/task/G1/vaic/skateboard_stu.yaml")
TD3_CONFIG_PATH = Path("cfg/TD3_bc_dagger.yaml")

EXPECTED_ENVIRONMENT_FINGERPRINT = (
    "08b5c7764e9ad98a1eae05fe5c5cf16b11606d3dbed1e4f57b7eae84790d0053"
)
EXPECTED_REWARD_FINGERPRINT = (
    "51c80ab22a4d6f71f2f5734dff2af6f7d7a8c19244248c0066c63aeac57cac74"
)
EXPECTED_BETA_REPLAY_FINGERPRINT = (
    "bbd857be8d596d0e9787c7c3c4f3249822839079a072769e65c9539d3783d027"
)

EXPECTED_ACTION_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
]
EXPECTED_ACTION_SCALES = [
    0.55,
    0.55,
    0.55,
    0.35,
    0.35,
    0.44,
    0.55,
    0.55,
    0.44,
    0.35,
    0.35,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
]

# Each entry is (effective term name, configured weight, enabled).  The full
# reward fingerprint below also locks bodies, joints, sigmas, thresholds, and
# every other configured parameter.
EXPECTED_REWARD_SIGNATURE = {
    "tracking": [
        ("tracking_upper_body_pos", 0.5, True),
        ("tracking_upper_body_ori", 0.5, True),
        ("tracking_lower_body_pos", 0.5, True),
        ("tracking_lower_body_ori", 0.5, True),
        ("tracking_root_pos", 0.5, True),
        ("tracking_root_ori", 0.5, True),
        ("tracking_body_linvel", 0.5, True),
        ("tracking_body_angvel", 0.5, True),
        ("joint_pos_tracking_product", 0.5, True),
        ("joint_vel_tracking_product", 0.5, True),
    ],
    "object_tracking": [
        ("object_pos_tracking", 1.0, True),
        ("object_ori_tracking", 1.0, True),
        ("object_vel_tracking", 1.0, False),
        ("eef_contact_exp", 1.0, True),
    ],
    "loco": [
        ("action_rate_l2", 0.1, True),
        ("joint_vel_l2", 5.0e-4, True),
        ("joint_pos_limits", 10.0, True),
        ("joint_torque_limits", 0.01, True),
        ("survival", 1.0, True),
    ],
    "feet": [
        ("impact_force_l2", 1.0, True),
        ("feet_slip", 0.5, True),
        ("feet_air_time", 1.0, False),
        ("feet_air_lift", 1.0, False),
        ("feet_contact", 1.0, False),
        ("survival", 1.0, True),
        ("feet_air_time_skateboard", 5.0, True),
    ],
    "debug": [
        ("feet_air_time", 5.0, False),
        ("feet_contact_count", 1.0, False),
        ("joint_pos_limits", 1.0, False),
        ("eef_contact_all", 1.0, False),
        ("root_pos_error", 1.0, False),
        ("root_ori_error", 1.0, False),
        ("body_pos_error_local", 1.0, False),
        ("body_ori_error_local", 1.0, False),
        ("joint_pos_error", 1.0, False),
    ],
}


# AST fingerprints deliberately cover only locked baseline symbols.  New TD3
# files can evolve without requiring these hashes to be regenerated.
EXPECTED_AST_FINGERPRINTS = {
    (
        "active_adaptation/envs/base.py",
        "_Env._compute_reward",
    ): "9742b4dd36489a87be398c259c8e5474fc13d4a4458952f68c3e947af171731a",
    (
        "active_adaptation/envs/base.py",
        "_Env._step",
    ): "f00e1d5cc4177c2c5f9c5bd9cd76db37bbd08f30c07e1f011077788261c908fc",
    (
        "active_adaptation/envs/base.py",
        "RewardGroup.compute",
    ): "db329af2b1a2a976ba4548fe9fa3cd5fff59e44b59495b4d85858bc777d353b3",
    (
        "active_adaptation/envs/mdp/base.py",
        "Reward.__call__",
    ): "7d04711aaf288f5fdd74ed69977e4db4987d12d18f78ffcd3de7e94e83acffa8",
    (
        "active_adaptation/envs/mdp/action.py",
        "JointPosition.__call__",
    ): "95094500fb11cf6e1b2dfe238708cf2b09d5f73b48fe9c8cae58dfc191b37dd1",
    (
        "active_adaptation/learning/ppo/ppo_vel.py",
        "PPOVEL.__init__.build_actor",
    ): "94b524744c4c4a37a27e48c653f485fb1186d8753b27b5b721d826fbb25da1ea",
    (
        "active_adaptation/learning/ppo/fastsac_vel.py",
        "DistributionalQNetwork",
    ): "01bfbc4bdaa372bb2a666449aa55329f860b23d8e1fa69767aad314724d33a6a",
    (
        "active_adaptation/learning/ppo/fastsac_vel.py",
        "TwinDistributionalQ",
    ): "9b80adf422ff8270e1fce09098dee672f88a5762d53397816c83876ab8f593ed",
    (
        "active_adaptation/learning/ppo/fastsac_vel.py",
        "_build_isolated_q_network",
    ): "10766437204559825001307038f9141f3a0ee076469e2a2b410cbca1d692265c",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "_linear_teacher_probability",
    ): "e8e83964ce0524582b5aa849ec4d7995751f021214808af2f0431ed093df5775",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "_DaggerRolloutPolicy.forward",
    ): "a5926aefc516c78ad94c8f335f11ab20913825c0edf3f42c44ae1e1c21582c55",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._project_execution_action",
    ): "06237b1025a8f7c4ec98353931e14a3021fe72e8da9259711260767a41a6331b",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._student_action_from_latent",
    ): "8387adbffe27af152c174d59760023feefce181982e3472294f8d8eb4fdca4d6",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._teacher_action",
    ): "c7833a0af39a86a627796baf867dc0c21d5d11f7ae6519640fca053dac2ee35b",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._student_latent",
    ): "88923a28ea515ec22faaa2c01f11e3380fa74e1d7fa5b011b24a52e456fad8b1",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._scalarize_q_reward",
    ): "8784bf2e140d7c8f10631520ea9fdea6f4a72b8fac1b2bd1f3bd43a247e57493",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._dagger_transition_chunks",
    ): "ba94526bb524dd6977a2d5de7ca4ad606f0967013f31cd8576623101571accfb",
    (
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._q_action_input",
    ): "2bd73f1848c25d3756c597ae4d55d8a50c566d2dc8d9af5c571d6b05d896637e",
    (
        "active_adaptation/learning/ppo/fastsac_vel.py",
        "_fastsac_latent_to_action",
    ): "170e9789e7636fedfcbca31d1dfff87ae60d5b9d5b85cbf052b2d86b22257b1a",
    (
        "active_adaptation/learning/ppo/fastsac_vel.py",
        "_project_to_execution_support",
    ): "ba2e01cdb2b188da20594958e950fb38a65ddffa63db492aafaf63b84b716c06",
    (
        "active_adaptation/learning/ppo/fastsac_vel.py",
        "_sac_bootstrap_mask",
    ): "00b45887232c4ecdf1f05eb124d66e7bcec77147bc9bd09b3fb3b90aa58d281a",
    (
        "active_adaptation/learning/ppo/fastsac_vel.py",
        "_vaic_truncation_mask",
    ): "39cc56483d290f20946958f2f88a1db915167098dd31f9fb8d1dcd9fb3da89a5",
    (
        "active_adaptation/envs/mdp/rewards/common.py",
        "survival",
    ): "bbd14ad0ac2a40126789d1f9923039af053d8e291192f3b99f5ab805ca299937",
    (
        "active_adaptation/envs/mdp/rewards/common.py",
        "joint_pos_limits",
    ): "eb7bbbe96ff6226281da7e14d1e2ae216efc044f08bd8b0b96d8b41d13036a89",
    (
        "active_adaptation/envs/mdp/rewards/common.py",
        "joint_torque_limits",
    ): "9b01e1b0690aea9e4616245d20c38c93981349a35e47832106785ad41565e417",
    (
        "active_adaptation/envs/mdp/rewards/common.py",
        "action_rate_l2",
    ): "526e03a6ea24b219da6a9f5af3fb675b414b74df1bcaf595d9efeb86677271ad",
    (
        "active_adaptation/envs/mdp/rewards/common.py",
        "joint_vel_l2",
    ): "5377fb901465ab76d5278f83f0f7054bdc025d95643ce1be878454b415881d93",
    (
        "active_adaptation/envs/mdp/rewards/feet.py",
        "impact_force_l2",
    ): "f032b34728a08c0e204d3076fcf8bc2c5b191f86681543a327735bf855e0b90c",
    (
        "active_adaptation/envs/mdp/rewards/feet.py",
        "feet_air_time_skateboard",
    ): "43d6181b1d3aee361eec89790d5eae011b13f49c36961f8866450eed0bc5b760",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "keypoint_pos_tracking_product",
    ): "46843d235585d960d300d759d7b3aa8e1287d9cfed461685beb0bde77df12265",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "keypoint_pos_tracking_local_product",
    ): "2c95c131bd8cfb4628ff791fe18d291c87cff419fdd46f80fea90adec3cd3c71",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "keypoint_ori_tracking_product",
    ): "1d722768b9d76cfa391925c500f771204e1265267f508b00589c2f8c2d2014e7",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "keypoint_ori_tracking_local_product",
    ): "e5ae6597626f4d6cf6d0ff3ad09c7b323915b546817f4b50cc77256c1bb58d87",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "keypoint_lin_vel_tracking_product",
    ): "316971996fec8f786e6d97e1e032ad78a11a5097754d6241ab2d1c8fee3ab9fc",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "keypoint_ang_vel_tracking_product",
    ): "24e20db9940db37427b7ba56461ddd30c8e823174628fe737e48b8f9ec00a39c",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "joint_pos_tracking_product",
    ): "e0191630a02f397699778136eeaba31a3365118dd5301b6ab04d2ff54f014180",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "joint_vel_tracking_product",
    ): "834ec76441493ce16fdf3c4261c3bbd86732f822ad4ef2ce0ea93321f90f6314",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "object_pos_tracking",
    ): "bff895e1e53759172bde197d70a492671e0b625c4d2599e27a6d01f84523945b",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "object_ori_tracking",
    ): "6a453335ca035cdaf959ee592fcd8d777c17952b6ecded63f813bff4828543c6",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "eef_contact_exp",
    ): "973730497fc9fc6f2b6487ea3d01b4421173b134b89f7bb9475eb6ad696de580",
    (
        "active_adaptation/envs/mdp/commands/hdmi/rewards.py",
        "feet_slip",
    ): "d81544ddf8d4be9caacf7b4e4fc6edb46d73796878a73ddbd29a3161cdc79c94",
    (
        "active_adaptation/envs/mdp/observations/common.py",
        "root_ang_vel_history",
    ): "b874034b0b8ca2b19a9134c38dc7cb0815f22fb27a80388e8d4b8ba435dda84c",
    (
        "active_adaptation/envs/mdp/observations/common.py",
        "projected_gravity_history",
    ): "4b6c59c8c8f7ad2b46433e1ba13f2d3c19948c9def5499904ade586d67a4f506",
    (
        "active_adaptation/envs/mdp/observations/common.py",
        "joint_pos_history",
    ): "d0963b4e0c64dc5bd3a8fd635f44bd69208c0986c54afffecb7cf7986ccd3adc",
    (
        "active_adaptation/envs/mdp/observations/common.py",
        "prev_actions",
    ): "c53c7e1c3efef7934d0ca36f55d2fcbbb9a8279dd51874563177b2c33cb7fc41",
    (
        "active_adaptation/envs/mdp/observations/common.py",
        "applied_action",
    ): "2db1c771e047c5d990c0ee2c13b848382b8fb2b77051f2ae6228b3b41c9ae739",
    (
        "active_adaptation/envs/mdp/observations/common.py",
        "applied_torque",
    ): "afcaba579aed8df0b38a782b9b1a045251bb7dc6cd717048f6222b7ec05413d6",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_joint_pos_future",
    ): "35ac5d113804f72d8fd2aaccd46bc8e3e5117ffe86aa15deb3aaa9c01ab15c0d",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_root_pos_future_b",
    ): "c48dfa98f93db27963c5eb5845067456f9036bdb2567d5bccace967df2384214",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_root_ori_future_b",
    ): "d96d6e840b04beafc8b28bbb4e5b609d030f76213b8bbd43b64321955d38e5a4",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_body_pos_future_local",
    ): "312ba2243a373716890f61a248e226a5441a8326b5e28cf76fdbf61b710a6ba3",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "diff_body_pos_future_local",
    ): "85c8dbae5f6ee1df831c49dcc1c0895f5f0c844e80256d1956d85b53bd36345c",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "diff_body_lin_vel_future_local",
    ): "3b96105488670a6fc2fed1981a6f8f74ed91b0d1870a54a6c4f97820f1894152",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "diff_body_ori_future_local",
    ): "2a809582882ca83dde8edb406ebaa01b6dd74e0b0b6851f5b5c152b828448ddf",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "diff_body_ang_vel_future_local",
    ): "62b1d15a60f91855c90b8d26e0f7cff54ede6a556f3b1182b0b7e97ee8dc7e24",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_motion_phase",
    ): "9f23bfa71bbfb376ee3db7e0a18bbfbae8aa5e4c092757f2eda3c498db72cae3",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_root_vel_future_b",
    ): "e83494965dedfca020f64f36cc9149e7fbbe7599ceb446e552c474c447c9ca38",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_contact_flag_future",
    ): "2c9ee74e4878e0c6a0957d7c448861513b54c99cb003edfb4bf0561522671b7f",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "ref_contact_pos_b",
    ): "cefa691cc209452400c3fe8654e29f960a6b4ced79a1cadfd688780d30455e56",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "diff_contact_pos_b",
    ): "3dfd66c5c044f72eac81a56d756161af70578232dae62bf4ae15f20dabdce1cd",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "object_xy_b",
    ): "16d801eab4816c9762b84ffcc1500157556e6442c137af1730e32071db58ea95",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "object_heading_b",
    ): "6c5bb33a65d68002e19fd7f17fc712a6b914d1372d6d1a1ab9c2fcd4fb3f34df",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "object_pos_b",
    ): "7761f875ff57c076777d9d2e32c6b77b563cda0f614dfb68925bb65d68e4a7f6",
    (
        "active_adaptation/envs/mdp/commands/hdmi/observations.py",
        "object_ori_b",
    ): "d7a4d7ae16f221c3d0ae90ff964255d1969fbda9e1234ceb504d2c3523f344b4",
}


def _load_yaml(path: Path):
    with (ROOT / path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _deep_merge(base, override):
    """Minimal OmegaConf-style recursive merge for plain task mappings."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _effective_task():
    return _deep_merge(_load_yaml(BASE_TASK_PATH), _load_yaml(SKATEBOARD_TASK_PATH))


def _fingerprint(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_names(patterns, candidates):
    if isinstance(patterns, str):
        patterns = [patterns]
    return [
        name
        for name in candidates
        if any(re.fullmatch(pattern, name) for pattern in patterns)
    ]


def _find_definition(tree: ast.AST, qualified_name: str):
    nodes = tree.body
    found = None
    for name in qualified_name.split("."):
        found = next(
            node
            for node in nodes
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
        nodes = found.body
    return found


class _StripDocstrings(ast.NodeTransformer):
    @staticmethod
    def _without_docstring(node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    def visit_ClassDef(self, node):  # noqa: N802 - ast visitor API
        self.generic_visit(node)
        return self._without_docstring(node)

    def visit_FunctionDef(self, node):  # noqa: N802 - ast visitor API
        self.generic_visit(node)
        return self._without_docstring(node)

    def visit_AsyncFunctionDef(self, node):  # noqa: N802 - ast visitor API
        self.generic_visit(node)
        return self._without_docstring(node)


def _definition_ast(path: str, qualified_name: str):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    return _find_definition(tree, qualified_name)


def _ast_fingerprint(path: str, qualified_name: str) -> str:
    node = copy.deepcopy(_definition_ast(path, qualified_name))
    node = _StripDocstrings().visit(node)
    normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _class_literal_default(path: str, class_name: str, field_name: str):
    class_node = _definition_ast(path, class_name)
    for node in class_node.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == field_name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {class_name}.{field_name} in {path}")


def _normalized_reward_name(name: str) -> str:
    return name.split("(", 1)[0]


def test_skateboard_task_and_reward_semantic_fingerprints_are_locked():
    effective = _effective_task()
    reward = effective.pop("reward")
    effective.pop("defaults", None)

    assert _fingerprint(effective) == EXPECTED_ENVIRONMENT_FINGERPRINT
    assert _fingerprint(reward) == EXPECTED_REWARD_FINGERPRINT


def test_skateboard_observation_dimensions_and_action_contract_are_locked():
    task = _effective_task()
    asset = json.loads((ROOT / "asset_meta.json").read_text(encoding="utf-8"))
    joint_names = asset["joint_names_isaac"]
    body_names = asset["body_names_isaac"]

    action_scales = task["action"]["action_scaling"]
    controlled = []
    for joint_name in joint_names:
        matches = [
            value
            for pattern, value in action_scales.items()
            if re.fullmatch(pattern, joint_name)
        ]
        assert len(matches) <= 1, f"overlapping action regexes for {joint_name}"
        if matches:
            controlled.append((joint_name, matches[0]))

    assert [name for name, _ in controlled] == EXPECTED_ACTION_JOINTS
    assert [scale for _, scale in controlled] == EXPECTED_ACTION_SCALES
    action_dim = len(controlled)
    assert action_dim == 23

    tracking_bodies = _selected_names(
        task["command"]["tracking_keypoint_names"], body_names
    )
    tracking_joints = _selected_names(
        task["command"]["tracking_joint_names"], joint_names
    )
    assert len(tracking_bodies) == 16
    assert tracking_joints == EXPECTED_ACTION_JOINTS

    future_steps = len(task["command"]["future_steps"])
    num_eefs = len(task["command"]["contact_eef_body_name"])
    with np.load(ROOT / task["command"]["data_path"] / "motion.npz") as motion:
        contact_channels = int(motion["object_contact"].shape[-1])
    assert future_steps == 5
    assert num_eefs == 2
    assert contact_channels == 1

    policy_cfg = task["observation"]["policy"]
    policy_widths = {
        "root_ang_vel_history": 3
        * len(policy_cfg["root_ang_vel_history"]["history_steps"]),
        "projected_gravity_history": 3
        * len(policy_cfg["projected_gravity_history"]["history_steps"]),
        "joint_pos_history": len(joint_names)
        * len(policy_cfg["joint_pos_history"]["history_steps"]),
        "prev_actions": action_dim * policy_cfg["prev_actions"]["steps"],
    }
    assert policy_widths == {
        "root_ang_vel_history": 3,
        "projected_gravity_history": 3,
        "joint_pos_history": 174,
        "prev_actions": 69,
    }
    policy_dim = sum(policy_widths.values())
    vel_command_dim = future_steps * (3 + contact_channels)
    latent_dim = _class_literal_default(
        "active_adaptation/learning/ppo/ppo_vel.py", "PPOConfig", "latent_dim"
    )
    assert (vel_command_dim, policy_dim, latent_dim) == (20, 249, 256)
    assert vel_command_dim + policy_dim + latent_dim == 525

    q_config_path = "active_adaptation/learning/ppo/ppo_bc_dagger.py"
    assert (
        _class_literal_default(
            q_config_path, "PPOBCDaggerFinetuneConfig", "q_hidden_dim"
        )
        == 768
    )
    assert (
        _class_literal_default(
            q_config_path, "PPOBCDaggerFinetuneConfig", "q_num_atoms"
        )
        == 501
    )
    assert (
        _class_literal_default(q_config_path, "PPOBCDaggerFinetuneConfig", "q_v_min")
        == -20.0
    )
    assert (
        _class_literal_default(q_config_path, "PPOBCDaggerFinetuneConfig", "q_v_max")
        == 20.0
    )
    assert (
        _class_literal_default(
            q_config_path, "PPOBCDaggerFinetuneConfig", "q_layer_norm"
        )
        is True
    )

    command_widths = {
        "ref_body_pos_future_local": future_steps * len(tracking_bodies) * 3,
        "ref_joint_pos_future": future_steps * len(tracking_joints),
        "ref_motion_phase": 1,
    }
    assert command_widths == {
        "ref_body_pos_future_local": 240,
        "ref_joint_pos_future": 115,
        "ref_motion_phase": 1,
    }
    command_dim = sum(command_widths.values())

    object_widths = {
        "object_xy_b": 2,
        "object_heading_b": 2,
        "ref_contact_pos_b": num_eefs * 3,
        "object_pos_b": 3,
        "object_ori_b": 9,
    }
    assert list(task["observation"]["object_"]) == list(object_widths)
    assert sum(object_widths.values()) == 22

    priv_cfg = task["observation"]["priv"]
    ankle_count = len(_selected_names(priv_cfg["body_pos_b"]["body_names"], body_names))
    height_count = len(
        _selected_names(priv_cfg["body_height"]["body_names"], body_names)
    )
    priv_widths = {
        "root_ang_vel_history": 3
        * len(priv_cfg["root_ang_vel_history"]["history_steps"]),
        "projected_gravity_history": 3
        * len(priv_cfg["projected_gravity_history"]["history_steps"]),
        "joint_pos_history": len(joint_names)
        * len(priv_cfg["joint_pos_history"]["history_steps"]),
        "ref_root_pos_future_b": future_steps * 3,
        "ref_root_ori_future_b": future_steps * 6,
        "diff_body_pos_future_local": future_steps * len(tracking_bodies) * 3,
        "diff_body_ori_future_local": future_steps * len(tracking_bodies) * 6,
        "diff_body_lin_vel_future_local": future_steps * len(tracking_bodies) * 3,
        "diff_body_ang_vel_future_local": future_steps * len(tracking_bodies) * 3,
        "root_linvel_b": 3,
        "body_pos_b": ankle_count * 3,
        "body_vel_b": ankle_count * 3,
        "body_height": height_count,
        "applied_action": action_dim,
        "applied_torque": len(joint_names),
        "object_pos_b": 3,
        "object_ori_b": 9,
        "diff_object_pos_future": future_steps * 3,
        "diff_object_ori_future": future_steps * 9,
        "ref_object_contact_future": future_steps * contact_channels,
        "diff_contact_pos_b": num_eefs * 3,
    }
    assert list(priv_cfg) == list(priv_widths)
    assert sum(priv_widths.values()) == 1714
    assert command_dim == 356
    assert sum(priv_widths.values()) + policy_dim + command_dim + 22 == 2341

    assert task["name"] == "G1Skateboard"
    assert task["num_envs"] == 512
    assert task["max_episode_length"] == 1000
    assert task["sim"] == {
        "step_dt": 0.02,
        "isaac_physics_dt": 0.005,
        "mujoco_physics_dt": 0.002,
    }
    assert task["sim"]["step_dt"] / task["sim"]["isaac_physics_dt"] == 4
    assert (task["camera_width"], task["camera_height"]) == (64, 36)
    assert task["action"]["min_delay"] == 2
    assert task["action"]["max_delay"] == 6
    assert task["action"]["alpha"] == [0.8, 1.0]
    assert task["randomization"]["random_joint_offset"][".*"] == [-0.01, 0.01]


def test_effective_reward_terms_weights_scaling_and_aggregation_are_locked():
    reward = _effective_task()["reward"]
    assert list(reward) == ["tracking", "object_tracking", "loco", "feet", "debug"]
    assert "_mult_dt_" not in reward  # environment default remains True

    actual_signature = {}
    for group_name, terms in reward.items():
        assert terms.get("_multiplicative", False) is False
        actual_signature[group_name] = [
            (
                _normalized_reward_name(term_name),
                float(params["weight"]),
                bool(params.get("enabled", True)),
            )
            for term_name, params in terms.items()
            if term_name != "_multiplicative"
        ]
    assert actual_signature == EXPECTED_REWARD_SIGNATURE

    feet = reward["feet"]
    assert feet["impact_force_l2"]["body_names"] == "left_ankle_roll_link"
    assert feet["feet_slip"]["body_names"] == "left_ankle_roll_link"
    assert feet["feet_air_time_skateboard"] == {
        "weight": 5.0,
        "enabled": True,
        "body_names": ".*ankle_roll_link",
        "thres": 0.2,
        "soft_discount": 1.0,
    }
    assert all(not params.get("enabled", True) for params in reward["debug"].values())

    object_contact = reward["object_tracking"]["eef_contact_exp"]
    assert object_contact == {
        "weight": 1.0,
        "enabled": True,
        "pos_sigma": 0.3,
        "frc_sigma": 40.0,
        "frc_thres": 10.0,
        "gain": 5.0,
    }

    scalarizer = _definition_ast(
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._scalarize_q_reward",
    )
    assert ast.unparse(scalarizer.body[-1].value) == "reward.sum(dim=-1)"


def test_beta_replay_action_identity_and_normalization_contract_are_locked():
    baseline = _load_yaml(Path("cfg/bc_dagger.yaml"))["algo"]
    locked_keys = [
        "dagger_control_mode",
        "dagger_safe_takeover_rms",
        "dagger_safe_release_rms",
        "dagger_safe_min_teacher_steps",
        "dagger_action_clip",
        "dagger_safe_zero_iteration",
        "dagger_beta_start",
        "dagger_beta_end",
        "dagger_beta_zero_iteration",
        "dagger_beta_decay_rollouts",
        "q_num_atoms",
        "q_action_fusion",
        "q_action_coordinates",
        "sac_q_normalize_actions",
        "sac_q_action_input_gain",
        "sac_clipped_double_q",
        "q_teacher_replay_ratio",
        "q_learning_starts_per_source",
    ]
    payload = {key: baseline[key] for key in locked_keys}
    assert _fingerprint(payload) == EXPECTED_BETA_REPLAY_FINGERPRINT

    constructor = _definition_ast(
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune.__init__",
    )
    assignments = {}
    for node in ast.walk(constructor):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assignments.setdefault(target.attr, node.value)
    assert ast.unparse(assignments["q_actor_keys"]) == (
        "[VEL_CMD_KEY, OBS_KEY, PRIV_PRED_KEY]"
    )
    assert ast.unparse(assignments["q_critic_keys"]) == (
        "[OBS_PRIV_KEY, OBS_KEY, command_key]"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func.value) == "self.q_critic_keys"
        and node.func.attr == "append"
        and ast.unparse(node.args[0]) == "OBJECT_KEY"
        for node in ast.walk(constructor)
    )

    transition_builder = _definition_ast(
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetune._dagger_transition_chunks",
    )
    transition_dict = None
    for node in ast.walk(transition_builder):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            (
                None
                if key is None
                else key.value
                if isinstance(key, ast.Constant)
                else ast.unparse(key)
            )
            for key in node.keys
        ]
        if "actions" in keys and "rewards" in keys and "dones" in keys:
            transition_dict = dict(zip(keys, node.values))
            break
    assert transition_dict is not None
    assert ast.unparse(transition_dict["actions"]) == (
        "current[ACTION_KEY].reshape(n, self.action_dim)"
    )
    assert ast.unparse(transition_dict["DAGGER_REPLAY_TEACHER_ACTIONS"]) == (
        "current[DAGGER_TEACHER_ACTION_KEY].reshape(n, self.action_dim)"
    )
    assert ast.unparse(transition_dict["rewards"]) == (
        "self._scalarize_q_reward(current[REWARD_KEY]).reshape(n)"
    )

    replay_raw_keys = _definition_ast(
        "active_adaptation/learning/ppo/ppo_bc_dagger.py",
        "PPOBCDaggerFinetuneConfig",
    )
    raw_default = next(
        node
        for node in replay_raw_keys.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "replay_raw_observation_keys"
    )
    assert ast.unparse(raw_default.value) == (
        "(VEL_CMD_KEY, OBS_KEY, OBS_PRIV_KEY, CMD_KEY)"
    )

    make_env = _definition_ast("scripts/helpers.py", "make_env_policy")
    obs_keys_assignment = next(
        node
        for node in ast.walk(make_env)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "obs_keys"
            for target in node.targets
        )
    )
    assert ast.unparse(obs_keys_assignment.value) == (
        "[key for key, spec in base_env.observation_spec.items(True, True) "
        "if not (spec.dtype == bool or key.endswith('_'))]"
    )
    append_calls = [
        node
        for node in ast.walk(make_env)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "transform"
        and node.func.attr == "append"
    ]
    raw_copy = next(
        node
        for node in append_calls
        if node.args and ast.unparse(node.args[0]).startswith("RenameTransform(")
    )
    vecnorm = next(
        node
        for node in append_calls
        if node.args and ast.unparse(node.args[0]) == "vecnorm"
    )
    assert raw_copy.lineno < vecnorm.lineno


def test_locked_runtime_source_ast_fingerprints_are_unchanged():
    mismatches = {}
    for (path, qualified_name), expected in EXPECTED_AST_FINGERPRINTS.items():
        actual = _ast_fingerprint(path, qualified_name)
        if actual != expected:
            mismatches[f"{path}:{qualified_name}"] = {
                "expected": expected,
                "actual": actual,
            }
    assert not mismatches, json.dumps(mismatches, indent=2, sort_keys=True)


def _default_group_names(defaults):
    groups = []
    for entry in defaults:
        if isinstance(entry, dict):
            values = entry.keys()
        elif isinstance(entry, str):
            values = [entry.split(":", 1)[0]]
        else:
            raise AssertionError(f"Unsupported Hydra defaults entry: {entry!r}")
        for value in values:
            normalized = str(value).removeprefix("override ").lstrip("/")
            groups.append(normalized)
    return groups


def _mapping_key_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key))
            yield path
            yield from _mapping_key_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _mapping_key_paths(child, (*prefix, str(index)))


def test_td3_config_cannot_override_the_locked_task_reward_or_environment():
    path = ROOT / TD3_CONFIG_PATH
    assert path.is_file(), f"missing required Phase-1 config: {TD3_CONFIG_PATH}"
    td3_cfg = _load_yaml(TD3_CONFIG_PATH)
    assert isinstance(td3_cfg, dict)
    assert isinstance(td3_cfg.get("defaults"), list)

    locked_top_level = {
        "task",
        "reward",
        "environment",
        "env",
        "sim",
        "action",
        "observation",
        "termination",
        "randomization",
        "camera_dr",
        "robot",
        "terrain",
        "enable_cameras",
        "camera_width",
        "camera_height",
        "max_episode_length",
        "num_envs",
    }
    assert locked_top_level.isdisjoint(td3_cfg), (
        "TD3 config overrides locked environment keys: "
        f"{sorted(locked_top_level.intersection(td3_cfg))}"
    )

    locked_nested_keys = locked_top_level | {
        "step_dt",
        "isaac_physics_dt",
        "mujoco_physics_dt",
        "decimation",
        "action_scaling",
    }
    nested_overrides = sorted(
        ".".join(path)
        for path in _mapping_key_paths(td3_cfg)
        if path[-1].removeprefix("override ").lstrip("/") in locked_nested_keys
    )
    assert not nested_overrides, (
        f"TD3 config contains nested protected environment keys: {nested_overrides}"
    )

    default_groups = set(_default_group_names(td3_cfg["defaults"]))
    forbidden_default_groups = {
        "task",
        "reward",
        "environment",
        "env",
        "sim",
        "action",
        "observation",
        "termination",
        "randomization",
    }
    assert default_groups.isdisjoint(forbidden_default_groups), (
        "TD3 config changes a protected Hydra config group: "
        f"{sorted(default_groups.intersection(forbidden_default_groups))}"
    )

    algo = td3_cfg.get("algo", {})
    assert isinstance(algo, dict)
    assert forbidden_default_groups.isdisjoint(algo), (
        "TD3 algo config nests a protected environment override: "
        f"{sorted(forbidden_default_groups.intersection(algo))}"
    )

    baseline = _load_yaml(Path("cfg/bc_dagger.yaml"))["algo"]
    immutable_if_redeclared = {
        "dagger_action_clip": 20.0,
        "dagger_beta_start": baseline["dagger_beta_start"],
        "dagger_beta_end": baseline["dagger_beta_end"],
        "dagger_beta_zero_iteration": baseline["dagger_beta_zero_iteration"],
        "dagger_beta_decay_rollouts": baseline["dagger_beta_decay_rollouts"],
        "dagger_safe_takeover_rms": baseline["dagger_safe_takeover_rms"],
        "dagger_safe_release_rms": baseline["dagger_safe_release_rms"],
        "dagger_safe_min_teacher_steps": baseline["dagger_safe_min_teacher_steps"],
        "q_num_atoms": 501,
        "q_v_min": -20.0,
        "q_v_max": 20.0,
        "q_action_fusion": "late",
        "q_action_coordinates": "absolute",
        "sac_q_normalize_actions": True,
        "sac_q_action_input_gain": 1.0,
        "q_normalize_actions": True,
        "q_action_input_gain": 1.0,
        "dagger_replay_raw_observations": True,
        "replay_raw_observation_keys": [
            "vel_command",
            "policy",
            "priv",
            "command",
        ],
        "q_teacher_replay_ratio": 0.5,
    }
    changed = {
        key: {"expected": expected, "actual": algo[key]}
        for key, expected in immutable_if_redeclared.items()
        if key in algo and algo[key] != expected
    }
    assert not changed, json.dumps(changed, indent=2, sort_keys=True)

    # A structural final guard: no section from the effective task is overlaid
    # by this algorithm-only file, so composing it cannot mutate the semantic
    # environment/reward fingerprint checked above.
    task_sections = set(_effective_task())
    task_sections.discard("defaults")
    assert task_sections.isdisjoint(td3_cfg)
