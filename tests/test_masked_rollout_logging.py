import ast
import time
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from scripts.helpers import (
    EpisodeStats,
    resolve_student_only_env_mask,
    validate_student_only_rollout_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_method(relative_path: str, class_name: str, method_name: str):
    tree = ast.parse((ROOT / relative_path).read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch, "time": time}
    exec(compile(module, relative_path, "exec"), namespace)
    return namespace[method_name]


def _rollout() -> TensorDict:
    rollout = TensorDict({}, batch_size=[2, 3])
    rollout.set(
        ("next", "done"),
        torch.tensor(
            [[[True], [False], [False]], [[False], [True], [True]]]
        ),
    )
    rollout.set(
        ("next", "stats", "success"),
        torch.tensor(
            [[[10.0], [20.0], [30.0]], [[1.0], [3.0], [4.0]]]
        ),
    )
    rollout.set(
        ("next", "stats", "task", "tracking"),
        torch.tensor(
            [[[10.0], [20.0], [30.0]], [[1.0], [3.0], [4.0]]]
        ),
    )
    rollout.set(
        "is_init",
        torch.tensor(
            [[[True], [False], [False]], [[True], [False], [True]]]
        ),
    )
    rollout.set(
        "is_dagger_env",
        torch.tensor([[True, True, True], [False, False, False]]),
    )
    return rollout


def test_episode_stats_include_only_completed_student_only_episodes():
    stats = EpisodeStats([("stats", "success")], device=torch.device("cpu"))

    stats.add(_rollout(), env_mask=torch.tensor([False, True]))

    assert len(stats) == 2
    result = stats.pop()
    assert result["stats", "success"].item() == pytest.approx(3.5)


def test_episode_stats_default_keeps_all_environment_behavior():
    stats = EpisodeStats([("stats", "success")], device=torch.device("cpu"))

    stats.add(_rollout())

    assert len(stats) == 3
    result = stats.pop()
    assert result["stats", "success"].item() == pytest.approx(17.0 / 3.0)


def test_reward_active_count_is_masked_without_changing_returned_reward():
    reward_call = _load_method(
        "active_adaptation/envs/mdp/base.py", "Reward", "__call__"
    )

    class Env:
        _stats_ema_env_mask = torch.tensor([False, True, True])

    class RewardStub:
        env = Env()
        weight = 2.0

        @staticmethod
        def compute():
            return (
                torch.tensor([[2.0], [5.0], [7.0]]),
                torch.tensor([[True], [False], [True]]),
            )

    reward, count = reward_call(RewardStub())
    assert torch.equal(reward, torch.tensor([[4.0], [0.0], [14.0]]))
    assert count == 1


def test_reward_group_masks_only_task_ema_sum_not_full_reward_or_timing():
    reward_group_compute = _load_method(
        "active_adaptation/envs/base.py", "RewardGroup", "compute"
    )

    class Env:
        _stats_ema_env_mask = torch.tensor([True, False, True])
        _stats_ema_decay = 0.5
        stats = {("task", "tracking"): torch.zeros(3, 1)}
        _stats_ema = {"task": {"tracking": (torch.tensor(2.0), torch.tensor(4.0))}}
        _perf_ema_reward = {
            "task": {"tracking": (torch.tensor(2.0), torch.tensor(4.0))}
        }

    class Group:
        env = Env()
        name = "task"

        class Reward:
            enabled = True

            @staticmethod
            def __call__():
                return torch.tensor([[1.0], [10.0], [100.0]]), 2

        funcs = {"tracking": Reward()}
        rew_buf = torch.zeros(3, 1)
        multiplicative = False

    reward = reward_group_compute(Group())

    assert torch.equal(reward, torch.tensor([[1.0], [10.0], [100.0]]))
    reward_sum, reward_count = Group.env._stats_ema["task"]["tracking"]
    assert reward_sum.item() == pytest.approx(102.0)
    assert reward_count.item() == pytest.approx(4.0)
    perf_sum, perf_count = Group.env._perf_ema_reward["task"]["tracking"]
    assert perf_sum.item() >= 1.0
    assert perf_count.item() == pytest.approx(3.0)


def test_student_only_mask_hook_and_rollout_provenance_must_agree():
    class Policy:
        def student_only_env_mask(self, *, num_envs, device):
            assert num_envs == 2
            return torch.tensor([False, True], device=device)

    mask = resolve_student_only_env_mask(
        Policy(), num_envs=2, device=torch.device("cpu")
    )
    assert torch.equal(mask, torch.tensor([False, True]))
    validate_student_only_rollout_provenance(_rollout(), mask)

    mismatched = _rollout()
    mismatched["is_dagger_env"] = torch.zeros(2, 3, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="provenance disagrees"):
        validate_student_only_rollout_provenance(mismatched, mask)


def test_policy_without_student_only_mask_preserves_legacy_logging():
    assert (
        resolve_student_only_env_mask(
            object(), num_envs=2, device=torch.device("cpu")
        )
        is None
    )
