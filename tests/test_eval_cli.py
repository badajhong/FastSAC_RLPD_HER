import pytest
from hydra import compose, initialize_config_dir

from scripts.eval_cli import normalize_eval_argv


@pytest.mark.parametrize(
    "overrides",
    [
        ["oracle_object_pose=true"],
        ["oracle_priv_pred=true"],
        ["oracle_object_pose=true", "oracle_priv_pred=true"],
        ["oracle_object_pose=false", "oracle_priv_pred=false"],
    ],
)
@pytest.mark.parametrize("has_defaults", [False, True])
def test_bare_oracle_overrides_compose_with_saved_and_current_configs(
    tmp_path, overrides, has_defaults
):
    config = "seed: 0\n"
    if has_defaults:
        config += "oracle_object_pose: false\noracle_priv_pred: false\n"
    (tmp_path / "cfg.yaml").write_text(config)
    argv = ["scripts/eval.py", *overrides, "seed=7"]

    with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
        cfg = compose(config_name="cfg", overrides=normalize_eval_argv(argv)[1:])

    assert cfg.seed == 7
    for override in overrides:
        key, value = override.split("=")
        assert cfg[key] is (value == "true")
    assert argv == ["scripts/eval.py", *overrides, "seed=7"]


def test_explicit_hydra_operations_and_unrelated_arguments_are_preserved():
    argv = [
        "scripts/eval.py",
        "--config-path=/saved/run/files/",
        "--config-name=cfg",
        "+oracle_object_pose=true",
        "++oracle_priv_pred=true",
        "~oracle_object_pose",
        "algo.oracle_object_pose=true",
        "oracle_object_pose_extra=true",
        "oracle_object_pose",
        "checkpoint_path=/saved/oracle_priv_pred=true/checkpoint.pt",
        "eval_render=false",
    ]

    assert normalize_eval_argv(argv) == argv


def test_normalization_is_idempotent_and_preserves_program_name():
    argv = ["oracle_object_pose=true", "oracle_priv_pred=true"]
    normalized = normalize_eval_argv(argv)

    assert normalized == ["oracle_object_pose=true", "++oracle_priv_pred=true"]
    assert normalize_eval_argv(normalized) == normalized


def test_empty_argument_list_is_preserved():
    assert normalize_eval_argv([]) == []
