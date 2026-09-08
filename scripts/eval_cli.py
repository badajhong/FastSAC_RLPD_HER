"""Command-line compatibility for evaluation with archived training configs."""

from collections.abc import Sequence


def normalize_eval_argv(argv: Sequence[str]) -> list[str]:
    """Allow bare oracle overrides even when the saved config lacks these keys.

    Hydra's ``++`` syntax both adds missing keys and overrides existing ones.
    Only the two new, bare options are rewritten; explicit Hydra operations and
    every other argument keep their usual behavior. The input includes argv[0].
    """
    normalized = list(argv)
    for index, argument in enumerate(normalized[1:], start=1):
        key, separator, _ = argument.partition("=")
        if separator and key in {"oracle_object_pose", "oracle_priv_pred"}:
            normalized[index] = "++" + argument
    return normalized
