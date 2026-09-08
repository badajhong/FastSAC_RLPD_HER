"""Versioned perception-pipeline contracts shared by training and evaluation.

The production contracts are ``legacy_v1`` and ``corrected_v1``.  The two
additional contracts exist only to isolate the depth-normalization and
depth-initialization changes in the planned 2x2 diagnostic.
"""

from __future__ import annotations

from typing import Final


LEGACY_PERCEPTION_PIPELINE_CONTRACT: Final = "legacy_v1"
DEPTH_NO_VECNORM_ONLY_CONTRACT: Final = "depth_no_vecnorm_only_v1"
DEPTH_INIT_ONLY_CONTRACT: Final = "depth_init_only_v1"
CORRECTED_PERCEPTION_PIPELINE_CONTRACT: Final = "corrected_v1"

PERCEPTION_PIPELINE_CONTRACTS: Final = frozenset(
    {
        LEGACY_PERCEPTION_PIPELINE_CONTRACT,
        DEPTH_NO_VECNORM_ONLY_CONTRACT,
        DEPTH_INIT_ONLY_CONTRACT,
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
    }
)

DEPTH_IDENTITY_CONTRACTS: Final = frozenset(
    {
        DEPTH_NO_VECNORM_ONLY_CONTRACT,
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
    }
)

DEPTH_CORRECT_INIT_CONTRACTS: Final = frozenset(
    {
        DEPTH_INIT_ONLY_CONTRACT,
        CORRECTED_PERCEPTION_PIPELINE_CONTRACT,
    }
)


def validate_perception_pipeline_contract(value: object) -> str:
    """Return a supported contract string or raise instead of guessing."""

    if not isinstance(value, str) or value not in PERCEPTION_PIPELINE_CONTRACTS:
        supported = ", ".join(sorted(PERCEPTION_PIPELINE_CONTRACTS))
        raise ValueError(
            "unsupported perception_pipeline_contract "
            f"{value!r}; expected one of: {supported}"
        )
    return value


def contract_uses_identity_depth(value: object) -> bool:
    """Whether raw unit-interval depth bypasses VecNorm."""

    return validate_perception_pipeline_contract(value) in DEPTH_IDENTITY_CONTRACTS


def contract_uses_corrected_depth_init(value: object) -> bool:
    """Whether only the depth CNN receives the corrected initialization."""

    return (
        validate_perception_pipeline_contract(value)
        in DEPTH_CORRECT_INIT_CONTRACTS
    )


__all__ = [
    "CORRECTED_PERCEPTION_PIPELINE_CONTRACT",
    "DEPTH_CORRECT_INIT_CONTRACTS",
    "DEPTH_IDENTITY_CONTRACTS",
    "DEPTH_INIT_ONLY_CONTRACT",
    "DEPTH_NO_VECNORM_ONLY_CONTRACT",
    "LEGACY_PERCEPTION_PIPELINE_CONTRACT",
    "PERCEPTION_PIPELINE_CONTRACTS",
    "contract_uses_corrected_depth_init",
    "contract_uses_identity_depth",
    "validate_perception_pipeline_contract",
]
