"""Train VAIC perception and actor_adapt from privileged-Teacher rollouts.

The filename intentionally preserves the ``percetpion_actor.py`` spelling in
the experiment command.  The frozen Teacher is the sole rollout controller;
``actor_adapt`` is optimized only by supervised Teacher BC conditioned on the
EMA privileged prediction captured during that rollout.
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from active_adaptation.learning.ppo.perception_actor import (
    ACTOR_BC_PERCEPTION_SOURCE,
    ACTOR_OBJECTIVE_SEMANTICS,
    OPTIMIZED_MODULES,
    TeacherRolloutPerceptionActor,
)
from active_adaptation.learning.ppo.perception_only import (
    PERCEPTION_OBJECTIVE_SEMANTICS,
    ROLLOUT_SEMANTICS,
)

try:
    from .percetpion import (
        apply_perception_iteration_controls,
        validate_perception_training_config,
    )
    from .train import run_training
except ImportError:
    from percetpion import (
        apply_perception_iteration_controls,
        validate_perception_training_config,
    )
    from train import run_training


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
EXPECTED_ALGO_TARGET = (
    "active_adaptation.learning.ppo.perception_actor."
    "TeacherRolloutPerceptionActor"
)
EXPECTED_ALGO_NAME = "teacher_rollout_perception_actor"


def validate_perception_actor_training_config(cfg: DictConfig) -> dict:
    return validate_perception_training_config(
        cfg,
        expected_algo_target=EXPECTED_ALGO_TARGET,
        expected_algo_name=EXPECTED_ALGO_NAME,
        policy_cls=TeacherRolloutPerceptionActor,
        entrypoint_name="percetpion_actor.py",
    )


@hydra.main(config_path=CONFIG_PATH, config_name="percetpion_actor", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    apply_perception_iteration_controls(cfg)
    audit = validate_perception_actor_training_config(cfg)
    noise = float(cfg.algo.load_noise_scale)

    print(
        "Perception+actor Teacher-BC contract verified:\n"
        f"  Teacher checkpoint: {audit['path']}\n"
        f"  Teacher rollout: action = ref_joint_pos + Normal(residual, {noise:g})\n"
        "  actor_adapt influence on rollout: none\n"
        "  actor_adapt initialization: BC-trained weights from Teacher checkpoint\n"
        f"  perception initialization: {cfg.algo.perception_initialization}\n"
        f"  optimized modules: {', '.join(OPTIMIZED_MODULES)}\n"
        "  actor_adapt BC input: detached rollout-cached EMA priv_pred\n"
        "  actor_adapt online priv_pred input: forbidden\n"
        f"  actor BC perception source: {ACTOR_BC_PERCEPTION_SOURCE}\n"
        "  actor_adapt BC target: clean absolute Teacher mean\n"
        f"  actor objective: {ACTOR_OBJECTIVE_SEMANTICS}\n"
        f"  perception objective: {PERCEPTION_OBJECTIVE_SEMANTICS}\n"
        f"  rollout semantics: {ROLLOUT_SEMANTICS}\n"
        f"  rollout iterations: {int(cfg.iteration)}"
    )
    return run_training(cfg)


if __name__ == "__main__":
    main()
