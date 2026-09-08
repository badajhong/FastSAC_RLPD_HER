#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_eval_oracles.sh [options] [-- HYDRA_OVERRIDE ...]

  --mode all|baseline|object_pose|priv_pred|both  (default: all)
  --model all|student|tvkd                      (default: all)
  --dry-run                                   Print commands without running
  -h, --help                                  Show this help

Environment: CUDA_VISIBLE_DEVICES=0, SEED=0, PYTHON_BIN=python,
             LOG_DIR=<repo>/logs/eval_oracles/<timestamp>

The default runs all four modes on both checkpoints sequentially.
Additional Hydra overrides follow --, for example: -- task.num_envs=64
Activate the vaic environment or set PYTHON_BIN to its Python interpreter.
EOF
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

mode=all
model=all
dry_run=false
extra_overrides=()
while (($#)); do
  case "$1" in
    --mode)
      (($# >= 2)) || fail '--mode requires a value'
      mode=$2
      shift 2
      ;;
    --model)
      (($# >= 2)) || fail '--model requires a value'
      model=$2
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_overrides=("$@")
      break
      ;;
    *) fail "Unknown option: $1 (put Hydra overrides after --)" ;;
  esac
done

case "$mode" in
  all) modes=(baseline object_pose priv_pred both) ;;
  baseline|object_pose|priv_pred|both) modes=("$mode") ;;
  *) fail "Invalid mode: $mode" ;;
esac
case "$model" in
  all) models=(student tvkd) ;;
  student|tvkd) models=("$model") ;;
  *) fail "Invalid model: $model" ;;
esac

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"
python_bin=${PYTHON_BIN:-python}
gpu=${CUDA_VISIBLE_DEVICES-0}
seed=${SEED:-0}
log_dir=${LOG_DIR:-"$repo_root/logs/eval_oracles/$(date +%Y%m%d_%H%M%S)_$$"}

checkpoint_for() {
  case "$1" in
    student)
      printf '%s\n' "$repo_root/students/vaic_skateboard_stu/wandb/latest-run/files/checkpoint_final.pt"
      ;;
    tvkd)
      printf '%s\n' "$repo_root/outputs/2026-09-07/23-57-17-G1SkateboardGeneralTracking-tvkd_fastsac_bc_dagger/wandb/latest-run/files/checkpoint_10200.pt"
      ;;
  esac
}

[[ -f scripts/eval.py ]] || fail "Missing evaluation script: $repo_root/scripts/eval.py"
for selected_model in "${models[@]}"; do
  checkpoint=$(checkpoint_for "$selected_model")
  [[ -f "$checkpoint" ]] || fail "Missing checkpoint: $checkpoint"
  [[ -f "${checkpoint%/*}/cfg.yaml" ]] || fail "Missing config: ${checkpoint%/*}/cfg.yaml"
done
if [[ "$dry_run" == false ]]; then
  command -v "$python_bin" >/dev/null || fail "Python interpreter not found: $python_bin"
  mkdir -p -- "$log_dir"
  printf 'Evaluation logs: %s\n' "$log_dir"
fi

for selected_model in "${models[@]}"; do
  checkpoint=$(checkpoint_for "$selected_model")
  for selected_mode in "${modes[@]}"; do
    object_pose=false
    priv_pred=false
    case "$selected_mode" in
      object_pose) object_pose=true ;;
      priv_pred) priv_pred=true ;;
      both) object_pose=true; priv_pred=true ;;
    esac
    command_args=(
      env "CUDA_VISIBLE_DEVICES=$gpu" PYTHONUNBUFFERED=1
      "$python_bin" scripts/eval.py
      "--config-path=${checkpoint%/*}/" --config-name=cfg
      "checkpoint_path=$checkpoint" "seed=$seed" eval_render=false
      "oracle_object_pose=$object_pose" "oracle_priv_pred=$priv_pred"
      "${extra_overrides[@]}"
    )
    printf '\nRunning %s / %s\n' "$selected_model" "$selected_mode"
    printf '%q ' "${command_args[@]}"
    printf '\n'
    if [[ "$dry_run" == false ]]; then
      "${command_args[@]}" 2>&1 | tee -- "$log_dir/${selected_model}_${selected_mode}.log"
    fi
  done
done
