#!/usr/bin/env bash
set -euo pipefail

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=6

NP=1
config_name="gena-lm-bert-base-t2t_bs64_lr3e-05_wd1e-04"
CONFIG="downstream_tasks/nucleotide_transformer_bench/configs/${config_name}.yaml"

declare -A TASK_NUM_LABELS=(
  ["H3K4me2"]=2
  ["enhancers_types"]=3
  ["splice_sites_donors"]=2
)

for TASK in "${!TASK_NUM_LABELS[@]}"; do
  CLS_NUM="${TASK_NUM_LABELS[$TASK]}"

  TMP_CONFIG="$(mktemp /tmp/ntbench_${TASK}_XXXX.yaml)"

  python - <<'PY' "$CONFIG" "$TMP_CONFIG" "$TASK" "$CLS_NUM"
import sys
from omegaconf import OmegaConf

src, dst, task, cls_num = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = OmegaConf.load(src)

if "args_params" in cfg and cfg.args_params is not None:
    cfg.args_params.task_name = task
    cfg.args_params.cls_num = cls_num
else:
    cfg["task_name"] = task
    cfg["cls_num"] = cls_num

OmegaConf.save(cfg, dst)
print(dst)
PY

  for FOLD in 0 1 2 3 4 5 6 7 8 9; do
    PORT=$((29500 + FOLD))

    GENALM_HOME="$(realpath ..)" accelerate launch \
      --main_process_port "$PORT" \
      --num_processes "$NP" \
      --module downstream_tasks.nucleotide_transformer_bench.run_finetune \
      --experiment_config "$TMP_CONFIG" \
      --valid_fold "$FOLD"
  done

  rm -f "$TMP_CONFIG"
done

echo "done"
