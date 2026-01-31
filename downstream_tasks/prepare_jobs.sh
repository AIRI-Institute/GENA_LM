#!/usr/bin/env bash
set -euo pipefail

jobsDir="jobs/gena/"

cd ../..

NP=1
mkdir -p "downstream_tasks/nucleotide_transformer_bench/${jobsDir}/"

# config_name="moderngena_base_bs128_lr5e-05_wd1e-04.yaml"
for config_name in downstream_tasks/nucleotide_transformer_bench/configs/gena-lm*
do
  config_name=$(basename $config_name)
  CONFIG="downstream_tasks/nucleotide_transformer_bench/configs/${config_name}"

  declare -A TASK_NUM_LABELS=(
    ["H3K4me2"]=2
    ["enhancers_types"]=3
    ["splice_sites_donors"]=2
  )

  for TASK in "${!TASK_NUM_LABELS[@]}"; do
    CLS_NUM="${TASK_NUM_LABELS[$TASK]}"

    TMP_CONFIG="downstream_tasks/nucleotide_transformer_bench/${jobsDir}/ntbench_${TASK}_${config_name}"


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
      PORT=$((29500 + $FOLD))
    cat > downstream_tasks/nucleotide_transformer_bench/${jobsDir}/ntbench_${TASK}_${config_name%.*}_${FOLD}.sh <<JOB
#!/bin/bash
#SBATCH --job-name=${TASK}_${config_name}_f${FOLD}
#SBATCH --nodes=1
#SBATCH --time=4:00:00 
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --reservation=rnd

date
echo "Starting ${TASK}_${config_name}_f${FOLD}"
cd $HOME/DNALM/GENA_LM/GENA_LM-nt_bench/
source $HOME/envs/mGenaNTbench/bin/activate
export HF_HOME=$HOME/.hf
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

GENALM_HOME="$(realpath .)" accelerate launch \
  --main_process_port $PORT \
  --num_processes $NP \
  --module downstream_tasks.nucleotide_transformer_bench.run_finetune \
  --experiment_config $TMP_CONFIG \
  --valid_fold $FOLD
date
echo "Done ${TASK}_${config_name}_f${FOLD}"
JOB
    done
  done
done
echo "done"
