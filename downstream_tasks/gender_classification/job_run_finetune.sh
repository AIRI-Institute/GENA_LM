#!/bin/bash
#SBATCH --job-name=gend0.25
#SBATCH --nodes=1
#SBATCH --time=480:00:00 
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=16
#SBATCH --reservation=rnd

date
cd $HOME/DNALM/GENA_LM/GENA_LM-task-gender_classification/downstream_tasks/gender_classification/
source $HOME/envs/gender/bin/activate

export HF_HOME=$HOME/.hf
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=0

echo "HF_HOME: $HF_HOME"
bash run_finetuning.sh

date
echo "Done!"