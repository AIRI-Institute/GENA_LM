#!/bin/bash

intervals=/scratch/tsoies-Expression/intervals
configs=/scratch/tsoies-Expression/configs
utils=${configs}/utils
expression_prediction=/scratch/tsoies-Expression/GENA_LM/downstream_tasks/expression_prediction
finetune_scripts=/scratch/tsoies-Expression/finetune_scripts/

for split in ${intervals}/*/; do 
python ${utils}/make_config.py ${split} --outdir ${configs}
done

python ${utils}/add_header.py ${utils}/common_header.yaml ${configs}

python generate_bash_scripts.py \
    --template ${expression_prediction}/slurm_finetune_expression.sh \
    --yaml-dir ${configs} \
    --output-dir ${finetune_scripts}

#for file_mapping in /scratch/tsoies-Expression/intervals/*/*csv; do 
#python process_file_mappings.py \
#    ${file_mapping} \
#    --output /scratch/tsoies-Expression/GENA_LM/downstream_tasks/expression_prediction/datasets/data/$(basename ${file_mapping})
#done

