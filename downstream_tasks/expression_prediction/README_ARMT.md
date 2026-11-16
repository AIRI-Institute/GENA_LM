# ARMT expression training

Run the commands below from the GENA_LM repository root. The examples in
`configs/` keep the 16,384-token dataset windows and process 1,024-token
segments around the TSS.

| Config | Segments | n_keys | Entry point suffix | Initialization used in the experiment |
| --- | ---: | ---: | --- | --- |
| `final_16384_armt_s1.yaml` | 1 | 8 | `run_expression_finetuning_armt_center` | `final_methyl` weights file |
| `final_16384_armt_s3_5n_airi.yaml` | 3 | 8 | `run_expression_finetuning_armt_center` | 1-segment checkpoint at step 10,250 |
| `final_16384_armt_s5_5n_airi.yaml` | 5 | 8 | `run_expression_finetuning_armt_center` | 3-segment checkpoint at step 49,500 |
| `final_16384_armt_s5_new_5n_airi.yaml` | 5 | 8 | `new` | 5-segment checkpoint at step 55,250 |
| `final_16384_armt_s5_memtpm_5n_airi.yaml` | 5 | 8 | `new` | 3-segment checkpoint at step 49,500 |
| `final_16384_armt_s9_5n_airi.yaml` | 9 | 4 | `run_expression_finetuning_armt_center` | 5-segment checkpoint at step 124,250 |

Set `GENALM_HOME` to the directory that contains `GENA_LM/`. The existing
dataset and interval paths expect the expression data to be available under
that checkout. The model paths default to `$GENALM_HOME/models/moderngena_large`
and `$GENALM_HOME/models/decoderdp0.1`. Set `ARMT_INIT_CHECKPOINT` to the
absolute path of a compatible checkpoint. The 1-segment config starts from
a model weights file and resets the optimizer, schedule and step counter.
The other configs expect a full checkpoint directory with trainer and
Accelerate state.

For example, on the same 40-GPU setup as the 9-segment experiment, run the
following on each node with its own `MACHINE_RANK` (0–4) and the same
`MAIN_PROCESS_IP` and `MAIN_PROCESS_PORT`:

```bash
export GENALM_HOME=/path/to/project-parent
export ARMT_INIT_CHECKPOINT=/path/to/stage5/model_124250
export ARMT_MIXED_PRECISION=bf16

accelerate launch --multi_gpu --num_machines 5 --num_processes 40 \
  --machine_rank "$MACHINE_RANK" --main_process_ip "$MAIN_PROCESS_IP" \
  --main_process_port "$MAIN_PROCESS_PORT" --mixed_precision bf16 \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_armt_center \
  --experiment_config downstream_tasks/expression_prediction/configs/final_16384_armt_s9_5n_airi.yaml \
  --batch_size 1 --gradient_accumulation_steps 24
```

For the two configs marked `new`, use
`--module downstream_tasks.expression_prediction.new`. Choose the process
count and gradient accumulation for your allocation. When restoring full
Accelerate state, use the checkpoint's distributed setup.

These examples use FlashAttention 2 in both ModernBERT blocks. To resume an
existing run within the same output directory, set `args_params.resume` to
the checkpoint path relative to `MODEL_PATH_ROOT`. This overrides
`init_checkpoint` and the reset flags.
