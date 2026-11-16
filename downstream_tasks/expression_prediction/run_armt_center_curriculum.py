"""Prepare new stage configs; launch training only with explicit --run.

Run as a module from the GENA_LM root. Stage lengths are required, not guessed.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


PREFIX = "downstream_tasks.expression_prediction"
DATASET_CLASSES = {
    "ExpressionDataset": "ARMTCenterExpressionDatasetLegacy",
    "ExpressionDatasetFixedDesc": "ARMTCenterExpressionDatasetFixedDesc",
    "ExpressionDatasetTSSWindow": "ARMTCenterExpressionDataset",
    "ExpressionDatasetMode2": "ARMTCenterExpressionDatasetMode2Legacy",
    "ExpressionDatasetMode2FixedDesc": "ARMTCenterExpressionDatasetMode2FixedDesc",
    "ExpressionDatasetMode2TSSWindow": "ARMTCenterExpressionDatasetMode2",
    "MethylationDataset": "ARMTCenterMethylationDatasetLegacy",
    "MethylationDatasetTSSWindow": "ARMTCenterMethylationDataset",
}


def make_stage_config(base, stage, iters, stages, home, initial_checkpoint=None,
                      segment_size=1024, max_segments=17, dna_model=None, decoder_model=None):
    if (not stages or any(k < 1 or k % 2 != 1 for k in stages) or
            any(a >= b for a, b in zip(stages, stages[1:])) or max(stages) > max_segments or
            max_segments % 2 != 1 or segment_size < 4 or segment_size % 2):
        raise ValueError("Stages must increase through positive odd values; segment_size must be even")
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    cfg.TASK_NAME = f"expression/armt_center_{stage}segments"
    cfg.HOME_PATH = str(home)
    args = cfg.args_params
    args.model_cls = f"{PREFIX}.expression_model_armt_center:ExpressionCountsARMT"
    args.init_checkpoint = str(initial_checkpoint) if initial_checkpoint else None
    args.resume = None
    args.iters = args.num_training_steps = int(iters)
    args.reset_lr = args.reset_optimizer = args.reset_iteration = True
    # The existing trainer's best-checkpoint reload expects an obsolete .pth path.
    # This separate runner exports an explicit final-stage weights file instead.
    args.save_best = False
    args.save_interval = min(int(args.get("save_interval", 2500)), int(iters))
    args.valid_interval = min(int(args.get("valid_interval", 500)), int(iters))
    args.log_interval = min(int(args.get("log_interval", 50)), int(iters))
    args.num_warmup_steps = min(int(args.get("num_warmup_steps", 1000)), max(0, int(iters) // 10))
    kwargs = cfg.model_kwargs
    kwargs.hf_model_name = str(dna_model or home / "moderngena-large")
    kwargs.hf_model_name_decoder = str(decoder_model or home / "models/decoderdp0.1")
    # Replace positional overrides with local ARMT settings, keeping losses and description choices.
    for key in ("max_position_embeddings", "max_position_embeddings_decoder"):
        kwargs.pop(key, None)
    kwargs.num_segments, kwargs.max_segments = stage, max_segments
    kwargs.segment_size, kwargs.num_mem_tokens, kwargs.d_mem = segment_size, 32, 64
    kwargs.attn_implementation, kwargs.memory_dtype = "flash_attention_2", "float32"
    kwargs.dna_attn_implementation = "flash_attention_2"
    if cfg.get("shared_dataset_params") is None:
        cfg.shared_dataset_params = {}
    cfg.shared_dataset_params.n_keys = int(cfg.shared_dataset_params.get("n_keys", 4))
    # Keep original lengths, offsets and explicit/default num_after values.
    # The model selects the curriculum subset from the SAME existing H5 window.
    for name, dataset in cfg.items():
        if not (str(name).startswith("train_dataset") or str(name).startswith("valid_dataset")):
            continue
        old_class = str(dataset.get("_target_", "")).split(".")[-1]
        if old_class not in DATASET_CLASSES:
            raise ValueError(f"Unsupported dataset class in {name}: {old_class}")
        dataset._target_ = f"{PREFIX}.expression_dataset_armt_center.{DATASET_CLASSES[old_class]}"
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--home", type=Path, default=Path("/mnt/20tb/aspeedok"))
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--stage-iters", type=int, nargs="+", required=True)
    parser.add_argument("--segment-size", type=int, default=1024)
    parser.add_argument("--max-segments", type=int, default=17)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--dna-model", type=Path)
    parser.add_argument("--decoder-model", type=Path)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--port", type=int, default=29617)
    parser.add_argument("--run", action="store_true", help="Actually run all stages; default only prepares new files")
    args = parser.parse_args()
    if len(args.stages) != len(args.stage_iters) or any(n < 1 for n in args.stage_iters):
        parser.error("Provide one positive --stage-iters value per stage")
    if args.init_checkpoint and not args.init_checkpoint.is_file():
        parser.error("--init-checkpoint must name a weights file, not an optimizer-state directory")
    if args.output_dir.exists():
        parser.error("Choose a NEW output directory; existing configs will not be overwritten")
    base = OmegaConf.load(args.base_config)
    configs = [make_stage_config(base, stage, iters, args.stages, args.home,
               args.init_checkpoint, args.segment_size, args.max_segments, args.dna_model, args.decoder_model)
               for stage, iters in zip(args.stages, args.stage_iters)]
    args.output_dir.mkdir(parents=True)
    checkpoint = args.init_checkpoint
    manifest = {"base_config": str(args.base_config.resolve()), "stages": [], "training_started": args.run}
    env = dict(os.environ, GENALM_HOME=str(args.home))
    import torch
    supports_bf16 = (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                     and torch.cuda.is_bf16_supported())
    precision = env.get("ARMT_MIXED_PRECISION", "bf16" if supports_bf16 else
                        ("fp16" if torch.cuda.is_available() else "no"))
    env["ARMT_MIXED_PRECISION"] = precision
    env.setdefault("HF_HOME", str(args.home / "hf_cache/hf_home"))
    trainer_source = args.home / "soft/t5-experiments"
    if trainer_source.is_dir():
        env["PYTHONPATH"] = str(trainer_source) + os.pathsep + env.get("PYTHONPATH", "")
    for stage, cfg in zip(args.stages, configs):
        if checkpoint is not None:
            cfg.args_params.init_checkpoint = str(checkpoint.resolve())
        if not args.run and stage != args.stages[0]:
            cfg.args_params.init_checkpoint = "${oc.env:ARMT_PREVIOUS_STAGE_CHECKPOINT}"
        stage_file = args.output_dir / f"stage_{stage}.yaml"
        OmegaConf.save(cfg, stage_file)
        command = [sys.executable, "-m", "accelerate.commands.launch",
                   "--num_processes", str(args.num_processes), "--mixed_precision", precision,
                   "--main_process_port", str(args.port), "--module",
                   f"{PREFIX}.run_expression_finetuning_armt_center",
                   "--experiment_config", str(stage_file.resolve()),
                   "--batch_size", str(args.batch_size),
                   "--gradient_accumulation_steps", str(args.gradient_accumulation_steps)]
        if args.num_processes > 1:
            command.insert(3, "--multi_gpu")
        print(" ".join(command), flush=True)
        entry = {"segments": stage, "config": str(stage_file.resolve()), "command": command,
                 "checkpoint_policy": "previous stage final weights; new optimizer/scheduler"}
        if args.run:
            root = args.home / "runs" / str(cfg.TASK_NAME)
            existing = set(root.glob("*/armt_stage_final.bin")) if root.exists() else set()
            subprocess.run(command, check=True, env=env)
            candidates = set(root.glob("*/armt_stage_final.bin")) - existing
            if len(candidates) != 1:
                raise RuntimeError(f"Expected exactly one new final-stage checkpoint under {root}; got {candidates}")
            checkpoint = candidates.pop()
            entry["final_checkpoint"] = str(checkpoint)
        manifest["stages"].append(entry)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.run:
        print("Prepared configs only. Cross-stage checkpoints are connected automatically when using --run.")


if __name__ == "__main__":
    main()
