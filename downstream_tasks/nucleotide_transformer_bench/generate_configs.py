from itertools import product
from pathlib import Path
import argparse
from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--out_dir", default="configs/")
    args = p.parse_args()

    BATCH_SIZES = [32, 64, 128]
    LRS = [1e-5, 3e-5, 5e-5]
    WEIGHT_DECAYS = [1e-2, 1e-3, 1e-4]

    base_cfg = {
        "HOME_PATH": "${oc.env:GENALM_HOME}",
        "args_params": {
            "_target_": "builtins.dict",

            "model": args.model,
            "attn": args.attn,

            "n_folds": 10,
            "stratified": True,
            "seed": 42,

            "tokenizer": "AIRI-Institute/gena-lm-bert-base-t2t",

            "input_seq_len": 1024,
            "data_n_workers": 1,

            "iters": 20000,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "lr": 5e-5,
            "lr_scheduler": "cosine",
            "num_warmup_steps": 1000,
            "optimizer": "AdamW",
            "weight_decay": 0.0001,

            "reset_lr": True,
            "reset_optimizer": True,
            "reset_iteration": True,
            "optimize_metric": "mcc",
            "optimize_mode": "max",
            "save_best": True,
            "valid_interval": 30,
            "log_interval": 30,
            "early_stopping_patience": 50,
            "validate_only": False,
        },
    }

    model_name = args.model.rstrip("/").split("/")[-1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bs, lr, wd in product(BATCH_SIZES, LRS, WEIGHT_DECAYS):
        cfg = OmegaConf.create(base_cfg)
        cfg.args_params.batch_size = bs
        cfg.args_params.lr = lr
        cfg.args_params.hf_model_name = args.model
        cfg.args_params.attn_implementation = args.attn
        cfg.args_params.weight_decay = wd
        path = out_dir / f"{model_name}_bs{bs}_lr{lr:.0e}_wd{wd:.0e}.yaml".replace("+", "")
        OmegaConf.save(cfg, str(path))
        print(path)


if __name__ == "__main__":
    main()
