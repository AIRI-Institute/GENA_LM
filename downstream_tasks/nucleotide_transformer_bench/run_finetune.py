import json
import logging
import os
from pathlib import Path
from datetime import datetime

from accelerate import Accelerator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, ModernBertForSequenceClassification 
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs

from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff, prepare_run
import lm_experiments_tools.optimizers as optimizers
from lm_experiments_tools import get_optimizer

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from transformers import DataCollatorWithPadding

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

torch.set_num_threads(4)

parser = HfArgumentParser(TrainerArgs)
parser.add_argument("--experiment_config", type=str, required=True, help="path to hydra experiment yaml")
parser.add_argument("--valid_fold", type=int, default=None, help="override valid fold [0..n_folds-1]")


if __name__ == "__main__":
    args = parser.parse_args()
    cli_valid_fold = args.valid_fold

    cfg_path = Path(args.experiment_config).expanduser().absolute()
    with initialize_config_dir(str(cfg_path.parent)):
        cfg = compose(config_name=cfg_path.name)

    if "args_params" in cfg:
        cfg_args = instantiate(cfg["args_params"])  
        for k, v in cfg_args.items():
            setattr(args, k, v)

    if cli_valid_fold is not None:
        args.valid_fold = int(cli_valid_fold)

    def _compact_sci(x: float) -> str:
        s = f"{float(x):.0e}"          
        mant, exp = s.split("e")
        exp_i = int(exp)
        if exp_i < 0:
            return f"{mant}e{abs(exp_i)}"
        return f"{mant}e{exp_i}"

    def _infer_model_tag(hf_model_name: str) -> str:
        name = Path(hf_model_name).name
        parts = name.split("_")
        return parts[-1] if len(parts) >= 2 else name

    def build_model_path(args) -> str:
        base = Path("runs") / "NT_benchmark" / str(args.task_name)

        model_tag = _infer_model_tag(str(args.hf_model_name))
        lr_tag = _compact_sci(args.lr)
        wd_tag = _compact_sci(args.weight_decay)

        run_name = (
            f"{args.task_name}_{model_tag}"
            f"_lr{lr_tag}_wd{wd_tag}"
            f"_bs{args.batch_size}"
        )

        if int(getattr(args, "gradient_accumulation_steps", 1)) != 1:
            run_name += f"_ga{int(args.gradient_accumulation_steps)}"

        if getattr(args, "early_stopping_patience", None) is not None:
            run_name += f"_p{int(args.early_stopping_patience)}"
        if getattr(args, "seed", None) is not None:
            run_name += f"_seed{int(args.seed)}"

        fold = int(getattr(args, "valid_fold", 0))
        return str(base / run_name / f"fold_{fold}")

    if getattr(args, "model_path", None) in (None, "", "null"):
        args.model_path = build_model_path(args)
        print(f"[auto] model_path = {args.model_path}")

    # Accelerate
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    if accelerator.is_main_process:
        logger.info(f"num processes: {world_size}")

    if accelerator.is_main_process and args.model_path is None:
        logger.warning("model_path is not set: config, logs and checkpoints will not be saved.")

    # Сreate model path and save configuration
    if accelerator.is_main_process and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        json.dump(args_dict, open(model_path / "config.json", "w"), indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def tokenization(batch):
        tok = tokenizer(
            batch["sequence"],
            add_special_tokens=True,
            truncation=True,
            max_length=args.input_seq_len,
            padding=False,
        )
        tok["labels"] = batch["label"]       
        return tok

    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",        
        return_tensors="pt",
        pad_to_multiple_of=8  )


    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"batch size = {args.batch_size} | gradient accumulation steps = {args.gradient_accumulation_steps}")
    global_batch_size = per_worker_batch_size * world_size
    kwargs = {"pin_memory": True, "num_workers": args.data_n_workers}

    # Load task config
    task = args.task_name
    # benchmark_config = pd.read_csv(
    #     "/home/jovyan/dnalm/downstream_tasks/nucleotide_transformer_downstream_tasks/nucl_tf_bench_config.csv"
    # )
    # metric_name = benchmark_config[benchmark_config["task"] == task]["metric_name"]
    cls_num = args.cls_num
    problem_type = "single_label_classification"

    # Load dataset 
    # get train dataset
    if accelerator.is_main_process:
        logger.info(f"preparing training data for: {task}")
    # load dataset
    dataset_all = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    dataset = dataset_all["train"].filter(lambda ex: ex["task"] == task)
    # split train set on new train and valid sets
    data_ind = np.arange(len(dataset))

    use_stratified = bool(getattr(args, "stratified", False)) and ("label" in dataset.column_names)
    y = np.array(dataset["label"]) if use_stratified else None

    splitter = (
        StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        if use_stratified
        else KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    )
    folds = list(splitter.split(data_ind, y))
    fold_id = int(args.valid_fold)

    if fold_id < 0 or fold_id >= len(folds):
        raise ValueError(f"--valid_fold must be in [0..{len(folds)-1}], got {fold_id}")

    train_ind, valid_ind = folds[fold_id]
    train_ind = train_ind.tolist()
    valid_ind = valid_ind.tolist()

    if accelerator.is_main_process:
        logger.info(
            f"10-fold split: fold={fold_id}/{args.n_folds} | "
            f"train={len(train_ind)} | valid={len(valid_ind)} | stratified={use_stratified}"
        )

    train_dataset = dataset[train_ind]
    train_dataset = Dataset.from_dict(train_dataset)
    train_dataset = train_dataset.map(
        tokenization,
        batched=True,
        remove_columns=train_dataset.column_names,   
    )
    train_sampler = DistributedSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        drop_last=False,
        seed=args.seed
    )
    train_dataloader = DataLoader(
    train_dataset,
    batch_size=per_worker_batch_size,
    sampler=train_sampler,
    collate_fn=data_collator,  
    **kwargs
)

    if accelerator.is_main_process:
        logger.info(f"len(train_dataset): {len(train_dataset)}")

    if accelerator.is_main_process:
        logger.info(f"preparing validation data for: {task}")

    valid_dataset = dataset[valid_ind]
    valid_dataset = Dataset.from_dict(valid_dataset)
    valid_dataset = valid_dataset.map(
        tokenization,
        batched=True,
        remove_columns=valid_dataset.column_names,
    )
    valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=world_size, shuffle=False)
    valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=per_worker_batch_size,
    sampler=valid_sampler,
    collate_fn=data_collator,   
    **kwargs
)

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    if accelerator.is_main_process:
        logger.info(f"len(valid_dataset): {len(valid_dataset)}")

    # Define model 
    hf_model_name = getattr(args, "hf_model_name", None)
    attn_impl = getattr(args, "attn_implementation", None)

    if hf_model_name is None:
        raise ValueError("For ModernBERT loading you must set args.hf_model_name in config (or via CLI).")

    model_cfg = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
    model_cfg.num_labels = cls_num
    model_cfg.problem_type = problem_type

    if accelerator.is_main_process:
        logger.info(f"Loading ModernBERT for classification from: {hf_model_name}")
        logger.info(f"num_labels={cls_num} | attn_implementation={attn_impl}")

    model = ModernBertForSequenceClassification.from_pretrained(
        hf_model_name,
        config=model_cfg,
        trust_remote_code=True,
        attn_implementation=attn_impl)  

    p = float(getattr(args, "dropout", 0.1))

    if hasattr(model, "classifier"):
        model.classifier = nn.Sequential(nn.Dropout(p), model.classifier)
        if accelerator.is_main_process:
            logger.info(f"Wrapped model.classifier with Dropout(p={p})")
    elif hasattr(model, "score"):
        model.score = nn.Sequential(nn.Dropout(p), model.score)
        if accelerator.is_main_process:
            logger.info(f"Wrapped model.score with Dropout(p={p})")
    else:
        if accelerator.is_main_process:
            logger.warning("No classifier/score head found to wrap with dropout.")


    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f"{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization")

    if accelerator.is_main_process:
        logger.info(f"Using optimizer class: {optimizer_cls}")

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(
            model.parameters(),
            lr=args.lr,
            scale_parameter=args.scale_parameter,
            relative_step=args.relative_step,
            warmup_init=args.warmup_init,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer = accelerator.prepare(model, optimizer)

    def batch_transform_fn(batch):
        return {
            "input_ids": batch["input_ids"],
            "token_type_ids": batch["token_type_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }

    def keep_for_metrics_fn(batch, output):
        data = {}
        data["labels"] = batch["labels"]
        data["predictions"] = output["logits"].detach()
        return data

    def metrics_fn(data):
        metrics = {}
        y = data["labels"].detach().cpu().numpy()
        p = torch.argmax(data["predictions"], dim=1).detach().cpu().numpy()
        metrics["mcc"] = matthews_corrcoef(y, p)
        metrics["acc"] = accuracy_score(y, p)
        metrics["f1_macro"] = f1_score(y, p, average="macro")
        metrics["f1_micro"] = f1_score(y, p, average="micro")
        metrics["f1_weighted"] = f1_score(y, p, average="weighted")
        return metrics
    
    b = next(iter(train_dataloader))
    print("keys:", b.keys())
    print("input_ids:", b["input_ids"].shape, b["input_ids"].dtype, "min/max", b["input_ids"].min().item(), b["input_ids"].max().item())
    print("attention_mask:", b["attention_mask"].shape, b["attention_mask"].dtype, "sum", b["attention_mask"].sum().item())
    print("labels:", b["labels"].shape, b["labels"].dtype, "unique", torch.unique(b["labels"])[:10])

    for k, v in b.items():
        if torch.is_tensor(v) and v.is_floating_point():
            if torch.isnan(v).any():
                print("NaN in", k)


    trainer = Trainer(
        args,
        accelerator,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader=valid_dataloader,
        train_sampler=train_sampler,
        batch_transform_fn=batch_transform_fn,
        keep_for_metrics_fn=keep_for_metrics_fn,
        metrics_fn=metrics_fn
    )

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        # if args.save_best:
        #     best_model_path = str(Path(args.model_path) / "model_best") 
        #     if accelerator.is_main_process:
        #         logger.info(f"Loading best saved model from {best_model_path}")
        #     trainer.load(best_model_path, weights_only=False)

    # # get test dataset
    # if accelerator.is_main_process:
    #     logger.info(f"preparing test data for: {task}")

    # test_dataset = dataset_all["test"].filter(lambda ex: ex["task"] == task)
    # test_dataset = test_dataset.map(tokenization, batched=True)
    # test_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    # test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler, **kwargs)

    # if accelerator.is_main_process:
    #     logger.info(f"len(test_dataset): {len(test_dataset)}")
    #     logger.info("Runnning validation on test data:")

    # metrics = trainer.validate(
    #     test_dataloader,
    #     split="test",
    #     write_tb=accelerator.is_main_process,   
    # )

    # accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    #     results_dir = Path(args.model_path) / "results"
    #     results_dir.mkdir(parents=True, exist_ok=True)

    #     payload = {
    #         "time": datetime.now().isoformat(timespec="seconds"),
    #         "task": task,
    #         "split": "test",
    #         "metrics": metrics,
    #     }

    #     out_json = results_dir / f"metrics_{task}_test.json"
    #     with open(out_json, "w") as f:
    #         json.dump(payload, f, indent=2)

    #     logger.info(f"Saved test metrics to: {out_json}")

