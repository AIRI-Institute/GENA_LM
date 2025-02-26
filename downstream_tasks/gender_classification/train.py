import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import accelerate
import numpy as np
import torch
import torch.distributed
import transformers
from mammals_gender_dataset import MultiSpeciesGenderDataChunkedDataset, collate_fn, worker_init_fn
from model import GenderChunkedClassifier
from safetensors.torch import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, Trainer, TrainingArguments
from transformers.integrations.integration_utils import TensorBoardCallback, rewrite_logs

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.DEBUG
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


def compute_metrics(p):
    probs, labels = p
    predictions = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': p,
        'recall': r,
        'f1': f1,
        'roc_auc': roc_auc_score(labels, probs)
    }


class TensorBoardCallbackWithTokens(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                    k = k.split('/')
                    split_prefix = k[0]
                    k = '/'.join(k[1:])
                    self.tb_writer.add_scalar(f'{split_prefix}/tokens/{k}', v, state.num_input_tokens_seen)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()


class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
                          worker_init_fn=worker_init_fn)

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
                          worker_init_fn=worker_init_fn)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        num_training_steps = int(num_training_steps / 0.9)  # to make final lr not zero, for linear it is lr/10.
        logger.info(f'setting num_training_steps for lr_scheduler to {num_training_steps}')
        return super().create_scheduler(num_training_steps, optimizer)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        main_input_name = getattr(self.model, "main_input_name", "input_ids")
        input_device = inputs[main_input_name].device
        # fixes num seen tokens device for gather op
        # see https://github.com/huggingface/transformers/issues/29889
        input_device = "cuda" if self.args.distributed_state.backend == "nccl" else input_device
        self.state.num_input_tokens_seen += torch.sum(
            self.accelerator.gather(
                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
            )
        ).item()
        return super().training_step(model, inputs)

    def log(self, logs: Dict[str, float]) -> None:
        logs['num_input_tokens_seen'] = self.state.num_input_tokens_seen
        # log early stopping patience
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs['patience'] = cb.early_stopping_patience_counter
                break
        return super().log(logs)


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    data_path: str = field(
        default='/home/jovyan/mammals_gender_data/',
    )
    n_chunks: Optional[int] = field(default=8)
    chunk_size: Optional[int] = field(default=3072)
    force_sampling_from_y: Optional[bool] = field(default=False)
    chrY_name: Optional[str] = field(default='chrY')
    chrY_ratio: Optional[float] = field(default=None)
    n_valid_samples: Optional[int] = field(default=4096)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='roc_auc')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)
    freeze_backbone: Optional[bool] = field(default=False)
    from_pretrained: Optional[str] = field(default='AIRI-Institute/gena-lm-bert-base-t2t')
    init_checkpoint: Optional[str] = field(default=None)


"""
e.g.,: CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --mixed_precision bf16 \
    --config_file accelerate.yaml train.py --exp_path ./runs/64x256_force_y --n_chunks 64 --chunk_size 256 \
    --per_device_batch_size 32 --force_sampling_from_y
"""
if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    # datasets.utils.logging.set_verbosity(logger.log_level)
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    if accel.is_main_process:
        config = {
            'cli_args': dict(vars(args)),
        }
        logger.info('saving experiment configuration..')
        Path(args.exp_path).mkdir(parents=True)
        json.dump(config, open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    model = AutoModel.from_pretrained(args.from_pretrained, trust_remote_code=True)
    model_module = importlib.import_module(model.__class__.__module__)
    cls = getattr(model_module, 'BertModel')
    model = cls.from_pretrained(args.from_pretrained, add_pooling_layer=False)

    args.data_path = Path(args.data_path)
    dataset = MultiSpeciesGenderDataChunkedDataset(data_path=args.data_path, split_name='train',
                                       n_chunks=args.n_chunks, chunk_size=args.chunk_size,
                                       force_sampling_from_y=args.force_sampling_from_y, chrY_ratio=args.chrY_ratio,
                                       seed=args.seed)

    max_n_samples_per_gpu = args.n_valid_samples // accel.num_processes
    valid_dataset = MultiSpeciesGenderDataChunkedDataset(data_path=args.data_path, split_name='valid',
                                             n_chunks=args.n_chunks, chunk_size=args.chunk_size,
                                             force_sampling_from_y=True, 
                                             max_n_samples=max_n_samples_per_gpu, seed=args.seed+1)

    def preprocess_collate_fn(samples):
        batch = collate_fn(samples)

        batch['chunks'] = np.array(batch['chunks'])
        shape = batch['chunks'].shape
        batch['chunks'] = list(batch['chunks'].flatten())

        tokenized_batch = tokenizer(batch['chunks'], padding='longest', max_length=512,
                                    truncation=True, return_tensors='pt')

        for k in tokenized_batch:
            tokenized_batch[k] = tokenized_batch[k].reshape(*shape, -1)

        batch['labels'] = torch.Tensor(batch['labels'])

        return {
            'input_ids': tokenized_batch['input_ids'],
            'attention_mask': tokenized_batch['attention_mask'],
            'labels': batch['labels']
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    gender_model = GenderChunkedClassifier(model)
    if args.init_checkpoint is not None:
        logger.info(f'loading model weights from {args.init_checkpoint}')
        missing_k, unexpected_k = load_model(gender_model, args.init_checkpoint)
        if len(missing_k) != 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    # freeze backbone model weights if specified
    if args.freeze_backbone:
        for param in gender_model.model.parameters():
            param.requires_grad = False
        logger.info("Backbone model weights are frozen and won't be trained")

    # prepare training arguments
    output_dir = Path(args.exp_path)

    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        args_total_bs = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        assert args.total_batch_size == args_total_bs

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        eval_on_start=True,
        eval_strategy='steps',
        save_strategy='steps',
        report_to='none',  # log to tensorboard with TensorBoardCallbackWithTokens
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    trainer = CustomTrainer(
        model=gender_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=preprocess_collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   TensorBoardCallbackWithTokens
                   ],
    )

    accel = trainer.accelerator

    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(valid_dataset)
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
