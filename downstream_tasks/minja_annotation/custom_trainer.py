import torch
from transformers import Trainer
from transformers.trainer import _is_peft_model
from torch.utils.data import DataLoader
from torchmetrics import Metric
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from transformers.trainer_callback import TrainerCallback

class NamedMean(Metric):
    def __init__(self, cls_name: str, **kwargs):
        super().__init__(**kwargs)
        self.cls_name = cls_name
        self.add_state("sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
    
    def update(self, model_outputs):
        self.sum += model_outputs[self.cls_name]
        self.count += 1
    
    def compute(self):
        return self.sum / self.count

class LogTrainMetricsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer = state.trainer_instance
        if trainer.log_train_metrics is not None and \
            state.global_step % trainer.log_train_metrics == 0 and \
            state.is_world_process_zero:
            
            for m in trainer.train_metrics:
                result = m.compute().cpu().numpy().item()
                m.reset()
                trainer.log({f'train_{m.cls_name}': result})
        control.is_in_train_step = False
        self.trainer_instance = None # remove trainer instance from state to avoid looping
        return control

class DetectTrainStepStart(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        control.is_in_train_step = True # save train step flag to state
        return control

class CustomTrainer(Trainer):
    """Custom trainer that properly handles DataLoader with worker_init_fn for multi-GPU training."""

    def __init__(self, *args, **kwargs):
        self.log_train_metrics = kwargs.pop('log_train_metrics', None)
        if self.log_train_metrics is not None:
            self.train_metrics = kwargs.pop('train_metrics', [])
            callbacks = kwargs.pop('callbacks', [])
            callbacks.append(LogTrainMetricsCallback())
            callbacks.append(DetectTrainStepStart())
            kwargs['callbacks'] = callbacks
        super().__init__(*args, **kwargs)
        if self.log_train_metrics is not None:
            self.control.is_in_train_step = False
            [m.to(self.args.device) for m in self.train_metrics]
    
    def get_train_dataloader(self) -> DataLoader:
        """Override to use worker_init_fn."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = self._get_train_sampler()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=self._get_worker_init_fn(),
        )
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Override to use worker_init_fn."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)
        
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=self._get_worker_init_fn(),
        )
    
    def _get_worker_init_fn(self):
        """Get worker_init_fn from the dataset if available."""
        if hasattr(self.train_dataset, 'worker_init_fn'):
            return self.train_dataset.worker_init_fn
        elif hasattr(self.eval_dataset, 'worker_init_fn'):
            return self.eval_dataset.worker_init_fn
        else:
            # Import the worker_init_fn from the dataset module
            from simple_annotation_dataset import worker_init_fn
            return worker_init_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.log_train_metrics is not None and self.control.is_in_train_step:
            self.state.trainer_instance = self # save trainer instance to state   
            # accumulate metrics
            for m in self.train_metrics:
                m.update(outputs)

        return (loss, outputs) if return_outputs else loss