import torch
from transformers import Trainer
# from transformers.trainer import _is_peft_model
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.utilities import dim_zero_cat
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from transformers.trainer_callback import TrainerCallback

class NamedMean(Metric):
    def __init__(self, cls_name: str, **kwargs):
        super().__init__(**kwargs)
        self.cls_name = cls_name
        self.log_name = f"{cls_name}"
        self.add_state("sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
    
    def update(self, inputs, model_outputs):
        self.sum += model_outputs[self.cls_name]
        self.count += 1
    
    def compute(self):
        return self.sum / self.count

class NamedBinaryAveragePrecision(BinaryAveragePrecision):
    def __init__(self, cls_name: str, **kwargs):
        super().__init__(**kwargs)
        self.cls_name = cls_name
        self.log_name = f"PRAUC_{cls_name}"
    
    def update(self, inputs, model_outputs):
        targets = inputs["targets"]
        predicts = model_outputs["logits"]
        class_index = 0
        updated = False
        for class_name in ["tss", "polya"]:
            for strand in ["+", "-"]:
                cls_name = f"primary_{class_name}_{strand}"
                if self.cls_name == cls_name:
                    X = predicts[:, :, class_index]
                    Y = targets[cls_name] > 0.5
                    super().update(X, Y)
                    updated = True
                    break
                class_index += 1
            if updated: break
        if predicts.shape[-1] == 6:
            for lidx, strand in enumerate(['+', '-']):
                cls_name = f"intragenic_regions_{strand}"
                if self.cls_name == cls_name:
                    X = predicts[:, :, 4 + lidx]
                    Y = targets[cls_name] > 0.5
                    super().update(X, Y)
                    updated = True
                    break
        if not updated:
            raise ValueError(f"Class name {self.cls_name} not found in targets: \n{targets.keys()}")
            
class LogTrainMetricsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer = state.trainer_instance
        if trainer.log_train_metrics is not None and \
            state.global_step % trainer.log_train_metrics == 0 and \
            state.is_world_process_zero:
            
            for m in trainer.train_metrics:
                result = m.compute().cpu().numpy().item()
                m.reset()
                original_should_log_state = control.should_log
                trainer.log({f'train_{m.log_name}': result}) # this will set should_log to False
                control.should_log = original_should_log_state
        control.is_in_train_step = False
        self.trainer_instance = None # remove trainer instance from state to avoid looping
        return control

class DetectTrainStepStart(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        control.is_in_train_step = True # save train step flag to state
        return control

class DataloaderWithEpochReseed(DataLoader):    
    def set_epoch(self, epoch):
        if hasattr(self.dataset, 'reseed_epoch'):
            self.dataset.reseed_epoch(epoch+1)

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
        
        return DataloaderWithEpochReseed(
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
        
        return DataloaderWithEpochReseed(
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model=model, inputs=inputs, return_outputs=True, **kwargs)
        if self.log_train_metrics is not None and self.control.is_in_train_step:
            self.state.trainer_instance = self # save trainer instance to state   
            # accumulate metrics
            for m in self.train_metrics:
                m.update(inputs, outputs)

        return (loss, outputs) if return_outputs else loss