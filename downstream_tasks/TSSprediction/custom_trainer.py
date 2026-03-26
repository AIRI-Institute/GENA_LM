import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback

from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision


class NamedMean(Metric):
    def __init__(self, cls_name: str, **kwargs):
        super().__init__(**kwargs)
        self.cls_name = cls_name
        self.log_name = f"{cls_name}"
        # Keep distributed reduction as you had it; compute() will sync across ranks.
        self.add_state("sum", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, inputs, model_outputs):
        v = model_outputs[self.cls_name]
        if isinstance(v, torch.Tensor):
            v = v.detach().float()
        self.sum += v
        self.count += 1.0

    def compute(self):
        return self.sum / torch.clamp_min(self.count, 1.0)


class NamedBinaryAveragePrecision(BinaryAveragePrecision):
    def __init__(self, cls_name: str, **kwargs):
        super().__init__(**kwargs)
        self.cls_name = cls_name
        self.log_name = f"PRAUC_{cls_name}"

    def update(self, inputs, model_outputs):
        targets = inputs["targets"]
        predicts = model_outputs["logits"]

        if isinstance(predicts, torch.Tensor):
            predicts = predicts.detach()
        # targets is a dict of tensors; keep them as tensors for torchmetrics
        class_index = 0
        updated = False

        for class_name in ["tss", "polya"]:
            for strand in ["+", "-"]:
                cls_name = f"primary_{class_name}_{strand}"
                if self.cls_name == cls_name:
                    X = predicts[:, :, class_index]
                    Y = (targets[cls_name] > 0.5)
                    if isinstance(Y, torch.Tensor):
                        Y = Y.detach()
                    super().update(X, Y)
                    updated = True
                    break
                class_index += 1
            if updated:
                break

        if not updated and predicts.shape[-1] == 6:
            for lidx, strand in enumerate(["+", "-"]):
                cls_name = f"intragenic_regions_{strand}"
                if self.cls_name == cls_name:
                    X = predicts[:, :, 4 + lidx]
                    Y = (targets[cls_name] > 0.5)
                    if isinstance(Y, torch.Tensor):
                        Y = Y.detach()
                    super().update(X, Y)
                    updated = True
                    break

        if not updated:
            raise ValueError(f"Class name {self.cls_name} not found in targets: \n{targets.keys()}")


class DetectTrainStepStart(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        control.is_in_train_step = True
        return control


class LogTrainMetricsCallback(TrainerCallback):
    """
    CRITICAL FIX:
    - ALL ranks must call metric.compute()/reset() on the same steps,
      otherwise TorchMetrics will hang in distributed sync collectives. :contentReference[oaicite:1]{index=1}
    - Only rank 0 logs.
    """
    def on_step_end(self, args, state, control, **kwargs):
        trainer = getattr(state, "trainer_instance", None)
        if trainer is None:
            control.is_in_train_step = False
            return control

        if trainer.log_train_metrics is None:
            control.is_in_train_step = False
            state.trainer_instance = None
            return control

        if state.global_step % trainer.log_train_metrics != 0:
            control.is_in_train_step = False
            state.trainer_instance = None
            return control

        # Compute+reset on ALL ranks to avoid deadlock
        results = {}
        for m in trainer.train_metrics:
            val = m.compute()
            # convert to python scalar
            if isinstance(val, torch.Tensor):
                val_item = val.detach().float().cpu().item()
            else:
                val_item = float(val)
            results[m.log_name] = val_item
            m.reset()

        # Only rank 0 logs
        if state.is_world_process_zero:
            original_should_log_state = control.should_log
            trainer.log({f"train_{k}": v for k, v in results.items()})
            control.should_log = original_should_log_state

        control.is_in_train_step = False
        state.trainer_instance = None
        return control


class DataloaderWithEpochReseed(DataLoader):
    def set_epoch(self, epoch):
        if hasattr(self.dataset, "reseed_epoch"):
            self.dataset.reseed_epoch(epoch + 1)


class CustomTrainer(Trainer):
    """
    Minimal trainer:
    - Forces DistributedSampler when WORLD_SIZE > 1 (so steps/epoch scales with GPUs)
    - Uses worker_init_fn
    - Removes prefetch_factor/persistent_workers extras
    - Fixes TorchMetrics logging deadlock under DDP
    """

    def __init__(self, *args, **kwargs):
        self.log_train_metrics = kwargs.pop("log_train_metrics", None)
        self.train_metrics = kwargs.pop("train_metrics", [])

        callbacks = kwargs.pop("callbacks", [])
        if self.log_train_metrics is not None:
            callbacks.append(LogTrainMetricsCallback())
            callbacks.append(DetectTrainStepStart())
        kwargs["callbacks"] = callbacks

        super().__init__(*args, **kwargs)

        if self.log_train_metrics is not None:
            self.control.is_in_train_step = False
            for m in self.train_metrics:
                m.to(self.args.device)

    @staticmethod
    def _env_world():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        return world_size, rank

    def _get_worker_init_fn(self):
        from simple_annotation_dataset import worker_init_fn
        return worker_init_fn

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None

        world_size, rank = self._env_world()
        if world_size > 1:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
                seed=self.args.seed,
            )
        return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
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
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        world_size, rank = self._env_world()
        if world_size > 1:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
                seed=self.args.seed,
            )
        else:
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model=model, inputs=inputs, return_outputs=True, **kwargs)

        if self.log_train_metrics is not None and getattr(self.control, "is_in_train_step", False):
            # Make trainer accessible to the callback in this process
            self.state.trainer_instance = self
            for m in self.train_metrics:
                m.update(inputs, outputs)

        return (loss, outputs) if return_outputs else loss