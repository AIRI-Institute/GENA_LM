import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from downstream_tasks.expression_prediction.expression_model_final import (
    ExpressionCounts as _BaseExpressionCounts,
)


logger = logging.getLogger(__name__)


class ExpressionCounts(_BaseExpressionCounts):
    """
    Drop-in replacement for expression_model_final.ExpressionCounts.

    Adds per-backward gradient logging for three top-level components:
    - dna_model          -> self.bert
    - description_model  -> self.desc_model
    - decoder            -> self.decoder

    For every backward pass it writes one JSONL record with:
    - batch_id
    - global_step (if provided)
    - B, N, BxN
    - gradient summary per component
    """

    def __init__(
        self,
        *args,
        grad_log_path: str = "expression_model_gradients.jsonl",
        grad_log_rank_zero_only: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.grad_log_path = str(grad_log_path)
        self.grad_log_rank_zero_only = bool(grad_log_rank_zero_only)

        self._grad_log_fh = None
        self._grad_log_batch_id = 0
        self._grad_log_context: Optional[Dict[str, Any]] = None
        self._grad_log_module_order = (
            "dna_model",
            "description_model",
            "decoder",
        )
        self._grad_log_callback_queued = False

        if self._should_log_gradients():
            grad_log_file = Path(self.grad_log_path).expanduser()
            grad_log_file.parent.mkdir(parents=True, exist_ok=True)
            self._grad_log_fh = grad_log_file.open("a", encoding="utf-8", buffering=1)
            self._write_grad_record(
                {
                    "type": "grad_log_started",
                    "path": str(grad_log_file),
                    "rank_zero_only": self.grad_log_rank_zero_only,
                }
            )

    def _should_log_gradients(self) -> bool:
        if not self.grad_log_rank_zero_only:
            return True
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @staticmethod
    def _collect_grad_stats(module: torch.nn.Module) -> Dict[str, Any]:
        total_sq = 0.0
        max_abs = 0.0
        n_tensors = 0
        n_none = 0
        n_nonfinite = 0

        for _, param in module.named_parameters(recurse=True):
            if param is None or not param.requires_grad:
                continue
            if param.grad is None:
                n_none += 1
                continue

            grad = param.grad.detach().float()
            n_tensors += 1

            if not torch.isfinite(grad).all():
                n_nonfinite += 1

            grad_l2 = float(torch.norm(grad, p=2).item())
            total_sq += grad_l2 * grad_l2

            grad_absmax = float(grad.abs().max().item())
            if grad_absmax > max_abs:
                max_abs = grad_absmax

        return {
            "grad_l2": math.sqrt(total_sq) if total_sq > 0 else 0.0,
            "grad_absmax": max_abs,
            "grad_tensors": int(n_tensors),
            "grad_none": int(n_none),
            "grad_nonfinite_tensors": int(n_nonfinite),
        }

    def _start_grad_batch(self, B: int, N: int, BxN: int, global_step: Optional[int]):
        if self._grad_log_fh is None:
            return
        self._grad_log_batch_id += 1
        self._grad_log_context = {
            "type": "batch_gradients",
            "batch_id": int(self._grad_log_batch_id),
            "global_step": int(global_step) if global_step is not None else None,
            "B": int(B),
            "N": int(N),
            "BxN": int(BxN),
        }
        self._grad_log_callback_queued = False

    def _write_grad_record(self, record: Dict[str, Any]):
        if self._grad_log_fh is None:
            return
        self._grad_log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _queue_grad_flush_callback(self):
        if self._grad_log_fh is None or self._grad_log_context is None or self._grad_log_callback_queued:
            return
        engine = getattr(torch.autograd.Variable, "_execution_engine", None)
        if engine is None or not hasattr(engine, "queue_callback"):
            logger.warning("[gradlog] torch execution engine has no queue_callback; skipping grad log for this batch")
            return
        engine.queue_callback(self._flush_grad_record)
        self._grad_log_callback_queued = True

    def _register_loss_grad_hook(self, loss: torch.Tensor):
        if self._grad_log_fh is None or self._grad_log_context is None or loss is None or not loss.requires_grad:
            return

        def _hook(grad: torch.Tensor):
            self._queue_grad_flush_callback()
            return grad

        loss.register_hook(_hook)

    def _flush_grad_record(self):
        if self._grad_log_context is None:
            return

        record = dict(self._grad_log_context)
        modules = {
            "dna_model": self.bert,
            "description_model": self.desc_model,
            "decoder": self.decoder,
        }
        record["modules"] = {
            name: self._collect_grad_stats(modules[name])
            for name in self._grad_log_module_order
        }
        self._write_grad_record(record)

        logger.info(
            "[gradlog] batch_id=%s global_step=%s B=%s N=%s BxN=%s dna_l2=%.6g desc_l2=%.6g decoder_l2=%.6g",
            record["batch_id"],
            record["global_step"],
            record["B"],
            record["N"],
            record["BxN"],
            record["modules"]["dna_model"]["grad_l2"],
            record["modules"]["description_model"]["grad_l2"],
            record["modules"]["decoder"]["grad_l2"],
        )

        self._grad_log_context = None
        self._grad_log_callback_queued = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_mask=None,
        labels=None,
        return_dict=None,
        desc_input_ids=None,
        desc_attention_mask=None,
        dataset_flag=None,
        global_step: Optional[int] = None,
    ):
        if dataset_flag is not None and self.training and torch.is_grad_enabled():
            B, N = dataset_flag.shape
            if input_ids is not None:
                if input_ids.dim() == 3:
                    BxN = int(input_ids.shape[0] * input_ids.shape[1])
                else:
                    BxN = int(input_ids.shape[0])
            else:
                BxN = int(B * N)
            self._start_grad_batch(B=B, N=N, BxN=BxN, global_step=global_step)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels_mask=labels_mask,
            labels=labels,
            return_dict=return_dict,
            desc_input_ids=desc_input_ids,
            desc_attention_mask=desc_attention_mask,
            dataset_flag=dataset_flag,
        )

        loss = None
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            loss = outputs[0]

        if self.training and torch.is_grad_enabled() and loss is not None:
            self._register_loss_grad_hook(loss)

        return outputs

    def __del__(self):
        try:
            if getattr(self, "_grad_log_fh", None) is not None:
                self._grad_log_fh.close()
        except Exception:
            pass
