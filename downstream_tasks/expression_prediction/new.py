"""Opt-in ARMT entry point: identical model/loss/data, cheaper metric plumbing.

Launch as --module downstream_tasks.expression_prediction.new, with the SAME
arguments as run_expression_finetuning_armt_center. Existing jobs are untouched.
Only intended for that runner's CLS-based keep_for_metrics_fn contract.
"""

from collections import defaultdict
from functools import wraps
from itertools import chain
import traceback
import os

import torch
import torch.distributed as dist
import numpy as np


def compact_metric_output(output):
    """Slice CLS before D2H; batch scalar D2H without dtype/value changes.

    The original output and its autograd graph are NOT modified. In particular,
    loss, full signal logits, all five segment losses and supervision stay intact.
    """
    compact = dict(output)
    for key in ('logits', 'labels_reshaped', 'labels_mask_reshaped'):
        tensor = output[key]
        if tensor is None or tensor.ndim != 3 or tensor.shape[1] < 1:
            raise ValueError(f'Expected nonempty [rows, tokens, 1] {key}')
        compact[key] = tensor.detach()[:, :1, :]

    diagnostics = dict(output.get('armt_metrics') or {})
    if output.get('armt_metrics') is not None:
        compact['armt_metrics'] = diagnostics
    groups = defaultdict(list)
    for key, value in diagnostics.items():
        if key.startswith('armt_') and isinstance(value, torch.Tensor) and value.numel() == 1:
            groups[(value.device, value.dtype)].append((diagnostics, key, value))
    for key in ('cls_loss', 'other_loss', 'multinomial_loss', 'deviation_loss'):
        value = output.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            groups[(value.device, value.dtype)].append((compact, key, value))
    for (device, _), entries in groups.items():
        if device.type == 'cpu':
            continue
        host = torch.stack([value.detach().reshape(()) for _, _, value in entries]).cpu()
        for index, (mapping, key, value) in enumerate(entries):
            mapping[key] = host[index].reshape(value.shape)
    return compact


def compact_keep_for_metrics(keep):
    @wraps(keep)
    def wrapped(batch, output):
        return keep(batch, compact_metric_output(output))
    return wrapped


def main_process_metrics(metrics_fn, accelerator):
    """Compute already-gathered global metrics once, return them to EVERY rank.

    All ranks must still participate in the base trainer's collectives. This is
    NOT a rank0-only collect_metrics implementation. Original metric algorithms,
    ordering, logging cadence and rank0 CSV side effects are preserved.
    """
    @wraps(metrics_fn)
    def wrapped(data):
        if accelerator.num_processes == 1:
            return metrics_fn(data)
        if not dist.is_initialized() or dist.get_world_size() != accelerator.num_processes:
            raise RuntimeError('Metric broadcasting requires the accelerator process group')
        result = [None]
        if accelerator.is_main_process:
            try:
                result[0] = (True, metrics_fn(data))
            except Exception:
                # Publish rank0 failures so other ranks do not hang in broadcast.
                result[0] = (False, traceback.format_exc())
        dist.broadcast_object_list(result, src=0)
        success, payload = result[0]
        if not success:
            raise RuntimeError('Global metric computation failed on rank0:\n' + payload)
        return payload
    return wrapped


def make_new_trainer(base_trainer, *, rank0_metrics=False):
    class NewTrainer(base_trainer):
        def __init__(self, args, accelerator, *positional, **kwargs):
            # The original runner supplies these two callbacks by keyword.
            if kwargs.get('keep_for_metrics_fn') is not None:
                kwargs['keep_for_metrics_fn'] = compact_keep_for_metrics(kwargs['keep_for_metrics_fn'])
            if rank0_metrics and kwargs.get('metrics_fn') is not None:
                kwargs['metrics_fn'] = main_process_metrics(kwargs['metrics_fn'], accelerator)
            super().__init__(args, accelerator, *positional, **kwargs)

        def collect_metrics(self, split):
            # One object collective instead of one PER metric/data key. Keep
            # rank order, per-rank list order, dtype and reduction algorithms.
            from accelerate.utils import gather_object
            has_data = self.keep_for_metrics_fn is not None and self.metrics_fn is not None
            local = {'batch': dict(self.batch_metrics[split]),
                     'data': dict(self.metrics_data[split]) if has_data else {}}
            ranks = gather_object([local])
            metrics = {}
            for key in sorted(set().union(*(r['batch'] for r in ranks))):
                values = list(chain.from_iterable(r['batch'].get(key, []) for r in ranks))
                metrics[key] = np.mean(values)
            if has_data:
                data = {}
                for key in sorted(set().union(*(r['data'] for r in ranks))):
                    values = list(chain.from_iterable(r['data'].get(key, []) for r in ranks))
                    shape = getattr(values[0], 'shape', None)
                    if shape is None:
                        data[key] = list(chain.from_iterable(values))
                    elif len(shape) == 0:
                        data[key] = torch.stack(values)
                    elif all(shape[1:] == value.shape[1:] for value in values):
                        data[key] = torch.cat(values)
                    else:
                        data[key] = list(chain.from_iterable(value.tolist() for value in values))
                metrics.update(self.metrics_fn(data))
            self._reset_batch_metrics(split)
            self._reset_metrics_data(split)
            self.metrics[split] = metrics
            return metrics
    return NewTrainer


def main():
    from downstream_tasks.expression_prediction import run_expression_finetuning_armt_center as original
    # Explicit entry-point binding only; no original file or global package patch.
    previous = original.Trainer
    # CPU work reduction was verified, but end-to-end latency was not. Keep this
    # optional until a full distributed A/B shows a wall-time improvement.
    original.Trainer = make_new_trainer(
        previous, rank0_metrics=os.environ.get('ARMT_NEW_RANK0_METRICS', '0') == '1')
    try:
        original.main()
    finally:
        original.Trainer = previous


if __name__ == '__main__':
    main()
