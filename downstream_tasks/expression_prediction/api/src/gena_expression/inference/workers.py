"""Persistent, bounded execution of independent forward batches on CUDA replicas."""

from __future__ import annotations

import multiprocessing as mp
import re
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.util import Finalize
from pathlib import Path
from queue import Empty
from threading import Lock
from time import monotonic
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping


@dataclass(frozen=True)
class _ModelLoadSpec:
    """Only the model resources needed to recreate a replica in a worker."""

    model_cls: str
    checkpoint: Path
    config: Path
    precision: Literal["default", "float32"]
    matmul_precision: str | None = None
    cudnn_allow_tf32: bool | None = None


_ForwardInputs = dict[str, Any]
_ForwardJob = tuple[int, _ForwardInputs]
_ReplicaInitializer = Callable[
    [str, _ModelLoadSpec],
    tuple[Callable[[_ForwardInputs], Any], Mapping[str, Any]],
]


def _cuda_devices(device: Any) -> tuple[str, ...]:
    """Return replicas for a CUDA list, or an empty tuple for a legacy device."""

    if not isinstance(device, str):
        return ()
    if device == "gpu" or device.startswith("gpu:"):
        raise ValueError("Use 'cuda' or 'cuda:<id>[,<id>,...]' for CUDA devices.")
    if "," not in device:
        return ()
    if not re.fullmatch(r"cuda:[0-9]+(?:,[0-9]+)+", device):
        raise ValueError("A multi-GPU device must use 'cuda:<id>,<id>[,...]'.")
    indices = tuple(int(value) for value in device.split(":", 1)[1].split(","))
    if len(set(indices)) != len(indices):
        raise ValueError("CUDA device IDs must be distinct.")

    import torch

    visible_count = torch.cuda.device_count()
    if any(index >= visible_count for index in indices):
        raise ValueError(f"Requested CUDA device IDs {indices}, but only {visible_count} CUDA devices are visible.")
    return tuple(f"cuda:{index}" for index in indices)


def _initialize_cuda_replica(device: str, spec: _ModelLoadSpec):
    """Load a model directly on its worker, without loading tokenizers or condition formatters."""

    import torch

    from .model import _call_model, _inference_context, _load_model_runtime

    torch.cuda.set_device(device)
    if spec.matmul_precision is not None:
        torch.set_float32_matmul_precision(spec.matmul_precision)
    if spec.cudnn_allow_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = spec.cudnn_allow_tf32
    model, provenance = _load_model_runtime(spec)
    model = model.to(device)
    model.eval()

    def forward(inputs: _ForwardInputs):
        kwargs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with _inference_context("cuda", True, spec.precision):
            return _call_model(model, kwargs).logits.detach().cpu()

    return forward, provenance


@dataclass(frozen=True)
class _WorkerResponse:
    device: str
    group_id: int | None
    result: Any = None
    error: str | None = None


def _replica_main(device, spec, requests, responses, initialize) -> None:
    """Start one replica and service bounded CPU forward inputs until shutdown."""

    group_id = None
    try:
        forward, provenance = initialize(device, spec)
        responses.put(_WorkerResponse(device, None, dict(provenance)))
        while True:
            job = requests.get()
            if job is None:
                return
            group_id, inputs = job
            result = forward(inputs)
            responses.put(_WorkerResponse(device, group_id, result))
            del inputs, job, result
            group_id = None
    except Exception:
        responses.put(_WorkerResponse(device, group_id, error=traceback.format_exc()))


def _shutdown_pool(processes, requests, responses, *, force=True) -> None:
    """Release spawned processes and queue resources during close or finalization."""

    if not force:
        for _ in processes:
            requests.put_nowait(None)
        deadline = monotonic() + 5.0
        for process in processes.values():
            process.join(max(0.0, deadline - monotonic()))
    for process in processes.values():
        if process.is_alive():
            process.terminate()
    for process in processes.values():
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
    for queue in (requests, responses):
        queue.cancel_join_thread()
        queue.close()


class _CudaWorkerPool:
    """Own one spawned replica per device and yield results in forward-plan order."""

    def __init__(
        self,
        devices: tuple[str, ...],
        spec: _ModelLoadSpec,
        *,
        initialize: _ReplicaInitializer = _initialize_cuda_replica,
        timeout: float = 600.0,
    ):
        self.devices = devices
        self._timeout = timeout
        self._capacity = 2 * len(devices)
        self._closed = False
        self._lock = Lock()
        self._processes = {}
        context = mp.get_context("spawn")
        self._requests = context.Queue(maxsize=self._capacity)
        self._responses = context.Queue(maxsize=self._capacity)
        self._finalizer = Finalize(
            self,
            _shutdown_pool,
            args=(self._processes, self._requests, self._responses),
            exitpriority=10,
        )
        try:
            for device in devices:
                process = context.Process(
                    target=_replica_main,
                    name=f"gena-expression-{device}",
                    args=(device, spec, self._requests, self._responses, initialize),
                    daemon=False,
                )
                process.start()
                self._processes[device] = process

            ready = {}
            deadline = monotonic() + timeout
            while len(ready) < len(devices):
                response = self._receive(deadline, "starting CUDA workers")
                if response.group_id is not None or response.device in ready:
                    raise RuntimeError("Received an unexpected CUDA worker startup response.")
                ready[response.device] = response.result
            self.provenance = ready[devices[0]]
        except BaseException:
            self._close(force=True)
            raise

    def _receive(self, deadline: float, stage: str) -> _WorkerResponse:
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out while {stage} on {', '.join(self.devices)}.")
            try:
                response = self._responses.get(timeout=min(remaining, 0.2))
            except Empty:
                for device, process in self._processes.items():
                    if process.exitcode is not None:
                        raise RuntimeError(
                            f"CUDA worker {device} exited with code {process.exitcode} while {stage}."
                        ) from None
                continue
            if response.error is not None:
                where = "startup" if response.group_id is None else f"forward batch {response.group_id}"
                raise RuntimeError(f"CUDA worker {response.device} failed during {where}:\n{response.error}")
            return response

    def run(self, jobs: Iterable[_ForwardJob]) -> Iterator[tuple[int, str, Any]]:
        """Run a bounded stream of batches and deliver each result in submission order."""

        with self._lock:
            if self._closed:
                raise RuntimeError("CUDA workers are closed. Load a new SequenceModel to resume prediction.")

            order: deque[int] = deque()
            pending: set[int] = set()
            completed: dict[int, _WorkerResponse] = {}
            iterator = iter(jobs)

            def submit() -> bool:
                job = next(iterator, None)
                if job is None:
                    return False
                group_id, _ = job
                if group_id in pending or group_id in completed:
                    raise RuntimeError(f"Duplicate CUDA forward batch {group_id}.")
                self._requests.put_nowait(job)
                order.append(group_id)
                pending.add(group_id)
                return True

            try:
                for _ in range(self._capacity):
                    if not submit():
                        break
                while order:
                    group_id = order[0]
                    deadline = monotonic() + self._timeout
                    while group_id not in completed:
                        response = self._receive(deadline, f"waiting for forward batch {group_id}")
                        response_id = response.group_id
                        if response_id not in pending:
                            raise RuntimeError(f"Received unexpected CUDA forward batch {response_id}.")
                        pending.remove(response_id)
                        completed[response_id] = response
                    response = completed.pop(order.popleft())
                    yield group_id, response.device, response.result
                    submit()
            except BaseException:
                self._close(force=True)
                raise

    def close(self) -> None:
        """Stop workers and reject further work. Safe to call repeatedly."""

        with self._lock:
            self._close(force=False)

    def _close(self, *, force: bool) -> None:
        if self._closed:
            return
        self._closed = True
        self._finalizer.cancel()
        _shutdown_pool(self._processes, self._requests, self._responses, force=force)
