#!/usr/bin/env python
import logging
import os
from pathlib import Path
import shutil
from hydra.utils import instantiate
import torch
from argparse import ArgumentParser
from hydra import initialize_config_dir, compose
import math
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

# Fix for PyTorch 2.6+ weights_only default change
import numpy
torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType
])

def gradient_accumulation_steps(batch_size, total_batch_size, num_processes=1, **_ignored) -> int:
    batch_size = int(batch_size)
    total_batch_size = int(total_batch_size)
    num_processes = int(num_processes) if num_processes is not None else 1

    batch_size = max(1, batch_size)
    total_batch_size = max(1, total_batch_size)
    num_processes = max(1, num_processes)

    # Global batch ~= per_device_batch * world_size * grad_accum
    return max(1, math.ceil(total_batch_size / (batch_size * num_processes)))

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='path to the experiment config')
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')
parser.add_argument('--output_dir', type=str, help='output directory')

def main():
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=args.log_level
    )
    logger = logging.getLogger()

    experiment_config_path = Path(args.config).expanduser().absolute()

    if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

    with initialize_config_dir(str(experiment_config_path.parents[0])):
        experiment_config = compose(config_name=experiment_config_path.name)

    # Match your multi-GPU Accelerate setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = gradient_accumulation_steps(
        batch_size=experiment_config.batch_size,
        total_batch_size=experiment_config.total_batch_size,
        num_processes=world_size
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        kwargs_handlers=[ddp_kwargs]
    )

    # Reduce risk of NCCL barrier “unknown devices” hang
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')
    logger.info(f'accelerator state: {accelerator.state}')
    logger.info(f'computed gradient_accumulation_steps: {grad_accum}')

    output_dir = experiment_config.output_dir
    logger.info(f"Output directory: {output_dir}")

    # Only main process does filesystem writes
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    accelerator.wait_for_everyone()

    # Initialize Trainer with custom trainer class
    trainer_config = experiment_config.trainer.copy()
    trainer = instantiate(trainer_config)

    # Train and evaluate (run on all processes; trainer should handle distributed)
    resume_from_checkpoint = experiment_config.get('resume_from_checkpoint', None)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.evaluate()

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
