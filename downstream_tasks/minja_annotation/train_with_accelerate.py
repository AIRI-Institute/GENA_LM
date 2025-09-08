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

# Fix for PyTorch 2.6+ weights_only default change
import numpy
torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType
])

def gradient_accumulation_steps(batch_size: int, total_batch_size: int) -> int:
	return min(1, math.ceil(total_batch_size / batch_size))

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='path to the experiment config') 
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')
parser.add_argument('--output_dir', type=str, help='output directory')

def main():
	# Initialize accelerator
	accelerator = Accelerator()
	
	# Set up logging
	args = parser.parse_args()
	logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=args.log_level)
	logger = logging.getLogger()
	experiment_config_path = Path(args.config).expanduser().absolute()

	if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
		os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

	logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
	logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

	with initialize_config_dir(str(experiment_config_path.parents[0])):
		experiment_config = compose(config_name=experiment_config_path.name)

	output_dir = experiment_config.output_dir
	logger.info(f"Output directory: {output_dir}")
	# copy experiment config to output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

	# Initialize Trainer with custom trainer class
	trainer_config = experiment_config.trainer.copy()
	trainer = instantiate(trainer_config)
	
	# Prepare trainer with accelerator
	trainer = accelerator.prepare(trainer)

	# Train and evaluate
	resume_from_checkpoint = experiment_config.get('resume_from_checkpoint', None)
	trainer.train(resume_from_checkpoint=resume_from_checkpoint)
	trainer.evaluate()

if __name__ == "__main__":
	main()
