from hydra import initialize_config_dir, compose
from argparse import ArgumentParser
import logging
import os
from pathlib import Path
import shutil
from hydra.utils import instantiate
import torch
from simple_annotation_dataset import collate_fn
from typing import List, Dict, Tuple
import pyBigWig as bw
from safetensors.torch import load_file
from tqdm import tqdm
import numpy as np

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='path to the experiment config') 
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')
parser.add_argument('--output_dir', type=str, help='output directory')

class bigWigExporter:
	def __init__(self, output_dir: str, chroms: List[str], chrom_lengths: List[int], logger: logging.Logger = None, context_fraction: float = 0.1):
		self.output_dir = output_dir
		self.bigWigFiles = {}
		self.chroms = chroms
		self.chrom_lengths = chrom_lengths
		self.logger = logger if logger is not None else logging.getLogger(__name__)
		self.context_fraction = context_fraction
		self.last_chrom = None
		self.last_end = None
		self.last_value = {}
		self.sigmoid = torch.nn.Sigmoid()
		self.chrom_data = {}
	
	def open_bw(self, output_name: str):
		self.bigWigFiles[output_name] = bw.open(os.path.join(self.output_dir, output_name+".bw"), "w")
		for chrom, length in zip(self.chroms, self.chrom_lengths):
			self.bigWigFiles[output_name].addHeader([(chrom, length)])
	
	def write_data(self):
		for chrom, data in self.chrom_data.items():
			for label, values in data.items():
				if label not in self.bigWigFiles.keys():
					self.open_bw(label)
				first_nonzero = (~np.isnan(values)).nonzero()[0][0]
				last_nonzero = (~np.isnan(values)).nonzero()[0][-1]
				print (first_nonzero, last_nonzero)
				starts = np.arange(first_nonzero, last_nonzero, dtype=np.int32)
				ends = np.arange(first_nonzero, last_nonzero, dtype=np.int32) + 1
				values = np.nan_to_num(values[first_nonzero:last_nonzero], -1).astype(np.float32)
				self.logger.info(f"Saving bigWig file for {label} with {len(starts)} entries")
				self.bigWigFiles[label].addEntries([chrom]*len(starts), 
												  starts.tolist(),
												  ends=ends.tolist(),
												  values=values.tolist(),
												  )

	def process_batch(self, outputs, metadata: List[Dict]):
		for s,m in zip(outputs["predicts"], metadata):
			s_om = np.array(m['offset_mapping'])
			s_chrom = m['chrom']
			if s_chrom not in self.chrom_data.keys():
				self.chrom_data[s_chrom] = {}
			s_start = min(1, int(self.context_fraction * len(s)))
			s_end = max(len(s)-1, int((1-self.context_fraction) * len(s)))

			# outputs have shape of num_tokens x num_labels
			class_number = 0
			for class_name in ["tss", "polya"]:
				for strand in ["+", "-"]:
					label = f"{class_name}_{strand}"
					if label not in self.chrom_data[s_chrom]:
						self.chrom_data[s_chrom][label] = np.empty(shape=(self.chrom_lengths[self.chroms.index(s_chrom)],))
						self.chrom_data[s_chrom][label][:] = np.nan
					values = self.sigmoid(s[s_start:s_end, class_number]).cpu().numpy()
					for v,(start,end) in zip(values, s_om[s_start:s_end]):
						if start == end:
							continue
						if np.isnan(self.chrom_data[s_chrom][label][m['start']+start:m['start']+end]).any():
							self.chrom_data[s_chrom][label][m['start']+start:m['start']+end] = v
						else:
							self.chrom_data[s_chrom][label][m['start']+start:m['start']+end] = (
								np.mean(self.chrom_data[s_chrom][label][m['start']+start:m['start']+end]) + v) / 2
					class_number += 1

	def write_data_and_close(self):
		self.write_data()
		for k in self.bigWigFiles.keys():
			self.bigWigFiles[k].close()

def main():	
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

	model_config = experiment_config.model
	model = instantiate(model_config)
	state = load_file(experiment_config.model_cpt)
	model.load_state_dict(state, strict=True)
	model.to(torch.device("cuda"))
	model.eval()

	dataset_config = experiment_config.eval_dataset
	dataset = instantiate(dataset_config)
	dataset.open_files()

	exporter = bigWigExporter(output_dir, list(dataset.chrom_info.keys()), list(dataset.chrom_info.values()), logger=logger)

	region = experiment_config.get('region', None)
	if region is not None:
		target_chrom = region.split(':')[0]
		target_start = int(region.split(':')[1].split('-')[0])
		target_end = int(region.split(':')[1].split('-')[1])

	sample_id = 0
	samples_written = 0

	progress_bar = tqdm(total=len(dataset), desc="Processing samples")

	while sample_id < len(dataset):
		slice_start = sample_id
		slice_end = min(sample_id+experiment_config.batch_size, len(dataset))

		if region is not None: 		# prefetch batch coordinates and check if we need to process them
			batch_coordinates = [dataset._decode_item_id(i) for i in range(slice_start, slice_end)]
			need_to_process = False
			for chrom, start, end in batch_coordinates:
				if chrom == target_chrom and start >= target_start and end <= target_end:
					need_to_process = True
					break
			if not need_to_process:
				sample_id += experiment_config.batch_size
				progress_bar.update(slice_end - slice_start)
				if batch_coordinates[-1][0] == target_chrom and batch_coordinates[-1][1] > target_end:
					logger.info(f"Reached end of region {region}")
					break
				continue
		
		batch = [dataset[i] for i in range(slice_start, slice_end)]		
		progress_bar.update(len(batch))
		
		if region is not None: # filter batch by region
			region_filtered_batch = []
			region_filtered_metadata = []
			for b in batch:
				if b['metadata']['chrom'] == target_chrom and \
					b['metadata']['start'] >= target_start and \
						b['metadata']['end'] <= target_end:

					region_filtered_batch.append(b)
					region_filtered_metadata.append(b['metadata'])
			
			assert len(region_filtered_batch) > 0
			batch = region_filtered_batch
			metadata = region_filtered_metadata
		else:
			metadata = [b['metadata'] for b in batch]

		samples_written += len(batch)
		batch = collate_fn(batch)
		for k in batch.keys(): # TODO: make it more general
			if isinstance(batch[k], torch.Tensor):
				batch[k] = batch[k].to(torch.device("cuda"))
			elif isinstance(batch[k], dict):
				for kk in batch[k].keys():
					if isinstance(batch[k][kk], torch.Tensor):
						batch[k][kk] = batch[k][kk].to(torch.device("cuda"))
					else:
						raise ValueError(f"Unsupported type: {type(batch[k][kk])}")
			else:
				raise ValueError(f"Unsupported type: {type(batch[k])}")
		with torch.no_grad():
			output = model.forward(**batch)
		exporter.process_batch(output, metadata)
		sample_id += experiment_config.batch_size

	dataset.close()
	exporter.write_data_and_close()

	logger.info(f"Exported {samples_written} samples to {output_dir}")

if __name__ == "__main__":
	main()
