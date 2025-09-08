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

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='path to the experiment config') 
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')
parser.add_argument('--output_dir', type=str, help='output directory')

class bigWigExporter:
	def __init__(self, output_dir: str, chroms: List[str], chrom_lengths: List[int]):
		self.output_dir = output_dir
		self.bigWigFiles = {}
		self.chroms = chroms
		self.chrom_lengths = chrom_lengths
		self.last_chrom = None
		self.last_end = None
		self.last_value = {}
		self.sigmoid = torch.nn.Sigmoid()
		self.bedFile = open(os.path.join(self.output_dir, "sample_coords.bed"), "w")
	
	def open_bw(self, output_name: str):
		self.bigWigFiles[output_name] = bw.open(os.path.join(self.output_dir, output_name+".bw"), "w")
		for chrom, length in zip(self.chroms, self.chrom_lengths):
			self.bigWigFiles[output_name].addHeader([(chrom, length)])
	
	def add_values(self, chrom: str, starts: List[int], ends: List[int], values: List[float], output_name: str):
		# cast starts and ends to int
		filtered_starts = []
		filtered_ends = []
		filtered_values = []
		for i in range(len(starts)):
			if starts[i] == ends[i]:
				continue
			else:
				filtered_starts.append(int(starts[i]))
				filtered_ends.append(int(ends[i]))
				filtered_values.append(values[i])
		if len(filtered_starts) > 0:
			self.last_end = max(filtered_ends)
			self.last_chrom = chrom
			chrom = [chrom]*len(filtered_starts)
			# if len(filtered_starts) == 1:
			# 	filtered_starts = [filtered_starts[0]]
			# 	filtered_ends = [filtered_ends[0]]
			# 	filtered_values = [filtered_values[0]]

			self.bigWigFiles[output_name].addEntries(chrom, filtered_starts, filtered_ends, values=filtered_values)
		else:
			print (chrom, starts[0], ends[-1], "no values")

	def process_sample(self, chrom, chrom_start, offset_mappings: List[Tuple[int, int]], values: Dict[str, List[float]]):
		starts = [om[0] + chrom_start for om in offset_mappings]
		ends = [om[1] + chrom_start for om in offset_mappings]
		
		for k,v in values.items():
			assert len(starts) == len(ends) == len(v), f"len(starts) = {len(starts)}, len(ends) = {len(ends)}, len(v) = {len(v)}"
			self.add_values(chrom, starts, ends, v, k)
		self.bedFile.write(f"{chrom}\t{starts[0]}\t{ends[-1]}\t{chrom}\n")

	def process_batch(self, outputs, metadata: List[Dict]):
		for s,m in zip(outputs["predicts"], metadata):
			s_om = m['offset_mapping']
			if self.last_chrom is None or self.last_chrom != m['chrom']:
				s_start = 1
			else:
				# find start: we start from the first token after the last end
				s_start = 0
				for ind,val in enumerate(s_om[:-1]):
					if m["start"] + val[1] > self.last_end and val[0] != val[1]:
						s_start = ind
						break
				if s_start == 0:
					print (m['chrom'], m['start'], m['end'], "no starts")
					return

			# outputs have shape of num_tokens x num_labels
			values = {}
			class_number = 0
			for class_name in ["tss", "polya"]:
				for strand in ["+", "-"]:
					label = f"{class_name}_{strand}"
					values[label] = self.sigmoid(s[s_start:,class_number]).cpu().numpy().tolist()
					# print (label, max(values[label][:-1]))
					if self.last_chrom is not None and self.last_chrom != m['chrom']:
						values[label][0] = (self.last_value[label] + values[label][0])/2.
	
					self.last_value[label] = values[label][-1]
					if label not in self.bigWigFiles.keys():
						self.open_bw(label)
					class_number += 1
					assert len(s_om[s_start:]) == len(values[label]), f"len(s_om[s_start:]) = {len(s_om[s_start:])}, len(values[label]) = {len(values[label])}"
			
			self.process_sample(m['chrom'], m["start"], s_om[s_start:], values)

	def close(self):
		for k in self.bigWigFiles.keys():
			self.bigWigFiles[k].close()
		self.bedFile.close()

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

	exporter = bigWigExporter(output_dir, dataset.chrom_info.keys(), list(dataset.chrom_info.values()))

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
	exporter.close()

	logger.info(f"Exported {samples_written} samples to {output_dir}")

if __name__ == "__main__":
	main()
