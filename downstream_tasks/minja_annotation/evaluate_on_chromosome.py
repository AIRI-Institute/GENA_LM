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
from torch.utils.data import DataLoader, Sampler
from collections.abc import Sized, Iterator
from simple_annotation_dataset import worker_init_fn, collate_fn

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='path to the experiment config') 
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')
parser.add_argument('--model_cpt', type=str, default=None, help='path to the model checkpoint')

class SequentialRegionalSampler(Sampler[int]):
	r"""Samples elements sequentially, always in the same order.
	If region is provided, samples are sampled sequentially from the region.

	Args:
		data_source (Dataset): dataset to sample from
	"""

	data_source: Sized

	def __init__(self, data_source: Sized, region: str) -> None:
		self.region = region
		self.data_source = data_source
		self.indices = []
		if region is not None:
			self.chrom = region.split(':')[0]
			self.region_start = int(region.split(':')[1].split('-')[0])
			self.region_end = int(region.split(':')[1].split('-')[1])
			for i in tqdm(range(len(self.data_source)), desc="Filtering samples by region"):
				chrom, start, end, strand = self.data_source._decode_item_id(i)
				if chrom == self.chrom and start >= self.region_start and end <= self.region_end:
					self.indices.append(i)
		else:
			self.indices = list(range(len(self.data_source)))

	def __iter__(self) -> Iterator[int]:
		return iter(self.indices)

	def __len__(self) -> int:
		return len(self.indices)

def identity_fn(x):
	return x

class bigWigExporter:
	strand2prefix = {
		"+": "",
		"-": "rev_comp_",
	}

	def __init__(self, output_dir: str, chroms: List[str], chrom_lengths: List[int], logger: logging.Logger = None, context_fraction: float = 0.1):
		self.output_dir = output_dir
		self.bigWigFiles = {}
		self.chroms = chroms
		self.chrom_lengths = chrom_lengths
		self.logger = logger if logger is not None else logging.getLogger(__name__)
		self.context_fraction = context_fraction
		self.last_chrom = None
		self.last_end = None
		self.last_strand = None
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
		for s, m in zip(outputs["predicts"], metadata):
			s_strand = m['strand']
			s_om = np.array(m['offset_mapping'])
			s_chrom = m['chrom']
			if s_chrom not in self.chrom_data.keys():
				self.chrom_data[s_chrom] = {}
			s_start = min(1, int(self.context_fraction * len(s)))
			s_end = max(len(s) - 1, int((1 - self.context_fraction) * len(s)))

			# --- NEW: infer number of classes and select class names (4-class or 6-class model) ---
			num_classes = s.shape[-1]  # NEW
			if num_classes == 4:  # NEW
				# old model: tss_+, tss_-, polya_+, polya_-  (2 logical classes x 2 strands)  # NEW
				class_names = ["tss", "polya"]  # NEW
			elif num_classes == 6:  # NEW
				# new model: tss_+, tss_-, polya_+, polya_-, intragenic_+, intragenic_-  # NEW
				class_names = ["tss", "polya", "intragenic"]  # NEW
			else:  # NEW
				raise ValueError(  # NEW
					f"Unsupported number of output classes: {num_classes}. "  # NEW
					"Expected 4 (tss, polya) or 6 (tss, polya, intragenic)."  # NEW
				)  # NEW

			# let's create a dictionary of values for each label
			class_number = 0
			values = {}
			# CHANGED: loop over dynamic class_names instead of hard-coded ["tss", "polya"]
			for class_name in class_names:  # CHANGED
				for strand in ["+", "-"]:
					label = f"{class_name}_{strand}" + self.strand2prefix[s_strand]
					if label not in self.chrom_data[s_chrom]:  # initialize the array with nan
						self.chrom_data[s_chrom][label] = np.empty(
							shape=(self.chrom_lengths[self.chroms.index(s_chrom)],)
						)
						self.chrom_data[s_chrom][label][:] = np.nan

					# outputs have shape of num_tokens x num_labels
					values[label] = self.sigmoid(s[s_start:s_end, class_number]).cpu().numpy()
					class_number += 1

			# let's process the sample, filling the chromosome data with values
			current_sample_coverage_start = None
			current_sample_coverage_end = None

			for token_index, s_om_i in enumerate(s_om[s_start:s_end]):
				start, end = s_om_i
				if start == end:
					continue

				if s_strand == "-":
					genome_start = m['end'] - end
					genome_end = m['end'] - start
				else:
					genome_start = m['start'] + start
					genome_end = m['start'] + end

				assert genome_end > genome_start, (
					f"genome_end is {genome_end} and genome_start is {genome_start}, "
					f"s_chrom is {s_chrom}, s_strand is {s_strand}, start is {start}, "
					f"end is {end}, om_i is {s_om_i}"
				)

				if current_sample_coverage_end is None:
					current_sample_coverage_end = genome_end
					current_sample_coverage_start = genome_start
				else:
					current_sample_coverage_end = max(current_sample_coverage_end, genome_end)
					current_sample_coverage_start = min(current_sample_coverage_start, genome_start)

				# we use tss_+ to check here, but it doesn't matter which one we use,
				# because all label arrays are filled simultaneously
				test_label = "tss_+" + self.strand2prefix[s_strand]

				if np.isnan(self.chrom_data[s_chrom][test_label][genome_start:genome_end]).any():
					for label, v in values.items():
						self.chrom_data[s_chrom][label][genome_start:genome_end] = v[token_index]
				else:
					for label, v in values.items():
						self.chrom_data[s_chrom][label][genome_start:genome_end] = (
							np.mean(self.chrom_data[s_chrom][label][genome_start:genome_end])
							+ v[token_index]
						) / 2

			# once we have processed the sample, let's ensure that it overlaps with the previous sample
			if (self.last_chrom is not None) and (self.last_chrom == s_chrom) and (self.last_strand == s_strand):
				# it's not first sample of the chromosome
				# let's ensure that samples overlap with the previous sample
				assert current_sample_coverage_start <= self.last_end, (
					f"current_sample_coverage_start is {current_sample_coverage_start} "
					f"and last_end is {self.last_end}, s_chrom is {s_chrom}, "
					f"s_strand is {s_strand}, start is {start}, end is {end}, "
					f"om_i is {s_om_i}"
				)

			self.last_chrom = s_chrom
			self.last_strand = s_strand
			self.last_end = current_sample_coverage_end


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

	assert not ("model_cpt" in experiment_config.keys() and (args.model_cpt is not None)), "model_cpt and args.model_cpt cannot both be provided"
	assert ("model_cpt" in experiment_config.keys()) or (args.model_cpt is not None), "model_cpt or args.model_cpt must be provided"

	model_cpt = experiment_config.model_cpt if "model_cpt" in experiment_config.keys() else args.model_cpt

	output_dir = os.path.join(
		os.path.dirname(model_cpt),
		"eval",
		os.path.basename(experiment_config.eval_dataset.path_to_fasta).split(".")[0],
		experiment_config.chromosome
		)
		
	logger.info(f"Output directory: {output_dir}")
	# copy experiment config to output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

	dataset_config = experiment_config.eval_dataset
	dataset_strandedness = dataset_config.get('strand', '+')
		
	dataset = instantiate(dataset_config)

	if dataset_strandedness == "random":
		assert len(dataset) % 2 == 0, "Dataset must have even number of samples for random strandedness"

	exporter = bigWigExporter(output_dir,
									chroms=list(dataset.chrom_info.keys()), 
									chrom_lengths=list(dataset.chrom_info.values()), 
									logger=logger)

	dataloader = instantiate(experiment_config.dataloader)

	model_config = experiment_config.model
	model = instantiate(model_config)
	state = load_file(model_cpt)
	model.load_state_dict(state, strict=True)
	model.to(torch.device("cuda"))
	model.eval()

	for batch in tqdm(dataloader, desc="Processing samples"):
		metadata = [b['metadata'] for b in batch]
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

	dataset.close()
	exporter.write_data_and_close()

	logger.info(f"Exported {len(dataloader.sampler)} samples to {output_dir}")

if __name__ == "__main__":
	main()
