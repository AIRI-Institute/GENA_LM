#!/usr/bin/env python
import os
from re import I
import numpy as np
import torch
from torch.utils.data import Dataset
import pysam
import gffutils
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
import logging

class GenomicAnnotationDataset(Dataset):
	"""
	PyTorch dataset for genomic annotation tasks.
	
	This dataset loads genomic sequences from a FASTA file, tokenizes them using
	a BPE tokenizer, and extracts annotation targets (TSS and polyA sites) from
	a GFF database.
	"""
	
	def __init__(
		self,
		model_input_length_token: int,
		av_token_len: float,
		path_to_gff_db: str,
		path_to_fasta: str,
		tokenizer,
		chunk_overlap: float = 0,
		max_genomic_chunk_ratio: float = 1.5,
		TSS_prob_width: float = 10, # in base pairs
		polyA_prob_width: float = 10, # in base pairs
		exclude_chromosomes: List[str] | str | None = None, # chromosomes to exclude
		include_chromosomes: List[str] | str | None = None, # chromosomes to include
		primary_transcript_types: List[str] | None = None, # transcript types to require for a transcript to be included as primary
		primary_tags: List[str] | None = ["GENCODE_Primary", "MANE_Select"], # tags to require for a transcript to be included as primary
		logger: logging.Logger = None,
	):
		"""
		Initialize the GenomicAnnotationDataset.
		
		Args:
			model_input_length_token: Number of tokens for model input
			av_token_len: Average token length in base pairs
			path_to_gff_db: Path to GFF database file (.db)
			path_to_fasta: Path to FASTA file
			tokenizer: AutoTokenizer from transformers (BPE tokenizer)
			chunk_overlap: Overlap between chunks as a fraction of the chunk length
			max_genomic_chunk_ratio: Ratio to increase genomic chunk size for tokenization
			TSS_prob_width: Width of the TSS probability distribution in base pairs, will be used to compute TSS probability as exp(-dist/TSS_prob_width)
			polyA_prob_width: Width of the polyA probability distribution in base pairs, will be used to compute polyA probability as exp(-dist/polyA_prob_width)
			exclude_chromosomes: Chromosomes to exclude
			include_chromosomes: Chromosomes to include
			primary_transcript_types: Transcript types to require for a transcript to be included as primary
			primary_tags: Tags to require for a transcript to be included as primary
		"""
		self.model_input_length_token = model_input_length_token
		self.av_token_len = av_token_len
		self.path_to_gff_db = path_to_gff_db
		self.path_to_fasta = path_to_fasta
		self.tokenizer = tokenizer
		self.chunk_overlap = chunk_overlap
		assert chunk_overlap >= 0 and chunk_overlap < 1, "Chunk overlap must be between 0 and 1"
		self.max_genomic_chunk_ratio = max_genomic_chunk_ratio
		self.TSS_prob_width = TSS_prob_width
		self.polyA_prob_width = polyA_prob_width
		self.logger = logger if logger is not None else logging.getLogger(__name__)

		if exclude_chromosomes is not None and isinstance(exclude_chromosomes, str):
			exclude_chromosomes = exclude_chromosomes.split(",")
		if include_chromosomes is not None and isinstance(include_chromosomes, str):
			include_chromosomes = include_chromosomes.split(",")

		self.exclude_chromosomes = exclude_chromosomes
		self.include_chromosomes = include_chromosomes

		if primary_transcript_types is not None and isinstance(primary_transcript_types, str):
			primary_transcript_types = primary_transcript_types.split(",")
		if primary_tags is not None and isinstance(primary_tags, str):
			primary_tags = primary_tags.split(",")
		self.transcript_type_filter = primary_transcript_types
		self.tags_filter = primary_tags
		
		assert self.exclude_chromosomes is None or self.include_chromosomes is None, "Only one of exclude_chromosomes and include_chromosomes can be provided"
				
		# Initialize FASTA and GFF database
		self._init_fasta()
		self._init_gff_db()
		
		# Compute chunk parameters
		self.chunk_length = int(model_input_length_token * av_token_len)
		self.genomic_chunk_length = int(self.chunk_length * max_genomic_chunk_ratio)
		
		# Get chromosome information and compute chunks
		self._compute_chunks()

		self.logger.info(f"Initialized dataset with {self.total_chunks} chunks")
		
	def _init_fasta(self):
		"""Initialize FASTA file reader using pysam."""
		if not os.path.exists(self.path_to_fasta):
			raise FileNotFoundError(f"FASTA file not found: {self.path_to_fasta}")
		
		self.fasta = pysam.FastaFile(self.path_to_fasta)
		
		# Get chromosome information
		self.chrom_info = {}
		for i, chrom in enumerate(self.fasta.references):
			if self.exclude_chromosomes is not None and chrom in self.exclude_chromosomes:
				continue
			if self.include_chromosomes is not None and chrom not in self.include_chromosomes:
				continue
			self.chrom_info[chrom] = self.fasta.get_reference_length(chrom)
		self.logger.info(f"Initialized FASTA file with {len(self.chrom_info)} out of {len(self.fasta.references)} chromosomes")
			
	def _init_gff_db(self):
		"""Initialize GFF database."""
		if not os.path.exists(self.path_to_gff_db):
			raise FileNotFoundError(f"GFF database not found: {self.path_to_gff_db}")
		
		self.gff_db = gffutils.FeatureDB(self.path_to_gff_db)
		
	def _compute_chunks(self):
		"""Compute chunk information for all chromosomes."""
		self.chunks = []
		total_chunks = 0

		shift = self.chunk_length - int(self.chunk_overlap * self.chunk_length)
		shift = max(shift, 1)
		step_size = shift


		for chrom, length in self.chrom_info.items():
			# Skip very small chromosomes
			if length < self.chunk_length:
				continue
				
			# Compute number of chunks for this chromosome
			num_chunks = max(1, (length - shift) // step_size)
			
			for chunk_idx in range(num_chunks):
				start_pos = chunk_idx * step_size
				end_pos = min(start_pos + self.chunk_length, length)
				
				self.chunks.append({
					'chrom': chrom,
					'start': start_pos,
					'end': end_pos,
					'chunk_id': total_chunks
				})
				total_chunks += 1
				
		self.total_chunks = total_chunks
		
	def __len__(self) -> int:
		"""Return the number of chunks in the dataset."""
		return self.total_chunks
		
	def _decode_item_id(self, item_id: int) -> Tuple[str, int, int]:
		"""
		Decode item_id to chromosome and position.
		
		Args:
			item_id: Index of the item
			
		Returns:
			Tuple of (chromosome, start_pos, end_pos)
		"""
		if item_id >= self.total_chunks:
			raise IndexError(f"Item ID {item_id} out of range (max: {self.total_chunks - 1})")
			
		chunk_info = self.chunks[item_id]
		return chunk_info['chrom'], chunk_info['start'], chunk_info['end']
		
	def _get_genomic_sequence(self, chrom: str, start: int, end: int) -> str:
		"""
		Extract genomic sequence from FASTA file.
		
		Args:
			chrom: Chromosome name
			start: Start position (0-based)
			end: End position (0-based, exclusive)
			
		Returns:
			Genomic sequence as string
		"""
		# Ensure we don't go beyond chromosome boundaries
		chrom_length = self.chrom_info[chrom]
		start = max(0, start)
		end = min(end, chrom_length)
		assert end > start
		
		# Get extended sequence for tokenization
		extended_end = min(end + int(self.genomic_chunk_length - self.chunk_length), chrom_length)
		assert extended_end >= end >= start
		sequence = self.fasta.fetch(chrom, start, extended_end)
		return sequence.upper()
		
	def _tokenize_sequence(self, sequence: str) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
		"""
		Tokenize genomic sequence and return offset mapping.
		
		Args:
			sequence: Genomic sequence string
			
		Returns:
			Tuple of (token_ids, offset_mapping)
		"""
		# Tokenize with offset mapping
		encoding = self.tokenizer(
			sequence,
			return_tensors='pt',
			return_offsets_mapping=True,
			add_special_tokens=True,
			truncation=True,
			max_length=self.model_input_length_token,
			padding='max_length'
		)
		
		token_ids = encoding['input_ids'].squeeze(0)
		offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()
		for idx, token_id in enumerate(token_ids):
			if token_id.item() == 5: # gap token
				offset_mapping[idx][0] = offset_mapping[idx-1][1]
		
		# sanity check. TODO: remove at some point
		for tok,om in zip(token_ids, offset_mapping):
			if tok.item() != 5 and om[0] < om[1]:
				token_chars = self.tokenizer.decode(tok.item())
				assert token_chars[0] == sequence[om[0]]
		return token_ids, offset_mapping

	def _is_main_transcript(self, feature: gffutils.Feature) -> bool:
		"""
		Check if a feature is the main transcript.
		"""
		# tag example:
		# tag "basic,Ensembl_canonical,GENCODE_Primary,MANE_Select,appris_principal_1,CCDS";
		tags = feature.attributes.get("tag", "unknown")
		transcript_types = feature.attributes.get("transcript_type", ["unknown"])
		filter1 = any(tag in tags for tag in self.tags_filter)
	
		if self.transcript_type_filter is None:
			filter2 = True
		else:
			filter2 = any(transcript_type in transcript_types for transcript_type in self.transcript_type_filter)
		return filter1 and filter2
		
	def _extract_annotation_targets(
		self, 
		chrom: str, 
		start: int, 
		end: int, 
		offset_mapping: List[Tuple[int, int]]
	) -> Dict[str, torch.Tensor]:
		"""
		Extract annotation targets (TSS and polyA sites) from GFF database.
		
		Args:
			chrom: Chromosome name
			start: Chunk start position
			end: Chunk end position
			offset_mapping: Token offset mapping
			
		Returns:
			Dictionary with annotation targets
		"""
		# Initialize target arrays
		len_bp = end - start
		assert len_bp > 0, f"len_bp is {len_bp} for {chrom}:{start}-{end}"

		targets = {
			'intragenic_regions': torch.zeros(len_bp, dtype=torch.long),
		}
		for strand in ["+", "-"]:
			for tss_type in ["primary", "uncertain"]:
				for polya_type in ["primary", "uncertain"]:
					targets[f"{tss_type}_tss_{strand}"] = torch.zeros(len_bp, dtype=torch.long)
					targets[f"{polya_type}_polya_{strand}"] = torch.zeros(len_bp, dtype=torch.long)
					
		try:
			# Query GFF database for features in this region
			features = list(self.gff_db.region(region=(chrom, start, end), featuretype="transcript"))
			
			for feature in features:
				feature_start = feature.start
				feature_end = feature.end
				assert feature.strand in ["+", "-"]
				TSS = feature_start if feature.strand == "+" else feature_end
				polyA = feature_end if feature.strand == "+" else feature_start

				TSS_probability = torch.exp(-torch.abs(torch.arange(start, end, dtype=torch.float32) - TSS)/self.TSS_prob_width)
				polyA_probability = torch.exp(-torch.abs(torch.arange(start, end, dtype=torch.float32) - polyA)/self.polyA_prob_width)
				# print ("TSS_probability.max(), polyA_probability.max()", TSS_probability.max(), polyA_probability.max())
				# print ("TSS_probability[TSS_probability<1].max(), polyA_probability[polyA_probability<1].max()", TSS_probability[TSS_probability<1].max(), polyA_probability[polyA_probability<1].max())

				# if feature.attributes["transcript_id"] == "ENST00000377443.7":
				# 	print ("I am here")
				# 	raise Exception("stop here")
				# print (feature.attributes.get("gene_type", ["unknown"]), "protein_coding" in feature.attributes.get("gene_type", ["unknown"]))
				if "protein_coding" in feature.attributes.get("transcript_type", ["unknown"]):
					# print (feature.attributes["transcript_id"])

					CDSs = list(self.gff_db.children(id=feature, featuretype="CDS", order_by="start", completely_within=True))
					assert len(CDSs) > 0, f"len(CDSs) is {len(CDSs)} for {chrom}:{start}-{end} (transcript_id: {feature.attributes['transcript_id']})"
					CDS_start = CDSs[0].start
					CDS_end = CDSs[-1].end

					relative_CDS_start = CDS_start - start
					relative_CDS_end = CDS_end - start

					if relative_CDS_end < 0 or relative_CDS_start >= len_bp:
						relative_CDS_start = 0
						relative_CDS_end = 0
					else:
						relative_CDS_start = max(relative_CDS_start, 0)
						relative_CDS_end = min(relative_CDS_end, len_bp)
	
					assert relative_CDS_start <= relative_CDS_end, f"relative_CDS_start is {relative_CDS_start} and relative_CDS_end is {relative_CDS_end} for {chrom}:{start}-{end} and transcript_id: {feature.attributes['transcript_id']}"
					assert relative_CDS_start >= 0 and relative_CDS_end <= len_bp, f"relative_CDS_start is {relative_CDS_start} and relative_CDS_end is {relative_CDS_end} for {chrom}:{start}-{end} and transcript_id: {feature.attributes['transcript_id']}"
	
					# print (relative_CDS_start, relative_CDS_end)

					# we should not start AFTER the start codon for sure
					TSS_probability[relative_CDS_start:relative_CDS_end] = 0
					
					# we should not stop BEFORE the stop codon for sure
					polyA_probability[relative_CDS_start:relative_CDS_end] = 0

				# print ("after CDS")
				# print ("TSS_probability.max(), polyA_probability.max()", TSS_probability.max(), polyA_probability.max())
				# print ("TSS_probability[TSS_probability<1].max(), polyA_probability[polyA_probability<1].max()", TSS_probability[TSS_probability<1].max(), polyA_probability[polyA_probability<1].max())
				# print (sorted(TSS_probability.tolist(), reverse=True)[:40])
				# print (sorted(polyA_probability.tolist(), reverse=True)[:40])				

				transcript_type = "primary" if self._is_main_transcript(feature) else "uncertain"
				# print ("transcript_type", transcript_type)

				# we use max of the current probability and pre-existing probability (which is computed based on other transcripts)
				targets[f"{transcript_type}_tss_{feature.strand}"] = torch.maximum(
					TSS_probability,
					targets[f"{transcript_type}_tss_{feature.strand}"]
				)
				targets[f"{transcript_type}_polya_{feature.strand}"] = torch.maximum(
					polyA_probability,
					targets[f"{transcript_type}_polya_{feature.strand}"]
				)

				feature_relative_start = max(feature_start - start, 0)
				feature_relative_end = min(feature_end - start, len_bp)
				targets[f"intragenic_regions"][feature_relative_start:feature_relative_end] = 1
		except Exception as e:
			# Handle cases where chromosome is not in GFF database
			print(f"Warning: Could not query GFF database for {chrom}:{start}-{end}: {e}")
			raise e

		# print ("after Processed all transcripts")
		# for target_type in targets.keys():
		# 	print (f"----{target_type}----")
		# 	print (f"{target_type}.max()", targets[target_type].max())
		# 	print (f"{target_type}[{target_type}<1].max()", targets[target_type][targets[target_type]<1].max())
		# 	print (sorted(targets[target_type].tolist(), reverse=True)[:40])

		# map basepair resolution targets to token resolution targets
		targets_token = {}
		for target_type in targets.keys():
			targets_token[target_type] = torch.zeros(len(offset_mapping), dtype=torch.float32)

			# if target_type.find("primary_polya_+") != -1:
			# 	print (target_type, target_type.find("primary_polya_+"))
			# 	print ("ids of max values in basepair resolution:")
			# 	N = 30
			# 	ids = torch.argsort(targets[target_type], descending=True).numpy()[:N]
			# 	print (ids)
			# 	print (targets[target_type][ids])
			# 	raise Exception("stop here")
			for tok_id in range(len(offset_mapping)):
				if offset_mapping[tok_id][0] == offset_mapping[tok_id][1]: # special token
					targets_token[target_type][tok_id] = -100
				else:
					targets_token[target_type][tok_id] = targets[target_type][offset_mapping[tok_id][0]:offset_mapping[tok_id][1]].max()

		# print ("after mapped to token resolution")
		# for target_type in targets.keys():
		# 	print (f"----{target_type}----")
		# 	print (f"{target_type}.max()", targets_token[target_type].max())
		# 	print (f"{target_type}[{target_type}<1].max()", targets_token[target_type][targets_token[target_type]<1].max())
		# 	print (sorted(targets_token[target_type].tolist(), reverse=True)[:40])

		return targets_token
		
	def __getitem__(self, item_id: int) -> Dict[str, torch.Tensor]:
		"""
		Get a genomic chunk with its annotations.
		
		Args:
			item_id: Index of the chunk
			
		Returns:
			Dictionary containing:
			- input_ids: Tokenized sequence
			- attention_mask: Attention mask for padding
			- targets: Dictionary with annotation targets
			- metadata: Chunk metadata
		"""
		# Decode chromosome and position
		chrom, start, end = self._decode_item_id(item_id)
		
		# Get genomic sequence
		sequence = self._get_genomic_sequence(chrom, start, end)
		
		# Tokenize sequence
		token_ids, offset_mapping = self._tokenize_sequence(sequence)
		end = start + max([i[1] for i in offset_mapping]) # for original end we used some overhead, now shrink it to the last token
		
		# Extract annotation targets
		targets_token = self._extract_annotation_targets(chrom, start, end, offset_mapping)
		
		# Create attention mask
		attention_mask = torch.ones(len(token_ids), dtype=torch.long)
		# set attention mask for PAD tokens to 0
		attention_mask[token_ids == self.tokenizer.pad_token_id] = 0
		
		return {
			'input_ids': token_ids,
			'attention_mask': attention_mask,
			'targets': targets_token,
			'metadata': {
				'chrom': chrom,
				'start': start,
				'end': end,
				'chunk_id': item_id,
				'sequence_length': len(sequence),
				'offset_mapping': offset_mapping,
				'index': item_id,
			}
		}
		
	def close(self):
		"""Close file handles."""
		if hasattr(self, 'fasta'):
			self.fasta.close()
			
	def __del__(self):
		"""Cleanup when object is destroyed."""
		self.close()

def toIGV(sample: Dict, tokenizer: AutoTokenizer, IGV_files_prefix: str, file_mode: str = "w"):
	"""
	Convert sample to IGV format. Saves each target as bedGraph file and sample coordinates as bed file.
	Args:
		sample: Sample from __getitem__
		tokenizer: Tokenizer
		IGV_files_prefix: Prefix for IGV files

	"""
	metadata = sample['metadata']
	chrom = metadata['chrom']
	start = metadata['start']
	end = metadata['end']
	region_name = f"{chrom}:{start}-{end}:{metadata['index']}"

	# Save coordinates as BED file
	bed_path = f"{IGV_files_prefix}_coords.bed"
	with open(bed_path, file_mode) as bed_file:
		bed_file.write(f"{chrom}\t{start}\t{end}\t{region_name}\n")

	# Save each target as bedGraph file
	targets = sample['targets']
	offset_mapping = sample['metadata']['offset_mapping']
	input_ids = sample['input_ids']

	for target_name, target_tensor in targets.items():
		bedgraph_path = f"{IGV_files_prefix}_{target_name}.bedGraph"
		if not os.path.exists(bedgraph_path):
			# write header
			with open(bedgraph_path, file_mode) as bg_file:
				bg_file.write(f"track type=bedGraph name=\"{target_name}\" description=\"{target_name}\" visibility=full autoScale=off viewLimits=0:1\n")

		with open(bedgraph_path, file_mode) as bg_file:
			# Assume target_tensor is 1D and corresponds to tokens along the region
			# Map each token to a genomic position using offset mapping if available
			# Here, we assume uniform distribution of tokens across the region
			for i in range(len(offset_mapping)):
				token_start = offset_mapping[i][0]
				token_end = offset_mapping[i][1]
				value = float(target_tensor[i].item())
				bg_file.write(f"{chrom}\t{start+token_start}\t{start+token_end}\t{value}\n")
	bedgraph_path = f"{IGV_files_prefix}_sequence.bed"
	with open(bedgraph_path, file_mode) as bg_file:
		for i in range(len(input_ids)):
			token_start = offset_mapping[i][0]
			token_end = offset_mapping[i][1]
			value = '"' + tokenizer.decode(input_ids[i].item()) + '"'
			bg_file.write(f"{chrom}\t{start+token_start}\t{start+token_end}\t{value}\n")

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
	"""
	Collate function for DataLoader.
	
	Args:
		batch: List of samples from __getitem__
		
	Returns:
		Batched data
	"""
	batched = {
		'input_ids': torch.stack([item['input_ids'] for item in batch]),
		'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
		'targets': {},
	}
	
	# Batch targets
	target_keys = batch[0]['targets'].keys()
	for key in target_keys:
		batched['targets'][key] = torch.stack([item['targets'][key] for item in batch])
		
	return batched


# Example usage
if __name__ == "__main__":
	from transformers import AutoTokenizer
	
	# Example parameters
	model_input_length_token = 512
	av_token_len = 9.0
	path_to_gff_db = "../../data/annotation/gencode.v48.annotation.gtf.db"
	path_to_fasta = "/mnt/nfs_dna/minja/hg38.fa"
	
	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")
	
	# Create dataset
	dataset = GenomicAnnotationDataset(
		model_input_length_token=model_input_length_token,
		av_token_len=av_token_len,
		path_to_gff_db=path_to_gff_db,
		path_to_fasta=path_to_fasta,
		tokenizer=tokenizer,
		primary_transcript_types=["protein_coding"]
	)
	
	print(f"Dataset length: {len(dataset)}")
	
	# Get a sample
	index = 1941
	sample = dataset[index]
	print(f"Sample keys: {sample.keys()}")
	print(f"Input shape: {sample['input_ids'].shape}")
	print(f"Target keys: {sample['targets'].keys()}")
	m = {k:v for k, v in sample['metadata'].items() if k != 'offset_mapping'}
	print(f"Metadata: {m}")
	toIGV(sample, tokenizer, f"../../data/annotation/tmp/test_{index}", file_mode="w")

	import datetime
	start_time = datetime.datetime.now()
	index_range = (1900,2100)
	samples = []
	for index in range(index_range[0], index_range[1]):
		sample = dataset[index]
		samples.append(sample)
	end_time = datetime.datetime.now()
	print (f"Time taken for {index_range[0]}-{index_range[1]}: {end_time - start_time}")
	print (f"Time taken per sample: {(end_time - start_time) / (index_range[1] - index_range[0])}")

	for sample in samples:
		toIGV(sample, tokenizer, f"../../data/annotation/tmp/test_{index_range[0]}-{index_range[1]}", file_mode="a")