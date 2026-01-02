import os
import numpy as np
import pysam
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
from torch import nn
from transformers.utils.logging import warning_once
from scipy.stats import entropy
import requests
from tqdm import tqdm
import argparse
import datetime
import configparser

from pathlib import Path
import sys

def build_modergena_model(checkpoint_filepath, modernbert_distr_path):
	from omegaconf import DictConfig, OmegaConf
	from omegaconf import OmegaConf as om
	print (modernbert_distr_path)
	sys.path.append(modernbert_distr_path)
	from ModernBERT.src import flex_bert as flex_bert_module
	from ModernBERT.src import hf_bert as hf_bert_module
	from ModernBERT.src import mosaic_bert as mosaic_bert_module
	from ModernBERT.src.bert_layers.model import init_mlm_model_from_pretrained
	from ModernBERT.src.bert_layers.configuration_bert import FlexBertConfig
	from composer.utils.checkpoint import _ensure_valid_checkpoint

	def build_model(cfg: DictConfig):
		if cfg.name == "hf_bert":
			return hf_bert_module.create_hf_bert_mlm(
				pretrained_model_name=cfg.pretrained_model_name,
				use_pretrained=cfg.get("use_pretrained", None),
				model_config=cfg.get("model_config", None),
				tokenizer_name=cfg.get("tokenizer_name", None),
				gradient_checkpointing=cfg.get("gradient_checkpointing", None),
			)
		elif cfg.name == "mosaic_bert":
			return mosaic_bert_module.create_mosaic_bert_mlm(
				pretrained_model_name=cfg.pretrained_model_name,
				pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
				model_config=cfg.get("model_config", None),
				tokenizer_name=cfg.get("tokenizer_name", None),
				gradient_checkpointing=cfg.get("gradient_checkpointing", None),
			)
		elif cfg.name == "flex_bert":
			return flex_bert_module.create_flex_bert_mlm(
				pretrained_model_name=cfg.pretrained_model_name,
				pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
				model_config=cfg.get("model_config", None),
				tokenizer_name=cfg.get("tokenizer_name", None),
				gradient_checkpointing=cfg.get("gradient_checkpointing", None),
				recompute_metric_loss=cfg.get("recompute_metric_loss", False),
				disable_train_metrics=cfg.get("disable_train_metrics", False),
			)
		else:
			raise ValueError(f"Not sure how to build model with name={cfg.name}")

	cpt_dir = os.path.dirname(checkpoint_filepath)
	cfg_path = os.path.join(cpt_dir, "cfg.yaml")
	yaml_cfg = om.load(cfg_path)
	model = build_model(yaml_cfg.model)
	print (f"Loading checkpoint from {checkpoint_filepath}")
	checkpoint_filepath = Path(checkpoint_filepath)
	assert checkpoint_filepath.exists(), f"Checkpoint {checkpoint_filepath} does not exist"

	# added weights_only=False to suppress this error: 
	# In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. 
	# Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code 
	# execution. Do it only if you got the file from a trusted source.
	state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu", weights_only=False) 
	state_dict = state.get("state", {})
	model_state = state_dict.get("model", {})
	assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"
	model.load_state_dict(model_state)
	return model

def parse_args():
	parser = argparse.ArgumentParser()
	valid_model_families = ['GENA', 'NTv2', 
						'ModernGENA']

	parser.add_argument("--name", type=str, 
						help="label of the experiment", default=None)
	parser.add_argument("--model_family", type=str, 
						choices=valid_model_families,
						default=None, help="Family of models to use for analysis")
	parser.add_argument("--cpt_path", type=str, 
						help="path to the checkpoint", default=None)
	parser.add_argument("--input_len_tokens", type=int, 
						help="maximum input length in number of tokens", default=None)
	parser.add_argument("--tokenizer_path", type=str, 
						help="path to the tokenizer", default=None)
	parser.add_argument("--genome_path", type=str, 
						help="Path to the gapless genome. Note that it should have specific structure. To prepare genome use [#TODO: add script to prepare genome]")
	parser.add_argument("--out_dir", type=str, default="data/")
	parser.add_argument("--chrm", type=str, help="Chromosome to process (e.g., 'chr1')")
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--limit_bp", type=int, help="Limit processing to this number of base pairs", default=None)
	parser.add_argument("--modernbert_distr_path", type=str, help="Path to ModernBERT distribution", 
						default=os.path.expanduser("~/DNALM/")
						)
	parser.add_argument("--seq_chunk_len",type=int, default=50_000, help="Chunk size used to split original fasta record for tokenization")
	parser.add_argument("--masking_fraction",type=float,
						help="Fraction of tokens to mask. If >=1, corresponds to number of tokens to mask. If <1, corresponds to fraction of tokens to mask.",
						default=0.3)
	parser.add_argument("--config", type=str, help="Path to configuration file", default=None)
	
	# Track which arguments were actually provided on command line (before parsing)
	cmdline_args_provided = set()
	arg_names = ['name', 'cpt_path', 'tokenizer_path', 'genome_path', 'out_dir', 'chrm', 'batch_size', 
				 'limit_bp', 'modernbert_distr_path', 	'seq_chunk_len', 'masking_fraction', 'config', 
				 'input_len_tokens', 'model_family']
	
	i = 1  # Skip script name
	while i < len(sys.argv):
		arg = sys.argv[i]
		if arg.startswith('--'):
			arg_name = arg[2:]  # Remove '--'
			if arg_name == 'config':
				# Skip --config and its value (if present)
				i += 2 if i + 1 < len(sys.argv) else 1
				continue
			if arg_name in arg_names:
				cmdline_args_provided.add(arg_name)
		i += 1
	
	# Parse arguments
	args = parser.parse_args()
	
	# If config file is provided, read from it
	if args.config:
		if not os.path.exists(args.config):
			raise FileNotFoundError(f"Configuration file not found: {args.config}")
		
		config = configparser.ConfigParser()
		config.read(args.config)
		
		# Read values from config file
		if 'DEFAULT' in config:
			section = config['DEFAULT']
		elif 'config' in config:
			section = config['config']
		else:
			# Use first section if DEFAULT or 'config' not found
			section = config[config.sections()[0]] if config.sections() else configparser.SectionProxy(config, {})
		
		# Check for conflicts: same argument in both config and command-line
		config_args_provided = set()
		conflicts = []
		
		# Map config file keys to argument names
		config_to_arg_map = {
			'name': 'name',
			'model_family': 'model_family',
			'cpt_path': 'cpt_path',
			'input_len_tokens': 'input_len_tokens',
			'tokenizer_path': 'tokenizer_path',
			'genome_path': 'genome_path',
			'out_dir': 'out_dir',
			'chrm': 'chrm',
			'batch_size': 'batch_size',
			'limit_bp': 'limit_bp',
			'modernbert_distr_path': 'modernbert_distr_path',
			'seq_chunk_len': 'seq_chunk_len',
			'masking_fraction': 'masking_fraction',
		}
		
		for config_key, arg_name in config_to_arg_map.items():
			if section.get(config_key) is not None:
				config_args_provided.add(arg_name)
				if arg_name in cmdline_args_provided:
					conflicts.append(arg_name)
		
		if conflicts:
			raise ValueError(f"Argument(s) provided both in config file and command-line: {', '.join(conflicts)}")
		
		# Merge: use command-line values if provided, otherwise use config values
		# For values not in config, keep command-line/default values
		if 'name' in section and 'name' not in cmdline_args_provided:
			args.name = section.get('name')
		if 'model_family' in section and 'model_family' not in cmdline_args_provided:
			args.model_family = section.get('model_family')
		if 'cpt_path' in section and 'cpt_path' not in cmdline_args_provided:
			args.cpt_path = section.get('cpt_path')
		if 'input_len_tokens' in section and 'input_len_tokens' not in cmdline_args_provided:
			args.input_len_tokens = section.getint('input_len_tokens')
		if 'tokenizer_path' in section and 'tokenizer_path' not in cmdline_args_provided:
			args.tokenizer_path = section.get('tokenizer_path')
		if 'genome_path' in section and 'genome_path' not in cmdline_args_provided:
			args.genome_path = section.get('genome_path')
		if 'out_dir' in section and 'out_dir' not in cmdline_args_provided:
			args.out_dir = section.get('out_dir', 'data/')
		if 'chrm' in section and 'chrm' not in cmdline_args_provided:
			args.chrm = section.get('chrm')
		if 'batch_size' in section and 'batch_size' not in cmdline_args_provided:
			args.batch_size = section.getint('batch_size', 16)
		if 'model' in section and 'model' not in cmdline_args_provided:
			args.model = section.get('model', 'gena-lm')
		if 'limit_bp' in section and 'limit_bp' not in cmdline_args_provided:
			limit_bp_str = section.get('limit_bp', 'None')
			args.limit_bp = None if limit_bp_str.lower() == 'none' else section.getint('limit_bp')
		if 'modernbert_distr_path' in section and 'modernbert_distr_path' not in cmdline_args_provided:
			args.modernbert_distr_path = section.get('modernbert_distr_path', os.path.expanduser("~/DNALM/"))
		
		# Validate required arguments
		requered_args = ['name', 'model_family', 'cpt_path', 'input_len_tokens', 'tokenizer_path', 'genome_path']
		for arg in requered_args:
			if not getattr(args, arg):
				raise ValueError(f"{arg} must be specified (either in config file or command-line)")
		
	return args

# Load model and tokenizer
def load_model_and_tokenizer(cpt_path, tokenizer_path, model_family, modernbert_distr_path=None):
	if model_family == 'GENA':
		tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
		model = AutoModel.from_pretrained(cpt_path, trust_remote_code=True)
	elif model_family == 'NTv2':
		tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
		model = AutoModelForMaskedLM.from_pretrained(cpt_path, trust_remote_code=True)
	elif model_family == 'ModernGENA':
		tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
		model = build_modergena_model(cpt_path, modernbert_distr_path)
	else:
		raise ValueError(f"Invalid model family: {model_family}")
	
	if torch.cuda.is_available():
		model = model.cuda()
	model.eval()
	return model, tokenizer

# Calculate metrics on model predictions
def calculate_token_metrics(model, tokenizer, inputs, ground_truth):
	# Tokenize masked sequence
	if torch.cuda.is_available():
		inputs = {k: v.cuda() for k, v in inputs.items()}
	
	# Get model predictions
	with torch.no_grad():
		if args.model_family == "ModernGENA":
			outputs = model(inputs)
			logits = outputs.logits.reshape(inputs['input_ids'].shape[0],
											inputs['input_ids'].shape[1],
											-1)
		else:
			outputs = model(**inputs)
			logits = outputs.logits

	# print (f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
	# print (f"outputs: {outputs}")
	# print (f"logits.shape: {logits.shape}")

	# Find positions of [MASK] tokens for each sequence in batch
	mask_positions = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()
	batch_indices = mask_positions[:, 0]
	token_positions = mask_positions[:, 1]
	# print (f"batch_indices: {batch_indices}")
	# print (f"token_positions: {token_positions}")

	# print (f"logits[batch_indices, token_positions].shape: {logits[batch_indices, token_positions].shape}")

	# Get probabilities for masked positions
	probs = torch.softmax(logits[batch_indices, token_positions], dim=1).cpu().numpy()
	# print (f"probs.shape: {probs.shape}")
	
	# get highest probabilities
	highest_probs = np.max(probs, axis=1)
	# print (f"highest_probs.shape: {highest_probs.shape}")

	# get highest probability tokens
	highest_prob_token_ids = np.argmax(probs, axis=1)
	highest_prob_tokens = tokenizer.convert_ids_to_tokens(highest_prob_token_ids)
	# print (f"highest_prob_tokens.shape: {highest_prob_tokens.shape}")

	# check if predictions are correct
	ground_truth = np.concatenate(ground_truth)
	assert ground_truth.shape[0] == highest_prob_token_ids.shape[0], "Ground truth and highest probability token ids have different number of sequences"
	correct = (highest_prob_token_ids == ground_truth).astype(int)
	# print (f"correct.shape: {correct.shape}")

	# calculate entropy for each prediction
	entropies = np.array([entropy(p, base=2) for p in probs])
	# print (f"entropies.shape: {entropies.shape}")

	# return metrics for each sequence
	return {
		"is_correct": correct,
		"entropy": entropies,
		"highest_prob": highest_probs,
		"highest_prob_token": highest_prob_tokens
	}

def process_batch(model, tokenizer, batch_input_ids, batch_attention_mask, ground_truths, positions, chrom, start, chunk_offsets, file_handlers):
	inputs = {
		'input_ids': torch.tensor(np.array(batch_input_ids)),
		'attention_mask': torch.tensor(np.array(batch_attention_mask)),
	}
		# 'token_type_ids': torch.tensor([[0] * len(batch_input_ids[0])] * len(batch_input_ids)),

	token_metrics_batch = calculate_token_metrics(model, tokenizer, inputs, ground_truths)
	# Write results for each position

	# we use pos - 1 because we added CLS token, while offset mapping was computed without it
	positions = np.concatenate(positions)
	token_starts = [start + chunk_offsets[pos - 1][0] for pos in positions]
	token_ends = [start + chunk_offsets[pos - 1][1] for pos in positions]
	
	for metric, _ in file_handlers.items():
		assert len(token_metrics_batch[metric]) == len(positions), f"Number of metrics is not equal to number of positions: {len(token_metrics_batch[metric])} != {len(positions)}"
		pos_metrics = token_metrics_batch[metric]
		for start, end, metric_value in zip(token_starts, token_ends, pos_metrics):
			file_handlers[metric].write(f"{chrom}\t{start}\t{end}\t{metric_value}\n")

# Process genome and save metrics to BED/GRAPH
def process_genome(fasta_path, model, tokenizer, output_path_prefix, target_chrom, max_model_len_tokens, masking_fraction,
					seq_chunk_len = 50_000, batch_size=16, limit_bp=None):

	# create output directory if it doesn't exist
	output_dir = os.path.dirname(output_path_prefix)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Open FASTA file
	fasta = pysam.FastaFile(fasta_path)
	
	# Open output BEDGRAPH file
	metrics = {"is_correct": "bedgraph", "entropy": "bedgraph", "highest_prob": "bedgraph", "highest_prob_token": "bed"}
	run_prefix = output_path_prefix + "_" + target_chrom + "_"
	if limit_bp is not None:
		run_prefix += f"{limit_bp}_"
	file_handlers = {metric: open(run_prefix + metric + "." + ftype, 'w') for metric, ftype in metrics.items()}
	unaccessible_regions_file = open(run_prefix + "unaccessible_regions.bed", 'w')
	with open(run_prefix + "run_params.txt", "w") as fout:
		fout.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
		fout.write(f"config: {args.config}\n")
		fout.write(f"model_family: {args.model_family}\n")
		fout.write(f"cpt_path: {args.cpt_path}\n")
		fout.write(f"tokenizer_path: {args.tokenizer_path}\n")
		fout.write(f"input_len_tokens: {args.input_len_tokens}\n")
		fout.write(f"genome_path: {fasta_path}\n")
		fout.write(f"target_chrom: {target_chrom}\n")
		fout.write(f"seq_chunk_len: {seq_chunk_len}\n")
		fout.write(f"batch_size: {batch_size}\n")
		fout.write(f"limit_bp: {limit_bp}\n")
		fout.write(f"max_model_len_tokens: {max_model_len_tokens}\n")
		fout.write(f"modernbert_distr_path: {args.modernbert_distr_path}\n")
		fout.write(f"seq_chunk_len: {seq_chunk_len}\n")
		fout.write(f"masking_fraction: {masking_fraction}\n")

	valid_chroms = [] # chrom_name (==name of gapless fragment in fasta file), orig_start, orig_end
	for chrom in fasta.references:
		# Parse original coordinates from header
		header = chrom
		if ':' in header and '-' in header:
			original_chrom, coords = header.split(':')
			if original_chrom != target_chrom:
				continue
			orig_start, orig_end = map(int, coords.split('-'))
			assert fasta.get_reference_length(chrom) == orig_end - orig_start, f"Sequence length mismatch: {fasta.get_reference_length(chrom)} != {orig_end - orig_start}"
			if fasta.get_reference_length(chrom) < seq_chunk_len:
				unaccessible_regions_file.write(f"{original_chrom}\t{orig_start}\t{orig_end}\n")
			else:
				valid_chroms.append((chrom, orig_start, orig_end))
		else:
			raise ValueError(f"Invalid header format: {header}")

	total_length = sum([orig_end - orig_start for _, orig_start, orig_end in valid_chroms])
	if limit_bp:
		total_length = min(total_length, limit_bp)
	pbar = tqdm(total=total_length, desc=f"Processing {target_chrom}", unit='bp')
	processed_bp = 0

	for chrom_name, orig_start, orig_end in valid_chroms: # iterate over gapless fragments in fasta file
		start = 0
		chrom_len = fasta.get_reference_length(chrom_name)
		while start < chrom_len: # process each fragment in chunks of seq_chunk_len
			# Check if we've reached the limit
			if limit_bp and processed_bp >= limit_bp:
				break

			# Fetch sequence chunk
			end_position = min(start + seq_chunk_len, chrom_len)
			# start and end denote 0-based, half-open intervals.
			sequence = fasta.fetch(chrom_name, start, end_position).upper() 
			
			# Tokenize without special tokens
			tokenized = tokenizer(sequence, add_special_tokens=False)
			tokens = tokenized['input_ids']

			# Generate offset mapping manually based on token lengths; not all tokenizers support offset mapping
			offset_mapping = []
			current_pos = 0
			for token in tokenizer.convert_ids_to_tokens(tokens):
				# assert that token contains only A, C, G, T
				assert set(token) <= set(['A', 'C', 'G', 'T']), f"Token contains invalid characters: {token}"
				token_len = len(token)
				offset_mapping.append((current_pos, current_pos + token_len))
				current_pos += token_len
			assert offset_mapping[-1][1] == end_position-start, f"Offset mapping mismatch: {offset_mapping[-1][1]} != {end_position-start}"

			add_sep = tokenizer.sep_token_id is not None
			N_meaningful_tokens = max_model_len_tokens - 1 - add_sep  # -2 for CLS and SEP
			
			if len(tokens) < N_meaningful_tokens:
				unaccessible_regions_file.write(f"{target_chrom}\t{start + orig_start}\t{end_position + orig_start}\n")
				start += end_position-start
				continue

			attention_mask = tokenized['attention_mask']
			
			# Check for gap tokens and raise error if found
			gap_token_id = tokenizer.convert_tokens_to_ids(['-'])[0]
			if gap_token_id in tokens:
				raise AssertionError("Gap tokens ('-') found in sequence. Input sequence must not contain gaps.")
			
			# Process in chunks of N_meaningful_tokens tokens (to allow for CLS and SEP)
			last_chunk_end_bp = None

			for chunk_start in range(0, len(tokens) - N_meaningful_tokens + 1, N_meaningful_tokens):
				# Check if we've reached the limit
				if limit_bp and processed_bp >= limit_bp:
					break

				chunk_tokens = tokens[chunk_start:chunk_start + N_meaningful_tokens]
				chunk_offsets = offset_mapping[chunk_start:chunk_start + N_meaningful_tokens]
				last_chunk_end_bp = chunk_offsets[-1][1]
				chunk_attention = attention_mask[chunk_start:chunk_start + N_meaningful_tokens]

				# mask certain fraction of tokens
				if masking_fraction >= 1:
					_masking_fraction = masking_fraction/len(chunk_input_ids) # convert to fraction of tokens
					if _masking_fraction >= 0.5:
						warning_once(f"Masking fraction is greater than 0.5, which may lead to poor performance. Will set to 0.5.")
						_masking_fraction = 0.5
				else:
					_masking_fraction = masking_fraction
				num_splits = int(1/_masking_fraction)
				assert num_splits > 0, "Number of splits must be greater than 0"

				# split array of tokens into splits of num_tokens_to_mask tokens using numpy array split
				chunk_input_ids = np.array(chunk_tokens)
				splits = np.array_split(chunk_input_ids, num_splits)

				# evenly distribute tokens across splits
				splits = [np.arange(len(split))*num_splits + i for i, split in enumerate(splits)]
				# there should be no empty splits
				assert all(len(split) > 0 for split in splits), "There are empty splits"
				# total number of tokens should be equal to length of chunk_input_ids
				assert sum([len(split) for split in splits]) == len(chunk_input_ids), "Total number of tokens in splits is not equal to length of chunk_input_ids"
				# fraction of tokens in each split should be <= 50%
				assert all([len(split)/len(chunk_input_ids) <= 0.5 for split in splits]), "Fraction of tokens in each split is greater than 50%"
				# all tokens should participate in at least one split
				concatenated_splits = np.concatenate(splits)
				assert np.unique(concatenated_splits).shape[0] == len(chunk_input_ids), "All tokens should participate in at least one split"

				# Add CLS and SEP tokens
				chunk_input_ids = [tokenizer.cls_token_id] + chunk_tokens
				chunk_attention_mask = [1] + chunk_attention
				splits = [split + 1 for split in splits] # add 1 to each masked-token-position-split to account for CLS token

				if add_sep:
					chunk_input_ids.append(tokenizer.sep_token_id)
					chunk_attention_mask.append(1)
				assert len(chunk_input_ids) == max_model_len_tokens, \
					f"Chunk input ids length is not equal to max model len tokens: {len(chunk_input_ids)} != {max_model_len_tokens}"
				
				# Process each token position
				# Create batch of masked sequences

				batch_input_ids = []
				batch_attention_mask = []
				ground_truths = []
				positions = []

				for split in splits:
					if len(batch_input_ids) >= batch_size:
						# print (f"Processing batch of size {len(batch_input_ids)}")
						process_batch(model, tokenizer, batch_input_ids, 
									batch_attention_mask, ground_truths, positions, 
									target_chrom, start + orig_start, chunk_offsets, file_handlers)
						batch_input_ids = []
						batch_attention_mask = []
						ground_truths = []
						positions = []
					# print (f"Processing token {pos} of {len(chunk_input_ids)}")
					masked_input_ids = np.array(chunk_input_ids).copy()
					ground_truths.append(masked_input_ids[split])
					masked_input_ids[split] = tokenizer.mask_token_id
					batch_input_ids.append(masked_input_ids)
					batch_attention_mask.append(chunk_attention_mask)
					positions.append(split)

				assert len(batch_input_ids) > 0, "No batch input ids"
				process_batch(model, tokenizer, batch_input_ids, 
							batch_attention_mask, ground_truths, positions, 
							target_chrom, start + orig_start, chunk_offsets, file_handlers)
				processed_bp += chunk_offsets[-1][1] - chunk_offsets[0][0]
				pbar.n = processed_bp
				pbar.refresh()

			# Update start position based on last token's end position
			assert last_chunk_end_bp is not None, "Last chunk end bp is not set, seems that we did not process any chunks"
			start += last_chunk_end_bp

		# Check if we've reached the limit
		if limit_bp and processed_bp >= limit_bp:
			break

	for metric in file_handlers.keys():
		file_handlers[metric].close()
	fasta.close()

# Main execution
def main(args):
	# models = {
	# 	"gena-lm": ["AIRI-Institute/gena-lm-bert-base-t2t", 512],
	# 	"nucleotide-transformer-v2-100m": ["InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", 2048],
	# 	"ModernGENA_t2t_test": ["/disk/10tb/home/fishman/DNALM/ModernBERT/runs/moderngena_t2t_testrun/", 1024],
	# 	"ModernGENA_prom_multi": ["/disk/10tb/home/fishman/DNALM/ModernBERT/runs/moderngena-base-pretrain-promoters_multi/", 1024],
	# 	"ModernGENA_prom_multi_ep11": ["/disk/10tb/home/fishman/DNALM/ModernBERT/runs/moderngena_prom_ep11-ba65900/", 1024]
	# }

	# Load model and tokenizer
	model, tokenizer = load_model_and_tokenizer(cpt_path=args.cpt_path, 
												tokenizer_path=args.tokenizer_path, 
												model_family=args.model_family, 
												modernbert_distr_path=args.modernbert_distr_path)
	
	# Process genome and save results
	output_path = os.path.join(args.out_dir, args.name + "_")
	process_genome(
		fasta_path=args.genome_path,
		model=model,
		tokenizer=tokenizer,
		output_path_prefix=output_path,
		target_chrom=args.chrm,
		seq_chunk_len=args.seq_chunk_len,
		batch_size=args.batch_size,
		limit_bp=args.limit_bp,
		max_model_len_tokens=args.input_len_tokens,
		masking_fraction=args.masking_fraction,
	)
	print(f"Results saved to {output_path}")

if __name__ == "__main__":
	args = parse_args()
	assert args.masking_fraction <= 0.5 or args.masking_fraction >= 1, "Masking fraction must be below 0.5 or number of tokens to mask must be above 1"
	main(args)