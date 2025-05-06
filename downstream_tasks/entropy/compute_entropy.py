import os
import numpy as np
import pysam
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
from scipy.stats import entropy
import requests
from tqdm import tqdm
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--genome_path", type=str)
	parser.add_argument("--out_dir", type=str, default="data/")
	parser.add_argument("--chrm", type=str, help="Chromosome to process (e.g., 'chr1')")
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--model", type=str, choices=['gena-lm', 'nucleotide-transformer-v2-100m'], 
						default='gena-lm', help="Model to use for analysis")
	parser.add_argument("--limit_bp", type=int, help="Limit processing to this number of base pairs", default=None)
	return parser.parse_args()

# Load model and tokenizer
def load_model_and_tokenizer(model_name, model_type):
	if model_type == 'gena-lm':
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
	elif model_type.find('nucleotide-transformer') != -1:  # nucleotide-transformer
		tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
		model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
	else:
		raise ValueError(f"Model type {model_type} not supported")
	
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
		outputs = model(**inputs)
		logits = outputs.logits

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
	correct = (highest_prob_token_ids == np.array(ground_truth)).astype(int)
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
		'input_ids': torch.tensor(batch_input_ids),
		'token_type_ids': torch.tensor([[0] * len(batch_input_ids[0])] * len(batch_input_ids)),
		'attention_mask': torch.tensor(batch_attention_mask),
	}
	token_metrics_batch = calculate_token_metrics(model, tokenizer, inputs, ground_truths)
	# Write results for each position
	for i, pos in enumerate(positions):
		token_start = start + chunk_offsets[pos - 1][0]  # -1 because we added CLS
		token_end = start + chunk_offsets[pos - 1][1]
		
		# Write to bedgraph
		for metric, _ in file_handlers.items():
			pos_metrics = token_metrics_batch[metric][i]
			file_handlers[metric].write(f"{chrom}\t{token_start}\t{token_end}\t{pos_metrics}\n")


# Process genome and save metrics to BED/GRAPH
def process_genome(fasta_path, model, tokenizer, output_path_prefix, target_chrom, batch_size=16, limit_bp=None):
	seq_chunk_len = 50_000  # 50kb

	# Open FASTA file
	fasta = pysam.FastaFile(fasta_path)
	
	# Open output BEDGRAPH file
	metrics = {"is_correct": "bedgraph", "entropy": "bedgraph", "highest_prob": "bedgraph", "highest_prob_token": "bed"}
	file_handlers = {metric: open(output_path_prefix + "_" + target_chrom + "_" + metric + "." + ftype, 'w') for metric, ftype in metrics.items()}
	
	valid_chroms = []
	for chrom in fasta.references:
		# Skip contigs shorter than seq_chunk_len
		if fasta.get_reference_length(chrom) < seq_chunk_len:
			continue
			
		# Parse original coordinates from header
		header = chrom
		if ':' in header and '-' in header:
			original_chrom, coords = header.split(':')
			if target_chrom and original_chrom != target_chrom:
				continue
			orig_start, orig_end = map(int, coords.split('-'))
			valid_chroms.append((chrom, orig_start, orig_end))
		else:
			raise ValueError(f"Invalid header format: {header}")

	total_length = sum([orig_end - orig_start for _, orig_start, orig_end in valid_chroms])
	if limit_bp:
		total_length = min(total_length, limit_bp)
	pbar = tqdm(total=total_length, desc=f"Processing {target_chrom}", unit='bp')
	processed_bp = 0

	for chrom_name, orig_start, orig_end in valid_chroms:
		start = 0
		while start + seq_chunk_len < orig_end:
			# Check if we've reached the limit
			if limit_bp and processed_bp >= limit_bp:
				break

			# Fetch sequence chunk
			sequence = fasta.fetch(chrom_name, start, min(start + seq_chunk_len, fasta.get_reference_length(chrom_name))).upper()
			
			# Tokenize without special tokens
			tokenized = tokenizer(sequence, add_special_tokens=False, return_offsets_mapping=True)
			tokens = tokenized['input_ids']
			assert len(tokens) > 510, f"Sequence is too short: {len(tokens)} for chromosome {chrom_name}, start: {start}"
			offset_mapping = tokenized['offset_mapping']
			attention_mask = tokenized['attention_mask']
			
			# Check for gap tokens and raise error if found
			gap_token_id = tokenizer.convert_tokens_to_ids(['-'])[0]
			if gap_token_id in tokens:
				raise AssertionError("Gap tokens ('-') found in sequence. Input sequence must not contain gaps.")
			
			# Process in chunks of 510 tokens (to allow for CLS and SEP)
			for chunk_start in range(0, len(tokens) - 510 + 1, 510):
				# Check if we've reached the limit
				if limit_bp and processed_bp >= limit_bp:
					break

				chunk_tokens = tokens[chunk_start:chunk_start + 510]
				chunk_offsets = offset_mapping[chunk_start:chunk_start + 510]
				chunk_attention = attention_mask[chunk_start:chunk_start + 510]
				
				# Add CLS and SEP tokens
				chunk_input_ids = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
				chunk_attention_mask = [1] + chunk_attention + [1]
				
				# Process each token position
				# Create batch of masked sequences

				batch_input_ids = []
				batch_attention_mask = []
				ground_truths = []
				positions = []
				
				for pos in range(1, len(chunk_input_ids) - 1):
					if len(batch_input_ids) >= batch_size:
						process_batch(model, tokenizer, batch_input_ids, 
									batch_attention_mask, ground_truths, positions, 
									target_chrom, start + orig_start, chunk_offsets, file_handlers)
						batch_input_ids = []
						batch_attention_mask = []
						ground_truths = []
						positions = []
					masked_input_ids = chunk_input_ids.copy()
					ground_truths.append(masked_input_ids[pos])
					masked_input_ids[pos] = tokenizer.mask_token_id
					batch_input_ids.append(masked_input_ids)
					batch_attention_mask.append(chunk_attention_mask)
					positions.append(pos)

				assert len(batch_input_ids) > 0, "No batch input ids"
				process_batch(model, tokenizer, batch_input_ids, 
							batch_attention_mask, ground_truths, positions, 
							target_chrom, start + orig_start, chunk_offsets, file_handlers)
				processed_bp += chunk_offsets[-1][1] - chunk_offsets[0][0]
				pbar.n = processed_bp
				pbar.refresh()

			# Update start position based on last token's end position
			if offset_mapping:
				start += offset_mapping[-1][1]
			else:
				raise ValueError(f"No offset mapping found for chromosome {chrom}, start: {start}")

		# Check if we've reached the limit
		if limit_bp and processed_bp >= limit_bp:
			break

	for metric in file_handlers.keys():
		file_handlers[metric].close()
	fasta.close()

# Main execution
def main(args):
	models = {
		"gena-lm": "AIRI-Institute/gena-lm-bert-base-t2t",
		"nucleotide-transformer-v2-100m": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
	}

	# Load model and tokenizer
	model, tokenizer = load_model_and_tokenizer(models[args.model], args.model)
	
	# Process genome and save results
	output_path = os.path.join(args.out_dir, args.model + "_")
	process_genome(args.genome_path, model, tokenizer, output_path, args.chrm, args.batch_size, args.limit_bp)
	print(f"Results saved to {output_path}")

if __name__ == "__main__":
	args = parse_args()
	main(args)