import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks

import logging
from pathlib import Path
from hydra.utils import instantiate
import torch
from typing import List, Dict
from safetensors.torch import load_file

# Support both: run as package (relative imports) and run as script (parent on path)
try:
    from ..simple_annotation_dataset import collate_fn
    from ..evaluate_on_chromosome import bigWigExporter
except ImportError:
    _parent_dir = Path(__file__).resolve().parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    from simple_annotation_dataset import collate_fn
    from evaluate_on_chromosome import bigWigExporter

import numpy as np
import pyBigWig as bw

################################ inference #####################################
class pseudoBigWig:
	def __init__(self):
		self.data = {}

	def chroms(self):
		return {k:len(v) for k,v in self.data.items()}
	
	def add_chrom(self, chrom: str, data: np.ndarray):
		self.data[chrom] = data
		print ("Adding chromosome {chrom} with data length {len(data)}")

	def values(self, chrom: str, start: int, end: int):
		assert chrom in self.chroms().keys(), f"Chromosome {chrom} not found"
		assert start >= 0 and end <= len(self.data[chrom]), f"Start {start} or end {end} out of range for chromosome {chrom}"
		return self.data[chrom][start:end]

	def close(self):
		pass

class inferenceHandler(bigWigExporter):
	def as_bigwig(self, label: str):
		result = pseudoBigWig()
		print (len(self.chrom_data))
		for chrom in self.chrom_data.keys():
			_ = None
			print (f"processing chromosome {chrom} with labels {self.chrom_data[chrom].keys()} and looking for label {label}")
			if label in self.chrom_data[chrom].keys():
				print ("Adding chromosome {chrom} with data length {len(self.chrom_data[chrom][label])}")
				result.add_chrom(chrom, self.chrom_data[chrom][label])
				print (result.chroms())
				_ = label
			if _ is None:
				raise ValueError(f"No results found for label {label} on chromosome {chrom}. Available labels: {self.chrom_data[chrom].keys()}")
		
		assert len(result.chroms()) > 0, f"No results found for label {label}"
		return result

	def __init__(self, chroms: List[str], chrom_lengths: List[int], logger: logging.Logger = None, context_fraction: float = 0.1):
		self.chroms = chroms
		self.chrom_lengths = chrom_lengths
		self.logger = logger if logger is not None else logging.getLogger(__name__)
		self.context_fraction = context_fraction
		self.last_chrom = None
		self.last_end = None
		self.last_strand = None
		self.sigmoid = torch.nn.Sigmoid()
		self.chrom_data = {}
	
def run_inference(experiment_config: Dict, model_cpt: str, logger: logging.Logger):
	"""
	Run inference on a dataset and return a datahandler object.
	datahandler has .chrom_data attr which is dictionary of dictionaries: chrom_data[s_chrom][label]
	where s_chrom is a chromosome name and label is a label name (i.e. "tss_+", "polya_-", "intragenic_+", "intragenic_-")
	chrom_data[s_chrom][label] is a numpy array of shape (chrom_len)
	which contains the values for the label on the chromosome (nan for unmeasured positions, i.e. at chromsome ends).
	"""

	dataset_config = experiment_config.eval_dataset
	dataset_strandedness = dataset_config.get('strand', '+')
		
	dataset = instantiate(dataset_config)

	if dataset_strandedness == "random":
		assert len(dataset) % 2 == 0, "Dataset must have even number of samples for random strandedness"

	datahandler = inferenceHandler(chroms=list(dataset.chrom_info.keys()), 
									chrom_lengths=list(dataset.chrom_info.values()), 
									logger=logger)

	dataloader = instantiate(experiment_config.dataloader)

	model_config = experiment_config.model
	model = instantiate(model_config)
	print (f"Loading model from {model_cpt}")
	state = load_file(model_cpt, device="cpu")
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
			elif batch[k] is None:
				assert k == 'targets' # only targets can be None for inference
			else:
				raise ValueError(f"Unsupported type: {type(batch[k])}")
		with torch.no_grad():
			output = model.forward(**batch)
		datahandler.process_batch(output, metadata)

	dataset.close()
	return datahandler

########################## post-processing - peak finding#####################################

def combine_strands(tss_plus, tss_minus, polya_plus, polya_minus):
	tss_combined = np.maximum(tss_plus, tss_minus)
	polya_combined = np.maximum(polya_plus, polya_minus)
	return tss_combined, polya_combined

def merge_rev_comp(signal_plus, signal_minus):
	return np.mean([signal_plus, signal_minus], axis=0)

def read_bigwig(bigwig_path, filename, chromosome):
	if isinstance(bigwig_path, str): # we read from a file
		filepath = os.path.join(bigwig_path, filename)
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"BigWig file not found: {filepath}")

		with bw.open(filepath) as bw_file:
			if chromosome not in bw_file.chroms():
				raise ValueError(f"Chromosome {chromosome} not found in {filename}")

			chrom_length = bw_file.chroms()[chromosome]
			print(f"Reading {filename}: chromosome {chromosome} (length: {chrom_length})")

			values = bw_file.values(chromosome, 0, chrom_length)
			values = [0.0 if np.isnan(v) else v for v in values]
		return np.array(values, dtype=np.float32)
	elif isinstance(bigwig_path, bigWigExporter): # we read from a bigWigExporter object
		assert filename.endswith(".bw"), "Filename must end with .bw"
		label = filename.rstrip(".bw")
		values = bigwig_path.chrom_data[chromosome][label]
		np.nan_to_num(values, copy=False, nan=0.0)
		return values.astype(np.float32)
	else:
		raise ValueError(f"Unsupported type: {type(bigwig_path)}")

def prepare_preds_for_peaks_onlyRC(files_path, chromosome, logger: logging.Logger):
	data = {}
	for label in ['tss', 'polya']:
		for file_strand in ['+', '-']:
			file_name = f"{label}_{file_strand}"
			data[file_name] = read_bigwig(files_path, f"{file_name}.bw", chromosome)

			rc_file_name = f"{label}_{file_strand}rev_comp_"
			data[rc_file_name] = read_bigwig(files_path, f"{rc_file_name}.bw", chromosome)

	for name in data.keys():
		logger.info(f"{name}.max = {np.max(data[name])}")

	tss_plus = merge_rev_comp(data["tss_+"], data["tss_-rev_comp_"])
	tss_minus = merge_rev_comp(data["tss_-"], data["tss_+rev_comp_"])

	polya_plus = merge_rev_comp(data["polya_+"], data["polya_-rev_comp_"])
	polya_minus = merge_rev_comp(data["polya_-"], data["polya_+rev_comp_"])

	return tss_plus, polya_plus, tss_minus, polya_minus

def fft_lowpass(x: np.ndarray, frac: float) -> np.ndarray:
	Xf = np.fft.rfft(x)
	k = int(np.clip(frac, 0.0, 1.0) * len(Xf))
	if k < 1:
		k = 1
	X_lp = np.zeros_like(Xf)
	X_lp[:k] = Xf[:k]
	y = np.fft.irfft(X_lp, n=len(x))
	return y

def call_peaks_on_segment(x: np.ndarray,
						  lp_frac: float,
						  pk_prom: float,
						  pk_dist: int,
						  pk_height: float):
	y_lp = fft_lowpass(x, frac=lp_frac)
	idx, _props = find_peaks(y_lp, prominence=pk_prom, distance=pk_dist, height=pk_height)
	return idx, y_lp

def peak_finding(X: np.ndarray, LP_FRAC: float, PK_PROM: float, PK_DIST: int, PK_HEIGHT: float):
	X = np.nan_to_num(X)
	N = X.shape[1]
	print("Input shape:", X.shape)

	mask = np.zeros((4, N), dtype=np.uint8)

	peak_counts = []
	for r in range(4):
		x = X[r, :].astype(float, copy=False)
		idx_local, _y_lp = call_peaks_on_segment(
			x,
			lp_frac=LP_FRAC,
			pk_prom=PK_PROM,
			pk_dist=PK_DIST,
			pk_height=PK_HEIGHT
		)
		mask[r, idx_local] = 1
		peak_counts.append(len(idx_local))

	return mask

	########################## post-processing - pairing #####################################

def find_tss_polya_pairs_right_left_only(arr, chrom_name, window_size=2_000_000, k=10,
										out_bed_path=None, progress_every=None, 
										logger=logging.Logger):
	if arr.ndim != 2 or arr.shape[0] != 4:
		raise ValueError("arr must have shape (4, X) in order: TSS+, PolyA+, TSS-, PolyA-")
	X = int(arr.shape[1])
	if window_size > X:
		window_size = X

	tss_plus_idx = np.flatnonzero(arr[0].astype(np.bool_, copy=False)); tss_plus_idx.sort()
	polya_plus_idx = np.flatnonzero(arr[1].astype(np.bool_, copy=False)); polya_plus_idx.sort()
	tss_minus_idx = np.flatnonzero(arr[2].astype(np.bool_, copy=False)); tss_minus_idx.sort()
	polya_minus_idx = np.flatnonzero(arr[3].astype(np.bool_, copy=False)); polya_minus_idx.sort()

	pairs_sign = {}

	def choose_k_nearest(seed: int, candidates: np.ndarray) -> np.ndarray:
		if k is None or k <= 0 or candidates.size <= k:
			return candidates
		dist = np.abs(candidates - seed)
		idx = np.argpartition(dist, k - 1)[:k]
		order = np.lexsort((candidates[idx], dist[idx]))
		return candidates[idx][order]

	def scan_tss_to_polya_one_sided(seeds_tss: np.ndarray, targets_polya: np.ndarray,
								   direction: str, strand_sign: str, label: str):
		if progress_every:
			logger.info(f"Scanning {len(seeds_tss):,} {label} TSS seeds one sided")
		for ii, i in enumerate(seeds_tss):
			i = int(i)

			if direction == "right":
				start_w = i
				end_w = min(X, i + window_size)
				left = targets_polya.searchsorted(start_w, side="left")
				right = targets_polya.searchsorted(end_w, side="left")
			elif direction == "left":
				start_w = max(0, i - window_size)
				end_w = i + 1
				left = targets_polya.searchsorted(start_w, side="left")
				right = targets_polya.searchsorted(end_w, side="left")
			else:
				raise ValueError("direction must be 'right' or 'left'")

			if right > left:
				js_window = targets_polya[left:right]
				js_use = choose_k_nearest(i, js_window)

				for j in js_use:
					j = int(j)
					a, b = (i, j) if i <= j else (j, i)
					pairs_sign.setdefault((a, b), strand_sign)

			if progress_every and (ii + 1) % progress_every == 0:
				logger.info(f"  processed {ii+1:,}/{len(seeds_tss):,}; pairs so far: {len(pairs_sign):,}")

	scan_tss_to_polya_one_sided(
		seeds_tss=tss_plus_idx,
		targets_polya=polya_plus_idx,
		direction="right",
		strand_sign="+",
		label="plus"
	)

	scan_tss_to_polya_one_sided(
		seeds_tss=tss_minus_idx,
		targets_polya=polya_minus_idx,
		direction="left",
		strand_sign="-",
		label="minus"
	)

	pairs_sorted = sorted(pairs_sign.keys(), key=lambda ab: (ab[0], ab[1]))

	final_pairs = []
	for a, b in pairs_sorted:
		final_pairs.append((chrom_name, a, b, pairs_sign[(a, b)], []))

	if out_bed_path is not None:
		with open(out_bed_path, "w") as bed:
			for a, b in pairs_sorted:
				bed.write(f"{chrom_name}\t{a}\t{b}\t{pairs_sign[(a, b)]}\n")

	return final_pairs

########################## post-processing - filtering pairs #####################################

def fetch_values(handle, chrom: str, start: int, end: int) -> np.ndarray:
	vals = handle.values(chrom, start, end)
	if vals is None:
		return np.zeros(max(0, end - start), dtype=np.float32)
	vals = np.array([0.0 if v is None else v for v in vals], dtype=np.float32)
	np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
	return vals


def ensure_chrom_available(handles: Dict[str, bw.pyBigWig], chroms_needed: List[str]):
	for name, handle in handles.items():
		chrom_dict = handle.chroms()
		missing = [c for c in chroms_needed if c not in chrom_dict]
		if missing:
			raise ValueError(f"[FATAL] bigWig '{name}' missing chromosomes: {missing}")
		sample = list(chrom_dict.items())[:3]
		print(f"[INFO] '{name}' bigWig has {len(chrom_dict)} chroms. Example entries: {sample}")


def open_bw_inputs(bw_dir_or_file: str, BW_PLUS: str, BW_MINUS: str, BW_PLUS_RC: str, BW_MINUS_RC: str):
	paths = {
		"intragenic_+":          BW_PLUS,
		"intragenic_-":          BW_MINUS,
		"intragenic_+rev_comp_": BW_PLUS_RC,
		"intragenic_-rev_comp_": BW_MINUS_RC,
	}

	if isinstance(bw_dir_or_file, bigWigExporter):
		return "stranded", {k: bw_dir_or_file.as_bigwig(v.rstrip(".bw")) for k,v in paths.items()}

	if os.path.isdir(bw_dir_or_file):
		bw_dir = bw_dir_or_file
		paths = {k: os.path.join(bw_dir, v) for k, v in paths.items()}
		for p in paths.values():
			if not os.path.exists(p):
				raise FileNotFoundError(f"[FATAL] Missing bigWig: {p}")

		print("[INFO] Mode: STRANDED (4 bigWigs in directory)")
		for k, p in paths.items():
			print(f"  - {k:20s}: {p}")

		handles = {k: bw.open(p) for k, p in paths.items()}
		return "stranded", handles

	if os.path.isfile(bw_dir_or_file) and bw_dir_or_file.endswith(".bw"):
		print("[INFO] Mode: UNSTRANDED (single bigWig file; already averaged)")
		print(f"  - intragenic_unstranded: {bw_dir_or_file}")
		handles = {"intragenic_unstranded": bw.open(bw_dir_or_file)}
		return "unstranded", handles

	raise FileNotFoundError(f"[FATAL] --bw_dir must be a directory or a .bw file: {bw_dir_or_file}")


def close_handles(handles: Dict[str, bw.pyBigWig]):
	for h in handles.values():
		try:
			h.close()
		except Exception:
			pass

def filter_bed_by_intragenic(pairs, bw_dir, BW_PLUS, BW_MINUS, BW_PLUS_RC, BW_MINUS_RC,
							prob_threshold,zero_fraction_drop_threshold, bed_out, logger: logging.Logger):
	intervals = pairs
	if not intervals:
		logger.warning("[WARN] No intervals to process. Writing empty output and exiting.")
		os.makedirs(os.path.dirname(os.path.abspath(bed_out)), exist_ok=True)
		with open(bed_out, "w") as _:
			pass
		return

	mode, handles = open_bw_inputs(bw_dir, BW_PLUS=BW_PLUS, BW_MINUS=BW_MINUS, BW_PLUS_RC=BW_PLUS_RC, BW_MINUS_RC=BW_MINUS_RC)

	unique_chroms = sorted({iv[0] for iv in intervals})
	ensure_chrom_available(handles, unique_chroms)

	logger.info(f"Thresholds: prob_threshold={prob_threshold}, "
		  f"zero_fraction_drop_threshold={zero_fraction_drop_threshold}")
	logger.info(f"Processing {len(intervals):,} intervals...")

	kept = []
	n_dropped = 0

	for chrom, start, end, strand, extra in tqdm(intervals, desc="Filtering", unit="interval"):
		if end <= start:
			n_dropped += 1
			continue

		L = end - start

		if mode == "unstranded":
			signal = fetch_values(handles["intragenic_unstranded"], chrom, start, end)
			if len(signal) <= 0:
				n_dropped += 1
				continue
			if len(signal) != L:
				signal = signal[:min(len(signal), L)]
				L = len(signal)

		else:
			v_plus = fetch_values(handles["intragenic_+"], chrom, start, end)
			v_minus = fetch_values(handles["intragenic_-"], chrom, start, end)
			v_plus_rc = fetch_values(handles["intragenic_+rev_comp_"], chrom, start, end)
			v_minus_rc = fetch_values(handles["intragenic_-rev_comp_"], chrom, start, end)

			minL = min(len(v_plus), len(v_minus), len(v_plus_rc), len(v_minus_rc))
			if minL <= 0:
				n_dropped += 1
				continue
			if minL != L:
				v_plus = v_plus[:minL]
				v_minus = v_minus[:minL]
				v_plus_rc = v_plus_rc[:minL]
				v_minus_rc = v_minus_rc[:minL]
				L = minL

			plus_signal = 0.5 * (v_plus + v_minus_rc)
			minus_signal = 0.5 * (v_minus + v_plus_rc)

			if strand == "+":
				signal = plus_signal
			elif strand == "-":
				signal = minus_signal
			else:
				signal = np.maximum(plus_signal, minus_signal)

		binary = (signal > prob_threshold).astype(np.uint8)
		zero_fraction = float((binary == 0).mean())

		if zero_fraction > zero_fraction_drop_threshold:
			n_dropped += 1
		else:
			kept.append((chrom, start, end, strand, extra))

	os.makedirs(os.path.dirname(os.path.abspath(bed_out)), exist_ok=True)
	with open(bed_out, "w") as out:
		for chrom, start, end, strand, extra in kept:
			row = [chrom, str(start), str(end)]
			row.append(strand)
			row += extra
			out.write("\t".join(row) + "\n")

	total = len(intervals)
	kept_n = len(kept)
	logger.info(f"Kept {kept_n:,} / {total:,} intervals "
		  f"({100.0 * kept_n / total:.2f}%). Dropped {n_dropped:,} ({100.0 * n_dropped / total:.2f}%).")
	logger.info(f"Intervals left after filtration: {kept_n:,}")

	close_handles(handles)

######################################################### BED post-processing #########################################################

def shift_bed_by_UCSC_chr_header(bed_path: str):
	df = pd.read_csv(bed_path, sep="\t", header=None)
	chroms = df.iloc[:,0].unique()
	assert len(chroms) == 1, f"Only one chromosome is supported. Found {len(chroms)} chromosomes"
	chrom = chroms[0]
	chrom_name = chrom.split(":")[0]
	chrom_start = chrom.split(":")[1].split("-")[0].replace(",", "")
	chrom_start = int(chrom_start)
	df.iloc[:,1] = df.iloc[:,1].astype(int) + chrom_start
	df.iloc[:,2] = df.iloc[:,2].astype(int) + chrom_start
	df.iloc[:,0] = chrom_name
	df.to_csv(bed_path, sep="\t", header=None, index=False)