"""Dataset for the convolutional (BPE-free) expression model.

Differences from :mod:`expression_dataset_final`:

* No BPE tokenizer. Each sample carries a fixed-size window of raw nucleotide
  codes (A=0, C=1, G=2, T=3, N/pad=4) centred on the TSS.
* One item is a gene together with ``n_keys`` tracks, as in the token model, but
  the DNA window is returned **once** instead of being repeated per track: the
  model broadcasts the tower output over the tracks. Measured on A100, the CNN,
  the projection and the tower are ~72% of a forward pass and are identical for
  every track of a gene, so sharing them is worth roughly 4x.
* There is no ``dataset_flag``: every item is one gene across tracks, so the
  de-duplication is structural rather than inferred from a per-block flag.
* Positions are fixed-width bins at the CNN output resolution instead of
  variable-width BPE tokens. Index 0 is reserved for the CLS position that
  carries the gene-level TPM/CPM target, exactly as before, so all downstream
  loss and metric code that reads ``[:, 0]`` keeps working unchanged.

Everything else (targets index, qnorm tables, description cache, bigWig signal
aggregation) is reused from :class:`ExpressionDataset` without modification.
"""

import hashlib
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from downstream_tasks.expression_prediction.alphagenome_cnn import (
    PAD_CODE,
    bin_coords,
    encode_nucleotides,
    reverse_complement_codes,
    window_for_tss,
)
from downstream_tasks.expression_prediction.expression_dataset_final import ExpressionDataset


class ExpressionDatasetCNN(ExpressionDataset):
    """One sample = one (gene, cell type) pair with a raw-DNA window.

    Parameters
    ----------
    dna_window_len:
        Size of the genomic window in bp. Must be divisible by ``cnn_total_stride``.
    cnn_total_stride:
        ``pool_stride ** num_pooling_stages`` of the CNN encoder, i.e. the bp per
        output bin. Determines how many label bins a sample has.
    window_offset:
        Shift of the window centre relative to the TSS, in bp, measured in the
        direction of transcription (positive = downstream).
    """

    def __init__(
        self,
        targets_path: str,
        genome: str,
        forward_intervals_path: str = None,
        reverse_intervals_path: str = None,
        dna_window_len: int = 131072,
        cnn_total_stride: int = 128,
        out_bins: Optional[int] = None,
        window_offset: int = 0,
        n_keys: Optional[int] = None,
        loglevel: int = logging.WARNING,
        seed: int = 42,
        transform_targets_bw=None,
        transform_targets_tpm=None,
        bw: str = "",
        tpm: str = "",
        hash_prefix=None,
        norm_bw: bool = False,
        text_tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
        text_max_seq_len: int = 510,
        profile_timing: bool = False,
    ):
        # NOTE: ExpressionDataset.__init__ is deliberately not called. It builds a
        # BPE token cache and the n_keys chunking, neither of which exists here.
        # Its *methods* are reused as-is; the original file stays untouched.
        Dataset.__init__(self)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=loglevel)

        if dna_window_len % cnn_total_stride != 0:
            raise ValueError(
                f"dna_window_len ({dna_window_len}) must be divisible by "
                f"cnn_total_stride ({cnn_total_stride})"
            )

        # Opt-in per-section timing for __getitem__; costs nothing when off.
        # Only meaningful with num_workers=0, since workers accumulate their own
        # counters in their own processes.
        self.profile_timing = bool(profile_timing)
        self._timings = defaultdict(float)
        self._timed_calls = 0

        self.dna_window_len = int(dna_window_len)
        self.cnn_total_stride = int(cnn_total_stride)
        # Bins the window would have if the encoder returned all of them. The
        # AlphaGenome CNN does; Flashzoi crops its output to a centred subset, so
        # out_bins can be smaller and the labels have to be shifted to match.
        self.n_bins_full = self.dna_window_len // self.cnn_total_stride
        self.n_bins = self.n_bins_full if out_bins is None else int(out_bins)
        if self.n_bins > self.n_bins_full:
            raise ValueError(
                f"out_bins ({self.n_bins}) exceeds the bins in the window "
                f"({self.n_bins_full} = {dna_window_len} // {cnn_total_stride}): "
                "the encoder cannot return more positions than the window holds"
            )
        if (self.n_bins_full - self.n_bins) % 2 != 0:
            raise ValueError(
                f"crop must be symmetric: {self.n_bins_full} - {self.n_bins} is odd, so "
                "the encoder's centre would sit half a bin away from these coordinates"
            )
        # bp dropped from each edge of the window by the encoder's centre crop.
        self.crop_offset = ((self.n_bins_full - self.n_bins) // 2) * self.cnn_total_stride
        self.window_offset = int(window_offset)
        # Same layout as the token model: [CLS] + content + [SEP]. There
        # gen_max_seq_len=1024 holds 1022 BPE tokens; here it holds 1022 bins.
        self.seq_len = self.n_bins + 2

        self.genome = genome
        self.seed = seed
        np.random.seed(self.seed)
        self.epoch = 0

        self.bw = bw
        self.tpm = tpm
        self.norm_bw = norm_bw
        self.targets_path = targets_path
        self.transform_targets_bw = transform_targets_bw
        self.transform_targets_tpm = transform_targets_tpm

        # Attributes the reused parent methods expect to find.
        self.sequences = None
        self.files_opened = False
        self.signals_cache = None
        self._genome_sizes_map = {}
        self._genome_sizes_base_dir = None
        genome_sizes_path = Path(self.genome).expanduser().parent.parent / "genome_sizes.tsv"
        if genome_sizes_path.exists():
            self._genome_sizes_base_dir = genome_sizes_path.parent
            df_sizes = pd.read_csv(genome_sizes_path, sep="\t")
            for p, size in zip(df_sizes["path"], df_sizes["size"]):
                if pd.isna(p) or pd.isna(size):
                    continue
                genome_path = Path(str(p)).expanduser()
                if not genome_path.is_absolute():
                    genome_path = self._genome_sizes_base_dir / genome_path
                self._genome_sizes_map[str(genome_path.resolve())] = size

        assert (
            forward_intervals_path is not None or reverse_intervals_path is not None
        ), "Either forward_intervals_path or reverse_intervals_path must be provided"
        self.intervals_hash = (
            self._name_and_size(forward_intervals_path)
            + "|"
            + self._name_and_size(reverse_intervals_path)
        )
        if hash_prefix is None:
            base = (
                os.path.dirname(forward_intervals_path)
                if forward_intervals_path is not None
                else os.path.dirname(reverse_intervals_path)
            )
            self.hash_prefix = os.path.join(base, "dataset_hash_cnn")
        else:
            self.hash_prefix = hash_prefix

        self.read_paths()
        self._bw_key_to_col = {k: i for i, k in enumerate(self.paths.keys())}
        self.all_keys = list(self.paths.keys())

        forward_genes = (
            pd.read_csv(forward_intervals_path, sep=None, engine="python")
            if forward_intervals_path is not None
            else pd.DataFrame()
        )
        forward_genes["strand"] = "+"
        reverse_genes = (
            pd.read_csv(reverse_intervals_path, sep=None, engine="python")
            if reverse_intervals_path is not None
            else pd.DataFrame()
        )
        reverse_genes["strand"] = "-"
        self.genes = pd.concat([forward_genes, reverse_genes], ignore_index=True)

        # Chromosome lengths are needed to clamp windows.
        self._ensure_sequences_open("window construction")
        self._chrom_lengths = {
            name: length
            for name, length in zip(self.sequences.references, self.sequences.lengths)
        }

        if self.tpm:
            assert all(
                self.paths[k][1] is not None for k in self.paths
            ), "TPM paths are not set for some of the keys"
            tpm_hash_path = self.get_tpm_hash_path()
            if os.path.exists(tpm_hash_path):
                self.logger.debug(f"Loading tpm cache from {tpm_hash_path}")
                with open(tpm_hash_path, "rb") as fh:
                    self.tpm_lookup = pickle.load(fh)
                assert len(self.tpm_lookup) == len(self.paths)
                assert all(key in self.tpm_lookup for key in self.paths.keys())
            else:
                tpm_cache = {}
                for key, (_bw_paths, tpm_path) in tqdm.tqdm(self.paths.items()):
                    tpm_cache[key] = pd.read_csv(tpm_path, dtype=np.float32)
                self.tpm_lookup = {
                    key: df.T.set_index(df.columns) for key, df in tpm_cache.items()
                }
                with open(tpm_hash_path, "wb") as fh:
                    pickle.dump(self.tpm_lookup, fh)

        if self.bw:
            self.signals_cache_path = self.get_signals_hash_path() + ".h5"
            if os.path.exists(self.signals_cache_path):
                self.signals_cache = h5py.File(self.signals_cache_path, "r")
            else:
                self.precompute_signals()

        self.n_keys = len(self.all_keys) if n_keys is None else int(n_keys)
        self.n_cell_chunks = ((len(self.all_keys) - 1) // self.n_keys) + 1

        self._build_valid_genes()
        self._init_descriptions(text_tokenizer, text_max_seq_len)
        self._close_forkable_handles()

    def _close_forkable_handles(self):
        """Drop pysam/pyBigWig handles before the DataLoader forks its workers.

        Both libraries return corrupted reads from a handle inherited through
        fork. Every worker reopens what it needs lazily -- the fasta via
        ``_ensure_sequences_open`` and the bigWigs via ``open_files`` (which
        ``worker_init_fn`` calls).
        """
        if self.sequences is not None:
            self.sequences.close()
            self.sequences = None
        for per_strand in getattr(self, "bigWigHandlers", {}).values():
            for handle in per_strand.values():
                try:
                    handle.close()
                except Exception:
                    pass
        self.bigWigHandlers = {}
        self.files_opened = False

    # ------------------------------------------------------------------
    # cache paths -- distinct from the BPE caches so the two never collide
    # ------------------------------------------------------------------

    def _cache_signature(self) -> str:
        m = hashlib.blake2b(digest_size=8)
        for part in (
            "cnn_bins",
            str(self.intervals_hash),
            self._name_and_size(self.targets_path),
            self._name_and_size(self.genome),
            str(self.dna_window_len),
            str(self.cnn_total_stride),
            # Empty when the encoder returns the whole window, so hashes of runs
            # made before out_bins existed stay byte-identical.
            "" if self.n_bins == self.n_bins_full else f"crop{self.n_bins}",
            str(self.window_offset),
            "norm_bw" if self.norm_bw else "",
            "".join(sorted(self.paths.keys())),
        ):
            m.update(str(part).encode("utf-8"))
        return m.hexdigest()

    def get_hash_path(self):
        return f"{self.hash_prefix}.{self._cache_signature()}"

    def get_signals_hash_path(self):
        return f"{self.hash_prefix}.signal.{self._cache_signature()}"

    def get_tpm_hash_path(self):
        return f"{self.hash_prefix}.tpm.{self._cache_signature()}"

    # ------------------------------------------------------------------
    # windows and bins
    # ------------------------------------------------------------------

    def _window_for_gene(self, row) -> Tuple[int, int]:
        return window_for_tss(
            tss=int(row["TSS"]),
            chrom_len=int(self._chrom_lengths[row["chromosome"]]),
            window_len=self.dna_window_len,
            strand=row["strand"],
            offset=self.window_offset,
        )

    def _bin_coords(self, start: int, end: int, strand: str) -> Tuple[np.ndarray, np.ndarray]:
        # crop_offset is 0 unless the encoder returns a centred crop of the window
        # (Flashzoi does); then the label bins must be shifted inwards to match.
        return bin_coords(
            start + self.crop_offset,
            end - self.crop_offset,
            strand,
            self.cnn_total_stride,
            self.n_bins,
        )

    def _fetch_window_codes(self, chrom: str, start: int, end: int, strand: str) -> np.ndarray:
        self._ensure_sequences_open("window sequence access")
        chrom_len = int(self._chrom_lengths[chrom])
        fetch_start = max(0, start)
        fetch_end = min(end, chrom_len)

        codes = np.full(self.dna_window_len, PAD_CODE, dtype=np.uint8)
        if fetch_end > fetch_start:
            sequence = self.sequences.fetch(chrom, fetch_start, fetch_end).upper()
            fetched = encode_nucleotides(sequence)
            left = fetch_start - start
            codes[left : left + fetched.shape[0]] = fetched

        if strand == "-":
            codes = reverse_complement_codes(codes)
        return np.ascontiguousarray(codes)

    # ------------------------------------------------------------------
    # (gene, cell) pairs
    # ------------------------------------------------------------------

    def _build_valid_genes(self):
        """Genes with a target in at least one track.

        One item is a gene together with ``n_keys`` tracks, so the DNA is read,
        convolved and pushed through the tower once per gene instead of once per
        (gene, track) pair. The per-track work that remains is the description
        encoder and the decoder.
        """
        gene_ids = self.genes["gene_id"].to_numpy()
        if self.tpm:
            has_target = np.zeros(len(gene_ids), dtype=bool)
            for key in self.all_keys:
                has_target |= np.isin(gene_ids, np.asarray(self.tpm_lookup[key].index))
            self.valid_indices = np.flatnonzero(has_target).astype(np.int64).tolist()
        else:
            self.valid_indices = list(range(len(gene_ids)))

        self.logger.info(
            f"{len(self.valid_indices):,} genes with targets x {len(self.all_keys):,} keys "
            f"(n_keys={self.n_keys}, {self.n_cell_chunks} chunk(s)) "
            f"-> {len(self):,} items, {len(self.valid_indices) * len(self.all_keys):,} "
            f"(gene, cell) pairs per epoch"
        )
        if not self.valid_indices:
            raise ValueError(
                f"No genes with targets for {self.targets_path}. "
                "Do the gene ids in the intervals file match the target table?"
            )

    def _stable_gene_seed(self, gene_id: str) -> int:
        h = hashlib.blake2b(
            f"{self.seed}|{self.epoch}|{gene_id}".encode("utf-8"), digest_size=8
        )
        return int.from_bytes(h.digest(), "little") % (2**32)

    def _get_selected_key_indices(self, gene_id: str, chunk_idx: int) -> List[int]:
        """Which tracks this item covers; reshuffled per gene and per epoch."""
        rng = np.random.default_rng(self._stable_gene_seed(gene_id))
        order = rng.permutation(len(self.all_keys))
        start = chunk_idx * self.n_keys
        return order[start : start + self.n_keys].tolist()

    # ------------------------------------------------------------------
    # descriptions
    # ------------------------------------------------------------------

    def _init_descriptions(self, text_tokenizer: str, text_max_seq_len: int):
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer, padding_side="left")
        self.text_max_seq_len = text_max_seq_len
        self.text_data: Dict[str, str] = {}
        self.text_data_keys = set()

        tokenizer_tag = text_tokenizer.replace("/", "_")
        descriptions_dir = Path(__file__).resolve().parent / "descriptions"
        descriptions_dir.mkdir(parents=True, exist_ok=True)
        targets_tag = hashlib.blake2b(
            self._name_and_size(self.targets_path).encode("utf-8"), digest_size=8
        ).hexdigest()
        desc_cache_name = (
            f"{Path(self.targets_path).name}.{targets_tag}.{tokenizer_tag}"
            f".{text_max_seq_len}.description.h5"
        )
        self.desc_h5_cache_path = str(descriptions_dir / desc_cache_name)

        if not os.path.exists(self.desc_h5_cache_path):
            self.load_descriptions_from_json(self.targets_path)
            self.precompute_descriptions()
        self.desc_h5_cache = h5py.File(self.desc_h5_cache_path, "r")

    # ------------------------------------------------------------------
    # bigWig signals, aggregated per bin instead of per BPE token
    # ------------------------------------------------------------------

    def precompute_signals(self):
        self.logger.info(f"Precomputing binned signals to {self.signals_cache_path}")
        temp_path = f"{self.signals_cache_path}.{os.getpid()}.temp"

        try:
            if not self.files_opened:
                self.open_files()

            with h5py.File(temp_path, "w") as h5f:
                pbar = tqdm.tqdm(total=len(self.genes), desc="Computing binned signals")
                for idx in range(len(self.genes)):
                    row = self.genes.iloc[idx]
                    gene_id = row["gene_id"]
                    chrom = row["chromosome"]
                    strand = row["strand"]

                    start, end = self._window_for_gene(row)
                    starts, ends = self._bin_coords(start, end, strand)

                    signals = np.zeros((self.n_bins, len(self.bigWigHandlers)), dtype=np.float32)
                    for i_key, (key, bw_pair) in enumerate(self.bigWigHandlers.items()):
                        track = self.process_region_signals(
                            bw_pair[strand], chrom, starts, ends, self.n_bins, strand
                        )
                        if self.norm_bw:
                            norm_factor = self.coverage_norm.get(key, {}).get(strand, 0)
                            if norm_factor:
                                track = track / norm_factor
                            else:
                                self.logger.warning(
                                    f"Missing/zero normalization factor for {key}, strand {strand}"
                                )
                        signals[:, i_key] = track

                    grp = h5f.create_group(gene_id)
                    grp.create_dataset("signals", data=signals)
                    if idx % 100 == 0:
                        h5f.flush()
                    pbar.update(1)
                pbar.close()
                h5f.flush()

            os.rename(temp_path, self.signals_cache_path)
            self.signals_cache = h5py.File(self.signals_cache_path, "r")
        except Exception as exc:
            self.logger.error(f"Error creating binned signals cache: {exc}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.valid_indices) * self.n_cell_chunks

    def __getitem__(self, idx):
        gene_row = self.valid_indices[idx // self.n_cell_chunks]
        chunk_idx = idx % self.n_cell_chunks

        row = self.genes.iloc[gene_row]
        gene_id = row["gene_id"]
        key_indices = self._get_selected_key_indices(gene_id, chunk_idx)
        selected_keys = [self.all_keys[k] for k in key_indices]
        n_sel = len(selected_keys)
        chrom = row["chromosome"]
        strand = row["strand"]
        reverse = 0 if strand == "+" else 1

        if self.bw and not self.files_opened:
            self.open_files()

        _t = time.perf_counter if self.profile_timing else None
        _t0 = _t() if _t else None

        start, end = self._window_for_gene(row)
        dna_codes = self._fetch_window_codes(chrom, start, end, strand)
        assert dna_codes.shape[0] == self.dna_window_len
        if _t:
            self._timings["dna_fetch"] += _t() - _t0
            _t0 = _t()

        starts, ends = self._bin_coords(start, end, strand)
        chrom_len = int(self._chrom_lengths[chrom])
        bin_valid = (starts >= 0) & (ends <= chrom_len)

        # The DNA is shared by every track of this gene and is returned once;
        # the model broadcasts the tower output over the n_keys rows. Index 0 is
        # CLS (gene-level target), 1..n_bins are the bins, the last position is
        # SEP and is never a target -- as in the token model.
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        attention_mask[1 : 1 + self.n_bins] = torch.from_numpy(bin_valid.astype(np.int64))

        labels = torch.zeros((self.n_keys, self.seq_len, 1), dtype=torch.float32)
        labels_mask = torch.zeros((self.n_keys, self.seq_len, 1), dtype=torch.bool)

        if self.bw:
            if self.signals_cache is not None:
                all_signals = self.signals_cache[gene_id]["signals"]  # (n_bins, n_tracks)
                cols = [self._bw_key_to_col[k] for k in selected_keys]
                signal = np.asarray(all_signals)[:, cols].T.astype(np.float32)  # (n_sel, n_bins)
            else:
                signal = np.stack([
                    self.process_region_signals(
                        self.bigWigHandlers[k][strand], chrom, starts, ends, self.n_bins, strand
                    )
                    for k in selected_keys
                ])
            if self.transform_targets_bw is not None:
                signal = self.transform_targets_bw(signal)
            labels[:n_sel, 1 : 1 + self.n_bins, 0] = torch.from_numpy(
                np.ascontiguousarray(signal)
            )
            labels_mask[:n_sel, 1 : 1 + self.n_bins, 0] = torch.from_numpy(bin_valid)

        if _t:
            self._timings["bins_and_bw"] += _t() - _t0
            _t0 = _t()

        tpm_values = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self.tpm:
            for i, key in enumerate(selected_keys):
                try:
                    tpm_values[i] = float(self.tpm_lookup[key].loc[gene_id].iloc[0])
                except KeyError:
                    pass
            if self.transform_targets_tpm is not None:
                tpm_values = self.transform_targets_tpm(tpm_values)
        tpm_valid = ~np.isnan(tpm_values)
        labels[:, 0, 0] = torch.from_numpy(np.where(tpm_valid, tpm_values, 0.0).astype(np.float32))
        labels_mask[:, 0, 0] = torch.from_numpy(tpm_valid)

        if _t:
            self._timings["tpm_lookup"] += _t() - _t0
            _t0 = _t()

        desc_input_ids, desc_attention_mask = [], []
        for key in selected_keys:
            grp = self.desc_h5_cache[str(key)]
            desc_input_ids.append(torch.tensor(grp["input_ids"][()], dtype=torch.long))
            desc_attention_mask.append(torch.tensor(grp["attention_mask"][()], dtype=torch.long))
        # Pad a short chunk (only possible when n_keys does not divide the track
        # count) by repeating the first entry; its labels_mask is already False.
        while len(desc_input_ids) < self.n_keys:
            desc_input_ids.append(desc_input_ids[0].clone())
            desc_attention_mask.append(desc_attention_mask[0].clone())
            selected_keys.append(selected_keys[0])

        if _t:
            self._timings["desc_h5"] += _t() - _t0
            self._timed_calls += 1

        # The collate builds the description de-duplication index from
        # (dataset_description, key): a track index alone is only unique within
        # one dataset, and a ConcatDataset batch mixes human and mouse, where
        # index 0 denotes different descriptions.
        return {
            "dna_codes": torch.from_numpy(dna_codes),
            "attention_mask": attention_mask,
            "labels": labels,
            "labels_mask": labels_mask,
            "desc_input_ids": desc_input_ids,
            "desc_attention_mask": desc_attention_mask,
            "gene_id": gene_id,
            "selected_keys": selected_keys,
            "dataset_description": self.dataset_description,
            "chrom": chrom,
            "reverse": reverse,
            "start": int(start),
            "end": int(end),
        }

    def timing_report(self) -> str:
        """Mean per-section cost of __getitem__, in ms. Requires profile_timing."""
        if not self._timed_calls:
            return "no timed calls (set profile_timing=True and iterate)"
        total = sum(self._timings.values())
        lines = [f"__getitem__ over {self._timed_calls} calls "
                 f"(n_keys={self.n_keys}, window={self.dna_window_len}bp):"]
        for name, secs in sorted(self._timings.items(), key=lambda kv: -kv[1]):
            ms = 1000 * secs / self._timed_calls
            lines.append(f"  {name:<14} {ms:7.3f} ms  {100 * secs / total:5.1f}%")
        ms_total = 1000 * total / self._timed_calls
        lines.append(f"  {'TOTAL':<14} {ms_total:7.3f} ms -> "
                     f"{1000 / ms_total:.0f} items/s ({1000 * self.n_keys / ms_total:.0f} pairs/s) per worker")
        return "\n".join(lines)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def describe(self):
        return (
            f"ExpressionDatasetCNN(items={len(self):,}, genes={len(self.valid_indices):,}, "
            f"cell_types={len(self.all_keys):,}, n_keys={self.n_keys}, "
            f"chunks={self.n_cell_chunks}, window={self.dna_window_len}bp, "
            f"bins={self.n_bins}/{self.n_bins_full}@{self.cnn_total_stride}bp "
            f"(crop {self.crop_offset}bp per edge), bw={self.bw!r}, tpm={self.tpm!r}, "
            f"dataset_description={getattr(self, 'dataset_description', None)!r})"
        )

    def __del__(self):
        for attr in ("sequences", "signals_cache", "desc_h5_cache"):
            try:
                handle = getattr(self, attr, None)
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        try:
            for per_strand in getattr(self, "bigWigHandlers", {}).values():
                for handle in per_strand.values():
                    try:
                        handle.close()
                    except Exception:
                        pass
        except Exception:
            pass
