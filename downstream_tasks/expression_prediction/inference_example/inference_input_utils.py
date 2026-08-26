import hashlib
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from downstream_tasks.expression_prediction.expression_dataset_final import ExpressionDataset


LOGGER = logging.getLogger(__name__)


def _ensure_tokenizer(tokenizer):
    if isinstance(tokenizer, str):
        kwargs = {"trust_remote_code": True}
        if "qwen" in tokenizer.lower():
            kwargs["padding_side"] = "left"
        return AutoTokenizer.from_pretrained(tokenizer, **kwargs)
    return tokenizer


def _tokenizer_tag(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__
    return str(name).replace("/", "_").replace(os.sep, "_")


def _file_signature(path: Optional[Path]) -> str:
    if path is None:
        return "none"
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


def _hash_string(parts: Iterable[str]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def _pad_to_len(tensor: torch.Tensor, length: int, fill_value):
    if tensor.shape[0] == length:
        return tensor
    output = torch.full((length,) + tensor.shape[1:], fill_value, dtype=tensor.dtype)
    output[: tensor.shape[0]] = tensor
    return output


def _read_intervals_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _display_name_from_row(row: pd.Series, used_names: set) -> str:
    gene_id = str(row["gene_id"])
    gene_name = row.get("gene_name", gene_id)
    if pd.isna(gene_name) or not str(gene_name).strip():
        gene_name = gene_id
    display_name = str(gene_name)
    if display_name in used_names:
        display_name = f"{display_name} ({gene_id})"
    used_names.add(display_name)
    return display_name


def _load_interval_records(
    forward_intervals_path: Path,
    reverse_intervals_path: Optional[Path] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    forward_df = _read_intervals_table(forward_intervals_path)
    forward_df["strand"] = "+"
    frames.append(forward_df)

    if reverse_intervals_path is not None:
        reverse_df = _read_intervals_table(reverse_intervals_path)
        reverse_df["strand"] = "-"
        frames.append(reverse_df)

    genes = pd.concat(frames, ignore_index=True)
    required_columns = {"gene_id", "chromosome", "TSS", "TES", "strand"}
    missing_columns = required_columns - set(genes.columns)
    if missing_columns:
        raise ValueError(
            f"Intervals file is missing required columns: {sorted(missing_columns)}"
        )
    if genes["gene_id"].duplicated().any():
        duplicated = genes.loc[genes["gene_id"].duplicated(), "gene_id"].tolist()[:5]
        raise ValueError(
            "gene_id values must be unique for inference cache creation. "
            f"Examples of duplicates: {duplicated}"
        )
    return genes.reset_index(drop=True)


def _build_gene_records(genes: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    used_names = set()
    for _, row in genes.iterrows():
        display_name = _display_name_from_row(row, used_names)
        records.append(
            {
                "display_name": display_name,
                "gene_id": str(row["gene_id"]),
                "gene_name": None if pd.isna(row.get("gene_name")) else str(row.get("gene_name")),
                "chromosome": str(row["chromosome"]),
                "strand": str(row["strand"]),
                "TSS": int(row["TSS"]),
                "TES": int(row["TES"]),
            }
        )
    return records


def _iter_json_files(json_dir: Path) -> List[Path]:
    json_files = sorted(path for path in json_dir.rglob("*.json") if path.is_file())
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    return json_files


def _experiment_name(json_path: Path, json_dir: Path) -> str:
    relative = json_path.relative_to(json_dir).with_suffix("")
    return str(relative).replace(os.sep, "__")


def _json_folder_signature(json_dir: Path, json_files: List[Path]) -> str:
    total_size = sum(path.stat().st_size for path in json_files)
    return f"{json_dir.name}:{len(json_files)}:{total_size}"


def prepare_descriptions_from_json_dir(
    json_dir: str,
    text_tokenizer,
    text_max_seq_len: int,
    cache_dir: str,
    repeat_to_num_genes: Optional[int] = None,
    dataset_cls=ExpressionDataset,
) -> Tuple[OrderedDict, Dict[str, Dict[str, torch.Tensor]], str]:
    text_tokenizer = _ensure_tokenizer(text_tokenizer)
    json_dir_path = Path(json_dir).expanduser().resolve()
    cache_dir_path = Path(cache_dir).expanduser().resolve()
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    json_files = _iter_json_files(json_dir_path)
    json_signature = _json_folder_signature(json_dir_path, json_files)
    desc_hash = _hash_string(
        [json_signature, _tokenizer_tag(text_tokenizer), str(text_max_seq_len),
         dataset_cls.__name__]
    )
    cache_path = cache_dir_path / (
        f"{json_dir_path.name}.{desc_hash}.{_tokenizer_tag(text_tokenizer)}."
        f"{text_max_seq_len}.description.h5"
    )

    experiments: "OrderedDict[str, str]" = OrderedDict()
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        experiment_name = _experiment_name(json_path, json_dir_path)
        experiments[experiment_name] = dataset_cls.make_description_from_json(
            meta=meta,
            description_id=experiment_name,
            meta_path=str(json_path),
        )

    if not cache_path.exists():
        temp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.temp")
        with h5py.File(temp_path, "w") as h5f:
            h5f.attrs["experiment_order"] = json.dumps(list(experiments.keys()))
            for experiment_name, text in experiments.items():
                encoding = text_tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=text_max_seq_len,
                    return_tensors="pt",
                )
                group = h5f.create_group(experiment_name)
                group.create_dataset("input_ids", data=encoding["input_ids"][0].numpy())
                group.create_dataset("attention_mask", data=encoding["attention_mask"][0].numpy())
            h5f.flush()
        os.replace(temp_path, cache_path)

    tokenized_descriptions: Dict[str, Dict[str, torch.Tensor]] = {}
    with h5py.File(cache_path, "r") as h5f:
        experiment_order = json.loads(h5f.attrs["experiment_order"])
        for experiment_name in experiment_order:
            group = h5f[experiment_name]
            input_ids = torch.tensor(group["input_ids"][()], dtype=torch.long)
            attention_mask = torch.tensor(group["attention_mask"][()], dtype=torch.long)
            if repeat_to_num_genes is not None:
                input_ids = input_ids.unsqueeze(0).repeat(repeat_to_num_genes, 1)
                attention_mask = attention_mask.unsqueeze(0).repeat(repeat_to_num_genes, 1)
            tokenized_descriptions[experiment_name] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    return experiments, tokenized_descriptions, str(cache_path)


def _load_single_interval_records(path: Path, strand: str) -> pd.DataFrame:
    df = _read_intervals_table(path)
    df["strand"] = strand
    return df


def _ensure_unique_gene_ids(genes: pd.DataFrame):
    if genes["gene_id"].duplicated().any():
        duplicated = genes.loc[genes["gene_id"].duplicated(), "gene_id"].tolist()[:5]
        raise ValueError(
            "gene_id values must be unique for inference cache creation. "
            f"Examples of duplicates: {duplicated}"
        )


def _interval_cache_path(
    intervals_path: Path,
    strand_label: str,
    genome: Path,
    gen_tokenizer,
    cache_dir_path: Path,
    num_before: int,
    token_len_for_fetch: int,
) -> Path:
    interval_hash = _hash_string(
        [
            strand_label,
            _file_signature(intervals_path),
            _file_signature(genome),
            _tokenizer_tag(gen_tokenizer),
            str(num_before),
            str(token_len_for_fetch),
        ]
    )
    return cache_dir_path / f"inference_dataset_hash.{strand_label}.{interval_hash}.h5"


def _tokenize_single_interval_file_like_dataset(
    intervals_path: Path,
    strand: str,
    strand_label: str,
    genome: Path,
    gen_tokenizer,
    cache_dir_path: Path,
    num_before: int,
    token_len_for_fetch: int,
    loglevel: int,
    dataset_cls=ExpressionDataset,
) -> Tuple[pd.DataFrame, Path]:
    genes = _load_single_interval_records(intervals_path, strand)
    _ensure_unique_gene_ids(genes)
    required_columns = {"gene_id", "chromosome", "TSS", "TES", "strand"}
    missing_columns = required_columns - set(genes.columns)
    if missing_columns:
        raise ValueError(
            f"Intervals file is missing required columns: {sorted(missing_columns)}"
        )

    cache_path = _interval_cache_path(
        intervals_path=intervals_path,
        strand_label=strand_label,
        genome=genome,
        gen_tokenizer=gen_tokenizer,
        cache_dir_path=cache_dir_path,
        num_before=num_before,
        token_len_for_fetch=token_len_for_fetch,
    )

    if cache_path.exists():
        return genes.reset_index(drop=True), cache_path

    gene_records = _build_gene_records(genes)
    proxy = dataset_cls.__new__(dataset_cls)
    proxy.logger = LOGGER
    proxy.logger.setLevel(loglevel)
    proxy.gen_tokenizer = gen_tokenizer
    proxy.genome = str(genome)
    proxy.num_before = num_before
    proxy.token_len_for_fetch = token_len_for_fetch
    proxy.genes = genes.reset_index(drop=True)
    proxy.sequences = None

    temp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.temp")
    try:
        with h5py.File(temp_path, "w") as h5f:
            h5f.attrs["gene_order"] = json.dumps([record["gene_id"] for record in gene_records])
            for idx, record in enumerate(gene_records):
                gene_id = record["gene_id"]
                _, tokens_df = dataset_cls.tokenize_genome(proxy, idx)
                group = h5f.create_group(gene_id)
                group.create_dataset(
                    "input_ids",
                    data=tokens_df["token_id"].values.astype(np.int32),
                )
                group.create_dataset(
                    "starts",
                    data=tokens_df["start"].values.astype(np.int64),
                )
                group.create_dataset(
                    "ends",
                    data=tokens_df["end"].values.astype(np.int64),
                )
                group.attrs["strand"] = proxy.genes.iloc[idx]["strand"]
                group.attrs["chrom"] = tokens_df["chrom"].iloc[0]
                group.attrs["display_name"] = record["display_name"]
            h5f.flush()
        os.replace(temp_path, cache_path)
    finally:
        if getattr(proxy, "sequences", None) is not None:
            proxy.sequences.close()

    return proxy.genes.reset_index(drop=True), cache_path


def tokenize_interval_genes_like_dataset(
    forward_intervals_path: str,
    genome_path: str,
    gen_tokenizer,
    gen_max_seq_len: int,
    cache_dir: str,
    reverse_intervals_path: Optional[str] = None,
    num_before: int = 512,
    token_len_for_fetch: int = 10,
    loglevel: int = logging.WARNING,
    dataset_cls=ExpressionDataset,
) -> Tuple[OrderedDict, Dict[str, torch.Tensor], Dict[str, str]]:
    gen_tokenizer = _ensure_tokenizer(gen_tokenizer)
    forward_path = Path(forward_intervals_path).expanduser().resolve()
    reverse_path = None if reverse_intervals_path is None else Path(reverse_intervals_path).expanduser().resolve()
    genome = Path(genome_path).expanduser().resolve()
    cache_dir_path = Path(cache_dir).expanduser().resolve()
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    forward_genes, forward_cache_path = _tokenize_single_interval_file_like_dataset(
        intervals_path=forward_path,
        strand="+",
        strand_label="forward",
        genome=genome,
        gen_tokenizer=gen_tokenizer,
        cache_dir_path=cache_dir_path,
        num_before=num_before,
        token_len_for_fetch=token_len_for_fetch,
        loglevel=loglevel,
        dataset_cls=dataset_cls,
    )
    frames = [forward_genes]
    cache_paths: Dict[str, str] = {"forward": str(forward_cache_path)}

    if reverse_path is not None:
        reverse_genes, reverse_cache_path = _tokenize_single_interval_file_like_dataset(
            intervals_path=reverse_path,
            strand="-",
            strand_label="reverse",
            genome=genome,
            gen_tokenizer=gen_tokenizer,
            cache_dir_path=cache_dir_path,
            num_before=num_before,
            token_len_for_fetch=token_len_for_fetch,
            loglevel=loglevel,
            dataset_cls=dataset_cls,
        )
        frames.append(reverse_genes)
        cache_paths["reverse"] = str(reverse_cache_path)

    genes = pd.concat(frames, ignore_index=True)
    _ensure_unique_gene_ids(genes)
    gene_records = _build_gene_records(genes)
    gene_id_to_record = {record["gene_id"]: record for record in gene_records}

    cls_id = gen_tokenizer.cls_token_id or gen_tokenizer.bos_token_id
    sep_id = gen_tokenizer.sep_token_id or gen_tokenizer.eos_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("DNA tokenizer must define CLS/BOS and SEP/EOS tokens")
    pad_id = gen_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    seqs: List[torch.Tensor] = []
    attns: List[torch.Tensor] = []
    toktypes: List[torch.Tensor] = []
    display_order: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    for strand_label in ("forward", "reverse"):
        cache_path_str = cache_paths.get(strand_label)
        if cache_path_str is None:
            continue
        with h5py.File(cache_path_str, "r") as h5f:
            gene_order = json.loads(h5f.attrs["gene_order"])
            for gene_id in gene_order:
                group = h5f[gene_id]
                n_tokens = group["input_ids"].shape[0]
                token_count = min(n_tokens, gen_max_seq_len - 2)
                if token_count <= 0:
                    raise ValueError(f"Empty token sequence for gene_id={gene_id}")

                raw_ids = torch.tensor(group["input_ids"][:token_count], dtype=torch.long)
                seq = torch.cat(
                    [torch.tensor([cls_id], dtype=torch.long), raw_ids, torch.tensor([sep_id], dtype=torch.long)],
                    dim=0,
                )
                attn = torch.ones(seq.shape[0], dtype=torch.long)
                toktypes.append(torch.zeros(seq.shape[0], dtype=torch.long))
                seqs.append(seq)
                attns.append(attn)
                record = dict(gene_id_to_record[gene_id])
                display_order[record["display_name"]] = record

    max_len = max(seq.shape[0] for seq in seqs)
    tokenized_dna = {
        "input_ids": torch.stack([_pad_to_len(seq, max_len, pad_id) for seq in seqs], dim=0),
        "attention_mask": torch.stack([_pad_to_len(attn, max_len, 0) for attn in attns], dim=0),
        "token_type_ids": torch.stack([_pad_to_len(tok, max_len, 0) for tok in toktypes], dim=0),
    }

    return display_order, tokenized_dna, cache_paths


def prepare_inference_inputs_from_intervals(
    json_dir: str,
    forward_intervals_path: str,
    genome_path: str,
    gen_tokenizer,
    text_tokenizer,
    gen_max_seq_len: int,
    text_max_seq_len: int,
    cache_dir: str,
    reverse_intervals_path: Optional[str] = None,
    num_before: int = 512,
    token_len_for_fetch: int = 10,
    loglevel: int = logging.WARNING,
    dataset_cls=ExpressionDataset,
) -> Dict[str, Any]:
    genes, tokenized_dna, gene_cache_paths = tokenize_interval_genes_like_dataset(
        forward_intervals_path=forward_intervals_path,
        reverse_intervals_path=reverse_intervals_path,
        genome_path=genome_path,
        gen_tokenizer=gen_tokenizer,
        gen_max_seq_len=gen_max_seq_len,
        cache_dir=cache_dir,
        num_before=num_before,
        token_len_for_fetch=token_len_for_fetch,
        loglevel=loglevel,
        dataset_cls=dataset_cls,
    )

    experiments, tokenized_descriptions, description_cache_path = prepare_descriptions_from_json_dir(
        json_dir=json_dir,
        text_tokenizer=text_tokenizer,
        text_max_seq_len=text_max_seq_len,
        cache_dir=cache_dir,
        repeat_to_num_genes=len(genes),
        dataset_cls=dataset_cls,
    )

    return {
        "genes": genes,
        "experiments": experiments,
        "tokenized_DNA": tokenized_dna,
        "tokenized_descriptions": tokenized_descriptions,
        "gene_cache_path": gene_cache_paths.get("forward"),
        "gene_cache_paths": gene_cache_paths,
        "description_cache_path": description_cache_path,
    }
