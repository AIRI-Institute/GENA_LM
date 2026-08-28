import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm
from Bio import SeqIO
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genalg import (
    MaxCriterion,
    ScoredSeq,
    cross,
    get_default_mut_mask,
    get_probs,
    get_score,
    mix,
    remove_duplicates,
    select_lowk_and_soft,
    select_parents,
    single_point_mutate,
)
from lm_experiments_tools.utils import get_cls_by_name
from utils import calc_ident


DEFAULT_TARGETS = [
    "McKellar2021_Myonuclei_Type_IIb",
    "McKellar2021_Myonuclei_Type_IIx",
]

DEFAULT_OFF_TARGETS = [
    "McKellar2021_Endothelial_Artery",
    "McKellar2021_Endothelial_Capillary",
    "McKellar2021_Endothelial_Vein",
    "McKellar2021_M2_Macro._Cx3cr1_hi",
    "McKellar2021_M2_Macro._Cx3cr1_lo",
    "McKellar2021_Smooth_Muscle_&_Pericytes",
    "McKellar2021_MuSCs",
    "McKellar2021_FAPs_Adipogenic",
    "McKellar2021_FAPs_Pro-remodeling",
    "McKellar2021_FAPs_Stem",
]


def load_model(experiment_config_path, device, ckpt="model_best", checkpoint_path=None):
    experiment_config_path = Path(experiment_config_path).expanduser().absolute()

    with initialize_config_dir(str(experiment_config_path.parent), version_base=None):
        experiment_config = compose(
            config_name=experiment_config_path.name,
            overrides=["args_params.model_path=null"],
        )

    model_kwargs = instantiate(experiment_config["model_kwargs"])
    model_cls = get_cls_by_name(experiment_config["args_params"]["model_cls"])
    model = model_cls(**model_kwargs)

    if checkpoint_path is None:
        checkpoint_path = experiment_config_path.parent / ckpt / "pytorch_model.bin"
    checkpoint_path = Path(checkpoint_path).expanduser().absolute()

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
    if missing_keys:
        logging.warning("Missing keys in state_dict: %s", missing_keys)
    if unexpected_keys:
        logging.warning("Unexpected keys in state_dict: %s", unexpected_keys)

    dna_tokenizer_name = experiment_config["args_params"]["gen_tokenizer"]
    text_tokenizer_name = experiment_config["shared_dataset_params"]["text_tokenizer"]
    dna_tokenizer = AutoTokenizer.from_pretrained(dna_tokenizer_name)
    text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name, padding_side="left")
    text_max_seq_len = int(experiment_config["shared_dataset_params"].get("text_max_seq_len", 510))

    model = model.to(device).eval()
    return model, dna_tokenizer, text_tokenizer, text_max_seq_len


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_initial_sequences(intervals_path, genome_path):
    initial_pool = pd.read_csv(intervals_path, sep="\t")
    genome = SeqIO.to_dict(SeqIO.parse(genome_path, format="fasta"))

    seqs = []
    for _, row in tqdm.tqdm(initial_pool.iterrows(), total=initial_pool.shape[0], desc="read genome"):
        seq = genome[row.chrom][row.TSS - 200 : row.TSS + 20].upper()
        seqs.append(str(seq.seq))
    return seqs


def progress_writer(progress_path, progress_queue):
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "a", encoding="utf-8") as handle:
        while True:
            event = progress_queue.get()
            if event is None:
                break
            event.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()


def emit_progress(progress_queue, **event):
    if progress_queue is not None:
        progress_queue.put(event)


def score_sequences_worker(rank, device_id, seqs, cfg, out_queue, progress_queue):
    device = f"cuda:{device_id}"
    seed_everything(cfg.seed + rank)
    emit_progress(
        progress_queue,
        stage="initial_scoring",
        event="start",
        worker_rank=rank,
        device=device_id,
        total=len(seqs),
    )
    model, tokenizer, text_tokenizer, text_max_seq_len = load_model(
        experiment_config_path=cfg.experiment_config_path,
        device=device,
        checkpoint_path=cfg.checkpoint_path,
    )
    criterion = MaxCriterion(
        maxes={key: 1 for key in cfg.selected_keys},
        targets=cfg.targets,
        off_targets=cfg.off_targets,
    )

    scored = []
    for i, seq in enumerate(tqdm.tqdm(seqs, desc=f"score init gpu{device_id}", position=rank), start=1):
        scored.append(
            get_score(
                model=model,
                tokenizer=tokenizer,
                seq=seq,
                criterion=criterion,
                device=device,
                label="genome",
                selected_keys=cfg.selected_keys,
                description_source=cfg.description_source,
                text_tokenizer=text_tokenizer,
                text_max_seq_len=text_max_seq_len,
            )
        )
        if i == len(seqs) or i % cfg.progress_each == 0:
            best_score = min(item.score for item in scored)
            emit_progress(
                progress_queue,
                stage="initial_scoring",
                event="progress",
                worker_rank=rank,
                device=device_id,
                done=i,
                total=len(seqs),
                best_score=best_score,
            )
    emit_progress(
        progress_queue,
        stage="initial_scoring",
        event="done",
        worker_rank=rank,
        device=device_id,
        total=len(seqs),
        best_score=min(item.score for item in scored) if scored else None,
    )
    out_queue.put((rank, scored))


def run_generation(worker_rank, device_id, gen_ids, genome_pool_sc, cfg, out_queue, progress_queue):
    device = f"cuda:{device_id}"
    seed_everything(cfg.seed + 1000 + worker_rank)
    emit_progress(
        progress_queue,
        stage="generation",
        event="worker_start",
        worker_rank=worker_rank,
        device=device_id,
        gen_ids=gen_ids,
    )
    model, tokenizer, text_tokenizer, text_max_seq_len = load_model(
        experiment_config_path=cfg.experiment_config_path,
        device=device,
        checkpoint_path=cfg.checkpoint_path,
    )
    criterion = MaxCriterion(
        maxes={key: 1 for key in cfg.selected_keys},
        targets=cfg.targets,
        off_targets=cfg.off_targets,
    )
    temps = torch.linspace(cfg.max_temperature, cfg.min_temperature, steps=cfg.n_iters)
    end_seqs = []

    for gen_id in gen_ids:
        gen_started_at = time.time()
        storage = {}
        mutated = {}
        cur_genome_pool = list(genome_pool_sc)
        emit_progress(
            progress_queue,
            stage="generation",
            event="gen_start",
            worker_rank=worker_rank,
            device=device_id,
            gen_id=gen_id,
            n_iters=cfg.n_iters,
        )

        genome_probs = get_probs(torch.FloatTensor([x.score for x in cur_genome_pool]), T=temps[0])
        n_take = min(cfg.population_size, len(cur_genome_pool))
        take_ind = torch.multinomial(genome_probs, n_take).tolist()
        taken = [cur_genome_pool[i] for i in take_ind]
        not_take = set(range(len(cur_genome_pool))) - set(take_ind)
        cur_genome_pool = [cur_genome_pool[i] for i in not_take]

        population = taken
        for p in population:
            storage[p.seq] = p.score

        losses = []
        best_par = min(population, key=lambda p: p.score)

        progress = tqdm.tqdm(
            range(cfg.n_iters),
            desc=f"gen {gen_id} gpu{device_id}",
            position=worker_rank,
            leave=False,
        )
        for i in progress:
            losses.append(min(p.score for p in population))
            if i % cfg.report_each == 0:
                ident = calc_ident(genome_pool_sc[0].seq, best_par.seq)
                print(f"[gpu {device_id} gen {gen_id} iter {i}] {losses[-1]} ident={ident} {best_par.dt}")
                emit_progress(
                    progress_queue,
                    stage="generation",
                    event="progress",
                    worker_rank=worker_rank,
                    device=device_id,
                    gen_id=gen_id,
                    iter=i,
                    n_iters=cfg.n_iters,
                    best_score=float(best_par.score),
                    current_min_score=float(losses[-1]),
                    population_size=len(population),
                    storage_size=len(storage),
                    ident=float(ident),
                    elapsed_sec=round(time.time() - gen_started_at, 2),
                )

            par1, par2 = select_parents(population, T=temps[i])
            child = mix(par1, par2)
            if child not in storage:
                child = get_score(
                    model=model,
                    tokenizer=tokenizer,
                    seq=child,
                    criterion=criterion,
                    device=device,
                    label="mix",
                    selected_keys=cfg.selected_keys,
                    description_source=cfg.description_source,
                    text_tokenizer=text_tokenizer,
                    text_max_seq_len=text_max_seq_len,
                )
                storage[child.seq] = child.score
                population.append(child)

            par1, par2 = select_parents(population, T=temps[i])
            child = cross(par1, par2, mean_length=cfg.mean_motif_length)
            if child not in storage:
                child = get_score(
                    model=model,
                    tokenizer=tokenizer,
                    seq=child,
                    criterion=criterion,
                    device=device,
                    label="cross",
                    selected_keys=cfg.selected_keys,
                    description_source=cfg.description_source,
                    text_tokenizer=text_tokenizer,
                    text_max_seq_len=text_max_seq_len,
                )
                storage[child.seq] = child.score
                population.append(child)

            best_par = min(population, key=lambda p: p.score)
            if best_par.seq not in mutated:
                mutated[best_par.seq] = get_default_mut_mask(cfg.required_length)
            if len(mutated[best_par.seq]) == 0:
                print(f"[gpu {device_id} gen {gen_id}] END of generation")
                emit_progress(
                    progress_queue,
                    stage="generation",
                    event="early_stop",
                    worker_rank=worker_rank,
                    device=device_id,
                    gen_id=gen_id,
                    iter=i,
                    best_score=float(best_par.score),
                    elapsed_sec=round(time.time() - gen_started_at, 2),
                )
                break

            mut = single_point_mutate(best_par.seq, mutated[best_par.seq])
            if mut not in storage:
                mut = get_score(
                    model=model,
                    tokenizer=tokenizer,
                    seq=mut,
                    criterion=criterion,
                    device=device,
                    label="snp",
                    selected_keys=cfg.selected_keys,
                    description_source=cfg.description_source,
                    text_tokenizer=text_tokenizer,
                    text_max_seq_len=text_max_seq_len,
                )
                storage[mut.seq] = mut.score
                population.append(mut)

            if i != 0 and i % cfg.remove_parents_each == 0:
                population = remove_duplicates(population)
                population = select_lowk_and_soft(population, k=cfg.population_size, T=temps[i])

            if i != 0 and i % cfg.add_next_portion == 0 and len(cur_genome_pool) != 0:
                n_take = min(cfg.population_size, len(cur_genome_pool))
                if len(cur_genome_pool) <= n_take:
                    taken = cur_genome_pool
                    cur_genome_pool = []
                else:
                    genome_probs = get_probs(torch.FloatTensor([x.score for x in cur_genome_pool]), T=temps[i])
                    take_ind = torch.multinomial(genome_probs, n_take).tolist()
                    taken = [cur_genome_pool[i] for i in take_ind]
                    not_take = set(range(len(cur_genome_pool))) - set(take_ind)
                    cur_genome_pool = [cur_genome_pool[i] for i in not_take]
                population.extend(taken)

        final_best = min(population, key=lambda p: p.score)
        end_seqs.append(final_best)
        emit_progress(
            progress_queue,
            stage="generation",
            event="gen_done",
            worker_rank=worker_rank,
            device=device_id,
            gen_id=gen_id,
            best_score=float(final_best.score),
            method=final_best.method,
            elapsed_sec=round(time.time() - gen_started_at, 2),
        )

    emit_progress(
        progress_queue,
        stage="generation",
        event="worker_done",
        worker_rank=worker_rank,
        device=device_id,
        n_finished=len(end_seqs),
    )
    out_queue.put((worker_rank, end_seqs))


def split_even(items, n_parts):
    return [items[i::n_parts] for i in range(n_parts)]


def collect_queue(queue, processes):
    chunks = []
    expected = len(processes)
    while len(chunks) < expected:
        try:
            chunks.append(queue.get(timeout=30))
        except Exception:
            failed = [process.exitcode for process in processes if process.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker failed before returning results: exit codes {failed}")
    return [chunk for _, chunk in sorted(chunks, key=lambda x: x[0])]


def write_outputs(end_seqs, out_prefix):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, item in enumerate(end_seqs):
        row = {
            "rank": i,
            "sequence": item.seq,
            "score": item.score,
            "method": item.method,
        }
        row.update(item.dt)
        rows.append(row)

    generated_df = pd.DataFrame(rows)
    csv_path = out_prefix.with_suffix(".csv")
    fasta_path = out_prefix.with_suffix(".fa")
    generated_df.to_csv(csv_path, index=False)

    with open(fasta_path, "w") as handle:
        for i, item in enumerate(end_seqs):
            handle.write(f">generated_{i}_score_{item.score:.4f}_method_{item.method}\n")
            handle.write(item.seq + "\n")

    return csv_path, fasta_path


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel multi-GPU expression sequence generation.")
    parser.add_argument("--gena-home", default="/home/jovyan/shares/SR003.nfs2/aspeedok/GENA_LM")
    parser.add_argument("--genalm-home", default="/home/jovyan/shares/SR003.nfs2/aspeedok")
    parser.add_argument(
        "--experiment-config-path",
        default="/home/jovyan/shares/SR003.nfs2/aspeedok/GENA_LM/downstream_tasks/expression_prediction/configs/final_02062026.yaml",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="/home/jovyan/shares/SR003.nfs2/aspeedok/runs/mikhail_experements/super_model_02.06_res2/pytorch_model.bin",
    )
    parser.add_argument(
        "--description-source",
        default="/home/jovyan/shares/SR003.nfs2/aspeedok/GENA_LM/downstream_tasks/expression_prediction/datasets/data/scRNA_McKellar2021_Smooth_Muscle/metadata",
    )
    parser.add_argument(
        "--intervals-path",
        default="/home/jovyan/shares/SR003.nfs2/aspeedok/GENA_LM/downstream_tasks/expression_prediction/intervals/human.test.forward.csv",
    )
    parser.add_argument(
        "--genome-path",
        default="/home/jovyan/shares/SR003.nfs2/aspeedok/GENA_LM/downstream_tasks/expression_prediction/datasets/data/genomes/hg38/hg38.fa",
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6")
    parser.add_argument("--n-gens", type=int, default=100)
    parser.add_argument("--n-iters", type=int, default=20000)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--report-each", type=int, default=500)
    parser.add_argument("--remove-parents-each", type=int, default=100)
    parser.add_argument("--add-next-portion", type=int, default=100)
    parser.add_argument("--max-temperature", type=float, default=15.0)
    parser.add_argument("--min-temperature", type=float, default=0.010)
    parser.add_argument("--mean-motif-length", type=int, default=12)
    parser.add_argument("--required-length", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="generated_muscle_miofibers_vs_others_parallel")
    parser.add_argument(
        "--progress-path",
        default=None,
        help="JSONL progress log path. Defaults to <out-prefix>.progress.jsonl.",
    )
    parser.add_argument(
        "--progress-each",
        type=int,
        default=100,
        help="Initial scoring progress interval. Generation progress uses --report-each.",
    )
    args = parser.parse_args()

    args.targets = DEFAULT_TARGETS
    args.off_targets = DEFAULT_OFF_TARGETS
    args.selected_keys = args.targets + args.off_targets
    args.devices = [int(device) for device in args.devices.split(",") if device.strip()]
    if args.progress_path is None:
        args.progress_path = str(Path(args.out_prefix).with_suffix(".progress.jsonl"))
    return args


def main():
    args = parse_args()
    os.environ["GENALM_HOME"] = args.genalm_home
    seed_everything(args.seed)

    mp.set_start_method("spawn", force=True)
    progress_queue = mp.Queue()
    progress_process = mp.Process(target=progress_writer, args=(args.progress_path, progress_queue))
    progress_process.start()
    emit_progress(
        progress_queue,
        stage="run",
        event="start",
        devices=args.devices,
        n_gens=args.n_gens,
        n_iters=args.n_iters,
        out_prefix=args.out_prefix,
    )

    initial_seqs = load_initial_sequences(args.intervals_path, args.genome_path)
    emit_progress(progress_queue, stage="run", event="loaded_initial_sequences", total=len(initial_seqs))
    score_queue = mp.Queue()
    score_shards = split_even(initial_seqs, len(args.devices))
    score_processes = []
    for rank, (device_id, shard) in enumerate(zip(args.devices, score_shards)):
        process = mp.Process(
            target=score_sequences_worker,
            args=(rank, device_id, shard, args, score_queue, progress_queue),
        )
        process.start()
        score_processes.append(process)

    scored_chunks = collect_queue(score_queue, score_processes)
    for process in score_processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Initial scoring worker failed with exit code {process.exitcode}")
    genome_pool_sc = [item for chunk in scored_chunks for item in chunk]
    emit_progress(
        progress_queue,
        stage="run",
        event="initial_scoring_done",
        total=len(genome_pool_sc),
        best_score=min(item.score for item in genome_pool_sc) if genome_pool_sc else None,
    )

    gen_ids = list(range(args.n_gens))
    gen_shards = split_even(gen_ids, len(args.devices))
    gen_queue = mp.Queue()
    gen_processes = []
    for rank, (device_id, shard) in enumerate(zip(args.devices, gen_shards)):
        process = mp.Process(
            target=run_generation,
            args=(rank, device_id, shard, genome_pool_sc, args, gen_queue, progress_queue),
        )
        process.start()
        gen_processes.append(process)

    end_chunks = collect_queue(gen_queue, gen_processes)
    for process in gen_processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Generation worker failed with exit code {process.exitcode}")

    end_seqs = [item for chunk in end_chunks for item in chunk]
    end_seqs = sorted(end_seqs, key=lambda item: item.score)
    csv_path, fasta_path = write_outputs(end_seqs, args.out_prefix)
    emit_progress(
        progress_queue,
        stage="run",
        event="done",
        csv_path=str(csv_path),
        fasta_path=str(fasta_path),
        best_score=float(end_seqs[0].score) if end_seqs else None,
    )
    progress_queue.put(None)
    progress_process.join()
    print(f"Wrote {csv_path}")
    print(f"Wrote {fasta_path}")
    print(f"Wrote {args.progress_path}")


if __name__ == "__main__":
    main()
