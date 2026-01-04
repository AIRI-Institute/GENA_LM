#!/usr/bin/env python
import logging
import os
from pathlib import Path
import shutil
from argparse import ArgumentParser

import torch
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to the experiment config")
parser.add_argument("--log_level", type=int, default=logging.INFO, help="log level")
parser.add_argument("--output_dir", type=str, default=None, help="override output directory")


import torch
import torch.distributed as dist

def verify_disjoint_sampler_indices(train_dataloader, k=200):
    """
    Collect first k indices from the sampler on each rank and check overlaps.
    Works when dataloader.sampler is a DistributedSampler.
    """
    sampler = train_dataloader.sampler
    if not hasattr(sampler, "__iter__"):
        print("Sampler has no __iter__; cannot verify.")
        return

    local = list(iter(sampler))[:k]
    local_set = set(int(x) for x in local)

    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_set)

        # only rank 0 reports
        if dist.get_rank() == 0:
            overlaps = []
            for i in range(len(gathered)):
                for j in range(i + 1, len(gathered)):
                    inter = gathered[i] & gathered[j]
                    overlaps.append(len(inter))
            print(f"[SamplerCheck] first {k} indices overlap counts (pairwise): {overlaps}")
            print(f"[SamplerCheck] per-rank sizes: {[len(s) for s in gathered]}")
    else:
        print("Not in distributed mode; skipping disjointness check.")



def main():
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=args.log_level,
    )
    logger = logging.getLogger()

    experiment_config_path = Path(args.config).expanduser().absolute()

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))

    # Distributed env (set by accelerate launch / torchrun)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")
    logger.info(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")

    # Pin the current process to the right GPU (important when multiple processes)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    with initialize_config_dir(str(experiment_config_path.parents[0]), version_base=None):
        cfg = compose(config_name=experiment_config_path.name)

    output_dir = args.output_dir or cfg.output_dir
    logger.info(f"Output directory: {output_dir}")

    # Only rank 0 does filesystem writes
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    # Instantiate Trainer (HF Trainer distributed logic will kick in automatically under accelerate launch)
    trainer_config = cfg.trainer.copy()
    trainer = instantiate(trainer_config)

    # Sanity check: ensure sampler is distributed and dataloader length scales with WORLD_SIZE
    try:
        dl = trainer.get_train_dataloader()
        verify_disjoint_sampler_indices(dl, k=200)
        sampler_name = type(getattr(dl, "sampler", None)).__name__
        logger.info(f"train_dataloader sampler: {sampler_name}")
        logger.info(f"len(train_dataset)={len(trainer.train_dataset)} len(train_dataloader)={len(dl)}")
    except Exception as e:
        logger.warning(f"Could not inspect train_dataloader: {e}")

    resume_from_checkpoint = cfg.get("resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.evaluate()


if __name__ == "__main__":
    main()
