#!/usr/bin/env python3
"""Pre-tokenize datasets concurrently with cache race-condition handling."""
import logging
import os
import threading
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
import torch
import numpy

# PyTorch 2.6+ safe globals fix
torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType
])

# Lock registry: one lock per cache_dir to serialize cache creation
_cache_locks = {}
_locks_mutex = threading.Lock()


def _get_cache_lock(cache_dir: str):
    """Get or create a thread lock for a specific cache directory."""
    with _locks_mutex:
        if cache_dir not in _cache_locks:
            _cache_locks[cache_dir] = threading.Lock()
        return _cache_locks[cache_dir]


def _collect_dataset_configs(cfg, prefix: str):
    """Collect dataset configs matching prefix."""
    return [v for k, v in cfg.items() if str(k).startswith(prefix)]


def _warmup_dataset(dataset_cfg, name: str, logger):
    """Instantiate dataset and trigger caching with proper locking."""
    pid = os.getpid()
    tname = threading.current_thread().name
    logger.info(f"[PID:{pid}][{tname}] Starting {name}")
    
    # Extract cache_dir for locking (adjust key name if your config differs)
    cache_dir = dataset_cfg.get('cache_dir', '.')
    lock = _get_cache_lock(str(cache_dir))
    
    with lock:
        dataset = instantiate(dataset_cfg)
        # Trigger tokenization: __len__ often forces lazy loading/caching
        _ = len(dataset)
    
    logger.info(f"[PID:{pid}][{tname}] Finished {name}")
    return name


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='experiment config path')
    parser.add_argument('--log_level', type=int, default=logging.INFO)
    parser.add_argument('--max_workers', type=int, default=10, help='concurrent threads')
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(asctime)s - [PID:%(process)d][%(threadName)s] - %(levelname)s - %(message)s',
        level=args.log_level
    )
    logger = logging.getLogger(__name__)
    
    cfg_path = Path(args.config).resolve()
    with initialize_config_dir(str(cfg_path.parent)):
        exp_cfg = compose(config_name=cfg_path.name)
    
    # Gather all dataset configs
    datasets = []
    for pfx in ['train_dataset', 'valid_dataset']:
        for i, cfg in enumerate(_collect_dataset_configs(exp_cfg, pfx)):
            datasets.append((cfg, f"{pfx}_{i}"))
    
    if not datasets:
        logger.info("No datasets found to tokenize")
        return
    
    logger.info(f"Warming up {len(datasets)} datasets with {args.max_workers} threads")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(_warmup_dataset, cfg, name, logger)
            for cfg, name in datasets
        ]
        for future in as_completed(futures):
            future.result()  # Propagate exceptions
    
    logger.info("All dataset tokenization completed")


if __name__ == "__main__":
    main()