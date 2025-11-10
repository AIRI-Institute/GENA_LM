#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Optional, List, Tuple

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate

# ---- NEW: pysam для прединдексации FASTA
try:
    import pysam  # type: ignore
    _HAS_PYSAM = True
except Exception:
    _HAS_PYSAM = False

LOG_FMT = "%(asctime)s | %(levelname)s | pid=%(process)d | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("precompute")

def _find_repo_root(start: Path) -> Optional[Path]:
    for p in [start] + list(start.parents):
        if (p / "downstream_tasks").is_dir():
            return p
    return None

_THIS_FILE = Path(__file__).resolve()
_repo = _find_repo_root(_THIS_FILE.parent) or _find_repo_root(Path.cwd().resolve())

if _repo is not None:
    repo_str = str(_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.environ["PYTHONPATH"] = (repo_str if not os.environ.get("PYTHONPATH") else f"{repo_str}:{os.environ['PYTHONPATH']}")
    log.info(f"Repo root detected and added to PYTHONPATH: {_repo}")
else:
    log.warning("Could not auto-detect repo root containing 'downstream_tasks/'. "
                "Set PYTHONPATH manually or install the project with `pip install -e <repo>`.")

for i, p in enumerate(sys.path[:5], 1):
    log.debug(f"sys.path[{i}]: {p}")

_home_default = os.environ.get("HOME_PATH", os.path.expanduser("~"))
from omegaconf import OmegaConf as _OC
_OC.register_new_resolver("HOME_PATH", lambda: _home_default)

def merge_default_params_with_dataset_config(dataset_config, default_params):
    def _merge_nested_dicts(target, source):
        for k, v in source.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                _merge_nested_dicts(target[k], v)
            elif k not in target:
                target[k] = v if not isinstance(v, dict) else v.copy()

    merged = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
    if default_params is not None:
        default_params_dict = OmegaConf.to_container(default_params, resolve=True)
        _merge_nested_dicts(merged, default_params_dict)
    return merged

def collect_dataset_cfgs(exp_cfg):
    ds_train = [v for k, v in exp_cfg.items() if str(k).startswith("train_dataset")]
    ds_valid = [v for k, v in exp_cfg.items() if str(k).startswith("valid_dataset")]
    shared = exp_cfg.get("shared_dataset_params", None)
    return ds_train, ds_valid, shared

def _short_ds_name_from_cfg_dict(cfg_dict):
    tpath = (cfg_dict.get("targets_path") or "").strip()
    genome = (cfg_dict.get("genome") or "").strip()
    base = Path(tpath).name if tpath else "unknown_targets.csv"
    if genome:
        try:
            gstem = Path(genome).stem
        except Exception:
            gstem = genome
        return f"{base}:{gstem}"
    return base

def _init_one_dataset(cfg_dict):
    t0 = time.time()
    name = _short_ds_name_from_cfg_dict(cfg_dict)

    # ограничим треды BLAS в воркерах
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if _repo is not None:
        repo_str = str(_repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        os.environ["PYTHONPATH"] = (repo_str if not os.environ.get("PYTHONPATH") else f"{repo_str}:{os.environ['PYTHONPATH']}")

    wlog = logging.getLogger(f"precompute.worker[{name}]")
    for h in list(wlog.handlers):
        wlog.removeHandler(h)
    wlog.setLevel(logging.INFO)

    try:
        wlog.info("initialization started")
        logging.getLogger().setLevel(logging.ERROR)

        cfg = OmegaConf.create(cfg_dict)
        ds = hydra_instantiate(cfg)
        _ = len(ds)
        del ds

        secs = time.time() - t0
        wlog.info(f"initialization finished in {secs:.2f}s")
        return {"name": name, "ok": True, "secs": secs, "error": None}
    except Exception as e:
        secs = time.time() - t0
        tb = traceback.format_exc(limit=20)
        wlog.error(f"FAILED in {secs:.2f}s: {e}\n{tb}")
        return {"name": name, "ok": False, "secs": secs, "error": f"{e}\n{tb}"}

def _setup_root_logging(log_file: Optional[str], verbose: bool):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(LOG_FMT))
    root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(LOG_FMT))
        root.addHandler(fh)

# ---- NEW: безопасная прединдексация всех FASTA (последовательно)
def _ensure_fai(fa_path: str):
    fa_path = str(Path(fa_path).expanduser().absolute())
    if not os.path.exists(fa_path):
        raise FileNotFoundError(f"Genome FASTA not found: {fa_path}")
    if not _HAS_PYSAM:
        log.warning("pysam not available; skipping faidx prebuild (may be racy).")
        return
    fai = fa_path + ".fai"
    if os.path.exists(fai) and os.path.getsize(fai) > 0:
        return
    # на всякий случай удалим битый индекс
    if os.path.exists(fai):
        try:
            os.remove(fai)
        except Exception:
            pass
    log.info(f"[FAIDX] Building index for {fa_path}")
    pysam.faidx(fa_path)
    # sanity check: откроем/закроем
    from pysam import FastaFile  # type: ignore
    ff = FastaFile(fa_path)
    refs = ff.references  # noqa: F841 (провоцируем фактическое чтение индекса)
    ff.close()

def main():
    parser = argparse.ArgumentParser(description="Parallel precompute of ExpressionDataset caches")
    parser.add_argument("--experiment_config", required=True, type=str, help="Path to Hydra experiment yaml")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes (default: ~half of CPUs)")
    parser.add_argument("--include_valid", action="store_true", help="Also warm up valid_* datasets")
    parser.add_argument("--n_keys", type=int, default=13, help="Force n_keys for all datasets (default: 13)")
    parser.add_argument("--log_file", type=str, default=None, help="Additionally write logs to a file")
    parser.add_argument("-v", "--verbose", action="store_true", help="More detailed logs (DEBUG)")
    args = parser.parse_args()

    _setup_root_logging(args.log_file, args.verbose)

    exp_cfg_path = Path(args.experiment_config).expanduser().absolute()
    if not exp_cfg_path.exists():
        raise FileNotFoundError(exp_cfg_path)

    cpu_count = os.cpu_count() or 2
    home_path_env = os.environ.get("HOME_PATH") or os.path.expanduser("~")
    log.info(f"experiment_config = {exp_cfg_path}")
    log.info(f"HOME_PATH = {home_path_env}")
    log.info(f"CPUs total = {cpu_count}")
    if _repo is not None:
        log.info(f"PYTHONPATH includes repo: {_repo}")
    else:
        log.warning("PYTHONPATH may miss repo root; imports might fail.")
    if args.log_file:
        log.info(f"logging to file: {args.log_file}")
    log.info(f"Forced n_keys = {args.n_keys}")

    with initialize_config_dir(str(exp_cfg_path.parent), version_base=None):
        exp_cfg = compose(config_name=exp_cfg_path.name)

    exp_cfg = OmegaConf.merge(OmegaConf.create({"HOME_PATH": home_path_env}), exp_cfg)
    log.info(f"Injected HOME_PATH into config: {home_path_env}")

    ds_train_cfgs, ds_valid_cfgs, shared = collect_dataset_cfgs(exp_cfg)
    log.info(f"found train datasets: {len(ds_train_cfgs)}, valid datasets: {len(ds_valid_cfgs)} (include_valid={args.include_valid})")

    merged_train = [merge_default_params_with_dataset_config(c, shared) for c in ds_train_cfgs]
    merged_valid = [merge_default_params_with_dataset_config(c, shared) for c in ds_valid_cfgs] if args.include_valid else []

    def finalize_cfgs(cfgs):
        out = []
        for cfg in cfgs:
            OmegaConf.update(cfg, "n_keys", int(args.n_keys), force_add=True)
            OmegaConf.update(cfg, "loglevel", logging.ERROR, force_add=True)
            out.append(cfg)
        return out

    merged_train = finalize_cfgs(merged_train)
    merged_all = merged_train + (finalize_cfgs(merged_valid) if merged_valid else [])

    if not merged_all:
        log.error("No datasets to initialize — exiting.")
        return

    # ---- NEW: собрать все genome и прединдексировать .fai последовательно (fail-fast при проблеме)
    # превращаем в dict-списки (resolve=True), как будет в задачах
    tasks = [OmegaConf.to_container(cfg, resolve=True) for cfg in merged_all]
    genome_paths = []
    for t in tasks:
        g = t.get("genome")
        if g:
            genome_paths.append(str(Path(str(g)).expanduser().absolute()))
    for fa in sorted(set(genome_paths)):
        _ensure_fai(fa)

    total = len(merged_all)
    if args.workers is not None and args.workers > 0:
        workers = min(args.workers, total)
    else:
        workers = max(1, min(total, cpu_count // 2))
    log.info(f"Starting {workers} processes for {total} datasets")

    names = [_short_ds_name_from_cfg_dict(t) for t in tasks]
    for i, nm in enumerate(names, 1):
        log.debug(f"[{i:02d}] {nm}")

    # ---- FAIL-FAST ПАРАЛЛЕЛИЗАЦИЯ
    ctx = mp.get_context("spawn")
    ok_cnt = 0
    fail_cnt = 0
    errors: List[Tuple[str, str]] = []

    t0 = time.time()
    try:
        with ctx.Pool(processes=workers) as pool:
            it = pool.imap_unordered(_init_one_dataset, tasks)
            for status in it:
                if status["ok"]:
                    ok_cnt += 1
                    log.info(f"[OK] {status['name']} in {status['secs']:.2f}s  ({ok_cnt + fail_cnt}/{total})")
                else:
                    # немедленно останавливаем пул и выходим с ошибкой
                    fail_cnt += 1
                    errors.append((status["name"], status["error"] or "unknown"))
                    log.error(f"[FAIL] {status['name']} in {status['secs']:.2f}s  ({ok_cnt + fail_cnt}/{total})")
                    pool.terminate()
                    pool.join()
                    # формируем краткий отчёт и выходим с ненулевым кодом
                    err_name, err_text = errors[-1]
                    raise SystemExit(f"Fail-fast: dataset '{err_name}' failed:\n{err_text}")
            pool.close()
            pool.join()
    except KeyboardInterrupt:
        log.error("Interrupted by user. Terminating pool…")
        raise
    finally:
        pass

    total_secs = time.time() - t0
    log.info(f"Done: success={ok_cnt}, fail={fail_cnt}, total={total}, time={total_secs:.2f}s")

    if fail_cnt:
        log.error("Errors summary:")
        for nm, err in errors:
            log.error(f"--- {nm} ---\n{err}\n")
        # на всякий случай: вернуть ненулевой код, если дошли сюда
        raise SystemExit(1)

if __name__ == "__main__":
    main()
