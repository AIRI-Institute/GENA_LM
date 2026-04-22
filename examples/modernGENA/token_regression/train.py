import logging
import os
import sys
from pathlib import Path
from typing import Dict

import hydra
import numpy
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


# Fix for PyTorch 2.6+ weights_only default change.
_safe_globals = [numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype]
if hasattr(numpy, "dtypes") and hasattr(numpy.dtypes, "UInt32DType"):
    _safe_globals.append(numpy.dtypes.UInt32DType)
torch.serialization.add_safe_globals(_safe_globals)


logger = logging.getLogger(__name__)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))


def _masked_pearson(x: numpy.ndarray, y: numpy.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = numpy.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
    if den == 0:
        return 0.0
    return float(num / den)


def compute_metrics(eval_pred) -> Dict[str, float]:
    preds, label_ids = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(label_ids, tuple):
        labels, loss_mask = label_ids
        mask = loss_mask > 0
    else:
        labels = label_ids
        mask = labels > -100
    if mask.sum() == 0:
        return {"mse": 0.0, "pearson": 0.0}

    y_true = labels[mask]
    y_pred = preds[mask]
    mse = float(numpy.mean((y_pred - y_true) ** 2))
    pearson = _masked_pearson(y_true, y_pred)
    return {"mse": mse, "pearson": pearson}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO),
        format=cfg.logging.format,
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None and torch.cuda.device_count() > 0:
        # Avoid implicit DataParallel with ModernBERT compile path.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logger.info("CUDA device count: %s", torch.cuda.device_count())

    output_dir = Path(cfg.training.output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = instantiate(cfg.tokenizer)
    trainer = instantiate(
        cfg.trainer,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    metrics = trainer.evaluate()
    logger.info("Test metrics: %s", metrics)

    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))


if __name__ == "__main__":
    main()
