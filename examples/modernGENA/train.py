import logging
import os
from pathlib import Path
from typing import Dict

import hydra
import numpy
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from transformers import (
    Trainer,
)


# Fix for PyTorch 2.6+ weights_only default change.
# Keep this defensive because numpy dtype symbols differ across versions.
_safe_globals = [numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype]
if hasattr(numpy, "dtypes") and hasattr(numpy.dtypes, "UInt32DType"):
    _safe_globals.append(numpy.dtypes.UInt32DType)
torch.serialization.add_safe_globals(_safe_globals)


logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pos_probs = probs[:, 1]
    if len(set(labels.tolist())) > 1:
        metrics["pr_auc"] = float(average_precision_score(labels, pos_probs))
        metrics["roc_auc"] = float(roc_auc_score(labels, pos_probs))
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO),
        format=cfg.logging.format,
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
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
    logger.info("Validation metrics: %s", metrics)

    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))


if __name__ == "__main__":
    main()
