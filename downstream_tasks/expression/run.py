import rootutils
rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True, cwd=False)

import hydra
from omegaconf import DictConfig

import torch
import numpy
torch.serialization.add_safe_globals([
    numpy._core.multiarray.scalar,
    numpy._core.multiarray._reconstruct,
    numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType
])

@hydra.main(version_base = None, config_path = "./configs/")
def main(config: DictConfig) -> None:


    trainer = hydra.utils.instantiate(
        config.get("trainer")
    )

    trainer.train()

if __name__ == "__main__":
    main()
