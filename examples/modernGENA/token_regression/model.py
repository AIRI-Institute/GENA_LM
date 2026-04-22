from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.utils import ModelOutput


@dataclass
class TokenRegressionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class ModernGenaTokenRegressionModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        initialize_from_pretrained: bool = True,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if initialize_from_pretrained:
            self.backbone = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.backbone = AutoModel.from_config(
                config,
                trust_remote_code=trust_remote_code,
            )
        self.regression_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> TokenRegressionOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = self.regression_head(outputs.last_hidden_state).squeeze(-1)

        loss = None
        if labels is not None:
            if loss_mask is None:
                valid_mask = torch.ones_like(labels, dtype=torch.float32)
            else:
                valid_mask = loss_mask.float()
            valid_count = valid_mask.sum()
            if valid_count > 0:
                squared = (logits - labels) ** 2
                loss = (squared * valid_mask).sum() / valid_count
            else:
                loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return TokenRegressionOutput(loss=loss, logits=logits)
