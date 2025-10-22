import torch
import hydra
from transformers import Trainer


def to_cpu(data: torch.Tensor) -> torch.Tensor:
    return data.detach().cpu()

def handle_input_output(data, map_func = to_cpu):
    if torch.is_tensor(data):
        return map_func(data)
    elif isinstance(data, dict):
        return {k: handle_input_output(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [handle_input_output(v) for v in iter(data)]
    elif hasattr(data, "asdict"):
        asdict = handle_input_output(data.asdict())
        return type(data)(**asdict)
    else:
        return data

class TrainerWrapper(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        custom_callbacks = kwargs.pop("custom_callbacks", [])
        if (callbacks_config := kwargs.pop("callbacks", None)):
            callbacks = hydra.utils.instantiate(callbacks_config)
            kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)
        self.custom_callbacks = custom_callbacks
        self.state.trainer_instance = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model=model, inputs=inputs, return_outputs=True, **kwargs)
        if not loss.requires_grad:
            if 0 < len(self.custom_callbacks):
                cpu_inputs = handle_input_output(inputs)
                cpu_outputs = handle_input_output(outputs)
                for callback in self.custom_callbacks:
                    callback.update(inputs, outputs)
            self.state.trainer_instance = self
        return (loss, outputs) if return_outputs else loss
