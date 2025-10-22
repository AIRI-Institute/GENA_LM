import numpy as np
from dataclasses import dataclass
from transformers.modeling_outputs import TokenClassifierOutput

from transformers import TrainerCallback

@dataclass
class ExpressionCountsModelOutput(TokenClassifierOutput):
    labels_reshaped: np.ndarray | None = None
    labels_mask_reshaped: np.ndarray | None = None
    cls_loss: np.ndarray | None = None
    other_loss: np.ndarray | None = None
    mean_loss: np.ndarray | None = None
    deviation_loss: np.ndarray | None = None

def join_or_pass(values):
    result = values
    if isinstance(values, (list, tuple)):
        result = np.concatenate(values)
    return result

#class ComputeLoss:
#    def __call__(self, model, inputs, return_outputs=False, num_items_in_batch=None):

class Metrics:
    def __call__(self, prediction_obj):
        preds = prediction_obj.predictions
        preds = [join_or_pass(p) for p in preds]
        output = ExpressionCountsModelOutput(*preds)


class LogTrainMetricsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if hasattr(state, "trainer_instance"):
                trainer = state.trainer_instance
                if trainer and trainer.is_in_train:
                    for callback in trainer.custom_callbacks:
                        result = callback.compute()
                        callback.reset()
                        trainer.log(result)
         # remove trainer instance from state to avoid looping
        state.trainer_instance = None
        return control
        