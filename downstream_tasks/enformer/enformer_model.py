from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput

from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from transformers import BigBirdPreTrainedModel, BigBirdModel


class BertForEnformer(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if getattr(config, 'add_head_dense', 0) == 0:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            interm_dense_size = getattr(config, 'add_head_dense')
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, interm_dense_size),
                nn.Tanh(),
                nn.Linear(interm_dense_size, config.num_labels)
            )

        self.activation = nn.Softplus()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        bins_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # select SEP tokens that represent target bins
        bins_output = sequence_output[bins_mask]

        bins_output = self.dropout(bins_output)
        logits = self.classifier(bins_output)
        pred = self.activation(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.PoissonNLLLoss(log_input=False, reduction='mean')
            labels = labels[labels_mask]
            loss = loss_fct(pred, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BigBirdForEnformer(BigBirdPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BigBirdModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if getattr(config, 'add_head_dense', 0) == 0:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            interm_dense_size = getattr(config, 'add_head_dense')
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, interm_dense_size),
                nn.Tanh(),
                nn.Linear(interm_dense_size, config.num_labels)
            )

        self.activation = nn.Softplus()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        bins_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_mask=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # select SEP tokens that represent target bins
        bins_output = sequence_output[bins_mask]

        bins_output = self.dropout(bins_output)
        logits = self.classifier(bins_output)
        pred = self.activation(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.PoissonNLLLoss(log_input=False, reduction='mean')
            labels = labels[labels_mask]
            loss = loss_fct(pred, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
