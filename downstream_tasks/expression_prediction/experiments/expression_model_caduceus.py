import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers.utils import cached_file

@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    
class ExpressionCounts(nn.Module):
    def __init__(
        self,
        loss_fct=nn.MSELoss(reduction="none"),
        activation = nn.Identity(),
        hidden_size_desc = 768,
        weight = 1,
        cell_type_specific_loss_fn = None,
        text_model = None,
        hf_model_name: str = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    ):
        super().__init__()
        # self.config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        self.hidden_size_desc = hidden_size_desc 
        self.hidden_size = 256

        # 1) Caduceus
        self.caduceus_model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)

        if text_model is not None:
            self.desc_fc = text_model
        else:
            # 2) MLP для desc_vectors
            self.desc_fc = nn.Sequential(
                nn.Linear(self.hidden_size_desc, self.hidden_size),
                nn.LeakyReLU(),
            # nn.Dropout(p=0.1),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        # 3) Caduceus cells
        self.caduceus_cells_model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)

        for p in self.caduceus_cells_model.backbone.embeddings.word_embeddings.parameters():
            p.requires_grad = False


        # 4) Classifier
        self.classifier = nn.Linear(self.hidden_size, 1)

        # 5) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight
        self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

        # self.post_init()


    def forward(
        self,
        input_ids=None,
        labels_mask=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        meta_input_ids=None,
        meta_attention_mask=None,
        desc_vectors = None,
        dataset_mean = None,
        dataset_deviation = None,
    ):

        return_dict = True

        # Прогоняем через GENA
        caduceus_outputs = self.caduceus_model(input_ids)
        # Notaton:
        # B - batch size
        # N - number of cell types (a.k.a experiment descriptors)
        # seq_len - sequence length (number of tokens in the input sequence)
        # hidden_size - hidden size

        # (B, seq_len, hidden_size)
        sequence_output = caduceus_outputs.last_hidden_state
        B, seq_len, hidden_size = sequence_output.shape # (B, seq_len, hidden_size)

        # Assuming that desc_vectors.shape -> (B, N, hidden_size_desc), where N - number of cell types (a.k.a experiment descriptors)
        N = desc_vectors.shape[1]

        # Расширяем выход
        # (B, seq_len, hidden_size) -> (B, N, seq_len, hidden_size)      
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1) # B, N, seq_len, hidden_size

        # Прогоняем desc_vectors через MLP 
        # (B, N, hidden_size) -> (B*N, hidden_size)
        desc_vectors_2d = desc_vectors.reshape(B*N, desc_vectors.size()[-1])
        desc_fc_output = self.desc_fc(desc_vectors_2d)  # (B*N, hidden_size)
        # (B*N, hidden_size) -> (B, N, hidden_size)
        desc_fc_output = desc_fc_output.reshape(B, N, hidden_size)

        # Добавляем desc_fc_output как первый токен перед последовательностью
        seq_out_expanded = seq_out_expanded.contiguous()
        desc_token = desc_fc_output.unsqueeze(2)  # (B, N, 1, hidden_size)
        seq_out_expanded = torch.cat([desc_token, seq_out_expanded], dim=2)  # (B, N, seq_len+1, hidden_size)

        seq_len = seq_len + 1
        # (B, N, seq_len+1, hidden_size) -> (B*N, seq_len+1, hidden_size)
        seq_out_flat = seq_out_expanded.reshape(B*N, seq_len , hidden_size)

        # Прогоняем через Encoder
        caduceus_cells_output = self.caduceus_cells_model(inputs_embeds=seq_out_flat,return_dict=True).last_hidden_state 
  # (B*N, seq_len, hidden_size)

        # Classifier -> (B*N, seq_len, 1)
        logits = self.classifier(caduceus_cells_output)
        logits = self.activation(logits)

        # Loss
        loss = None
        if labels is not None:
            # labels, labels_mask: (B, seq_len, N)
            # Нужно:   (B*N, seq_len, 1)
            # 1) permute(0,2,1) -> (B, N, seq_len)
            # 2) reshape -> (B*N, seq_len)
            # 3) unsqueeze -> (B*N, seq_len, 1)
            # расширяем labels и labels_mask, добавляя "пустой" токен для desc_token
            pad = torch.zeros((labels.size(0), 1, labels.size(2)), device=labels.device, dtype=labels.dtype)
            labels = torch.cat([pad, labels], dim=1)

            pad_mask = torch.zeros((labels_mask.size(0), 1, labels_mask.size(2)), device=labels_mask.device, dtype=labels_mask.dtype)
            labels_mask = torch.cat([pad_mask, labels_mask], dim=1)

            labels_reshaped = labels.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)
            labels_mask_reshaped = labels_mask.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)

            # loss
            # Cчитаем общий лосс
            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B*N, seq_len, 1)

            if labels_mask_reshaped.sum() > 0:
                # Разделяем маску на последний токен (CLS) и остальные токены
                cls_mask = labels_mask_reshaped[:, -1:, :]  # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, :-1, :]  # (B*N, seq_len-1, 1)

                # Считаем лосс для CLS (последний токен)
                cls_loss = None
                if cls_mask.sum() > 0:
                        cls_loss = (unreduced_loss[:, -1:, :] * cls_mask).sum() / cls_mask.sum()
                        mean_loss = None
                        diviation_loss = None

                # Считаем лосс для остальных токенов (все, кроме последнего)
                other_loss = None
                if other_mask.sum() > 0:
                    other_loss = (unreduced_loss[:, :-1, :] * other_mask).sum() / other_mask.sum()

                # Объединяем лоссы
                if cls_loss is not None and other_loss is not None:
                    loss = cls_loss + self.weight * other_loss
                elif cls_loss is not None:
                    loss = cls_loss
                elif other_loss is not None:
                    loss = self.weight * other_loss

        if not return_dict:
            return (loss, logits)

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            other_loss=other_loss
        )

