import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, BertConfig
from transformers.utils import cached_file

@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    mean_loss: Optional[torch.FloatTensor] = None
    diviation_loss: Optional[torch.FloatTensor] = None

class ExpressionCounts(BertPreTrainedModel):
    """
    Размерности:
      - input_ids: (B, seq_len)
      - attention_mask: (B, seq_len)
      - token_type_ids: (B, seq_len) [опционально]
      - desc_vectors: (B, N, hidden_size)        
      - labels: (B, seq_len, N) -> приводим к (B*N, seq_len, 1)
      - labels_mask: (B, seq_len, N) -> приводим к (B*N, seq_len, 1)
    
    Шаги:
      1) Прогоняем через GENA -> (B, seq_len, hidden_size)
      2) Расширяем выход -> (B, N, seq_len, hidden_size)
      3) Прогоняем desc_vectors через MLP 
      4) Складываем desc_vectors с CLS
      5) Превращаем (B, N, seq_len, hidden_size) -> (B*N, seq_len, hidden_size)
      6) Прогоняем через Encoder
      7) classifier -> (B*N, seq_len, 1)
      8) Меняем labels и labels_mask -> (B*N, seq_len, 1), считаем loss
    """

    def __init__(
        self,
        config,
        loss_fct=nn.MSELoss(reduction="none"),
        activation = nn.Identity(),
        hidden_size_desc = 768,
        hidden_ff = 1024,
        num_encoder_layers = 3,
        nhead = 8,
        weight = 1,
        bert_cpt = '/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        cell_type_specific_loss_fn = None,
        text_model = None,
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size_desc = hidden_size_desc 

        # 1) GENA
        if hf:
            hf_config = BertConfig.from_pretrained(hf_model_name)
            self.bert = BertModel(hf_config, add_pooling_layer=False)
            weights_path = cached_file(hf_model_name, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")
            updated_state_dict = {
                k.replace("bert.", ""): v for k, v in state_dict.items() if k.startswith("bert.")
            }

            config = hf_config
                                            
        else:
            self.bert = BertModel(config, add_pooling_layer=False)

            checkpoint = torch.load(bert_cpt, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            updated_state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()}

        missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)

        if len(missing_k) != 0:
            print(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
        if len(unexpected_k) != 0:
            print(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")

        self.desc_model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()


        # 3) Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4) Classifier
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 5) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight
        self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

        self.post_init()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        desc_input_ids=None,
        desc_attention_mask=None,
        desc_vectors = None,
        dataset_mean = None,
        dataset_deviation = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Прогоняем через GENA
        bert_outputs = self.bert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # Notaton:
        # B - batch size
        # N - number of cell types (a.k.a experiment descriptors)
        # seq_len - sequence length (number of tokens in the input sequence)
        # hidden_size - hidden size

        # (B, seq_len, hidden_size)
        sequence_output = bert_outputs.last_hidden_state
        B, seq_len, hidden_size = sequence_output.shape # (B, seq_len, hidden_size)

        # Прогоняем desc_vectors через MLP 
        # (B, N, hidden_size) -> (B*N, hidden_size)
        assert desc_input_ids is not None and desc_attention_mask is not None, "desc_input_ids и desc_attention_mask обязательны!"
        N = desc_input_ids.shape[1]
        desc_seq_len = desc_input_ids.shape[2]
        desc_input_ids_2d = desc_input_ids.reshape(B*N, desc_seq_len)
        desc_attention_mask_2d = desc_attention_mask.reshape(B*N, desc_seq_len)
        desc_output_embeddings = self.desc_model(input_ids = desc_input_ids, attention_mask=desc_attention_mask)  # (B*N, hidden_size)
        desc_output = desc_output_embeddings.last_hidden_states[:, -1]
        # (B*N, hidden_size) -> (B, N, hidden_size)
        desc_output = desc_output.reshape(B, N, hidden_size)

        # Расширяем выход
        # (B, seq_len, hidden_size) -> (B, N, seq_len, hidden_size)      
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1) # B, N, seq_len, hidden_size

        # Складываем desc_vectors с CLS
        # CLS-токен — seq_out_expanded[:, :, 0, :]  (B, N, hidden_size)
        seq_out_expanded = seq_out_expanded.contiguous()
        seq_out_expanded[:, :, 0, :] = seq_out_expanded[:, :, 0, :].clone() + desc_output

        # (B, N, seq_len, hidden_size) -> (B*N, seq_len, hidden_size)
        seq_out_flat = seq_out_expanded.reshape(B*N, seq_len, hidden_size)

        # Прогоняем через Encoder
        encoder_output = self.transformer_encoder(seq_out_flat)  # (B*N, seq_len, hidden_size)

        # Classifier -> (B*N, seq_len, 1)
        logits = self.classifier(encoder_output)
        logits = self.activation(logits)

        # Loss
        loss = None
        if labels is not None:
            # labels, labels_mask: (B, seq_len, N)
            # Нужно:   (B*N, seq_len, 1)
            # 1) permute(0,2,1) -> (B, N, seq_len)
            # 2) reshape -> (B*N, seq_len)
            # 3) unsqueeze -> (B*N, seq_len, 1)
            labels_reshaped = labels.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)
            labels_mask_reshaped = labels_mask.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)

            # loss
            # Cчитаем общий лосс
            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B*N, seq_len, 1)

            if labels_mask_reshaped.sum() > 0:
                # Разделяем маску на CLS и остальные токены
                cls_mask = labels_mask_reshaped[:, 0:1, :]  # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, 1:, :]  # (B*N, seq_len-1, 1)

                # Считаем лосс для CLS  
                cls_loss = None
                mean_loss = None
                diviation_loss = None
                if cls_mask.sum() > 0:
                    # if self.cell_type_specific_loss_fn is not None:
                    #     cls_loss, mean_loss, diviation_loss = self.cell_type_specific_loss_fn(
                    #         cls_targets = labels_reshaped[:, 0:1, :].reshape(B,N),
                    #         cls_preds = logits[:, 0:1, :].reshape(B,N),
                    #         cls_mask = cls_mask.reshape(B,N),
                    #         dataset_mean = dataset_mean,
                    #         dataset_deviation = dataset_deviation
                    #     )
                    # else:
                    cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / cls_mask.sum()

                # Считаем лосс для остальных токенов
                other_loss = None
                if other_mask.sum() > 0:
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / other_mask.sum()

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
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            mean_loss=mean_loss,
            diviation_loss=diviation_loss,
            other_loss=other_loss
        )


