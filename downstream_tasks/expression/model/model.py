import torch
import torch.nn as nn

from dataclasses import dataclass

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

@dataclass
class ExpressionCountsModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    mean_loss: Optional[torch.FloatTensor] = None
    deviation_loss: Optional[torch.FloatTensor] = None

class ExpressionCountsModel(BertPreTrainedModel):
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
        weight = 1.0,
        bert_cpt = None,
        text_model = None,
        hf: bool = True,
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

            missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)
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

        if text_model is not None:
            self.desc_fc = text_model
        else:
            # 2) MLP для desc_vectors
            self.desc_fc = nn.Sequential(
                nn.Linear(self.hidden_size_desc, config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )

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

    def save_cell_type_embeddings(self, save_path, cell_type_names=None):
        """
        Save the learned cell type embeddings from the nn.Embedding layer.
        
        Args:
            save_path (str): Path where to save the embeddings (e.g., 'cell_type_embeddings.pt')
            cell_type_names (list, optional): List of cell type names corresponding to embedding indices.
                                            If None, will use indices as names.
        """
        # Get the embedding weights from the separate embedding layer
        embeddings = self.cell_type_embedding.weight.data.cpu()  # Shape: (num_embeddings, embedding_dim)
        
        # Create a dictionary to save
        save_dict = {
            'embeddings': embeddings,
            'embedding_dim': embeddings.shape[1],
            'num_cell_types': embeddings.shape[0],
            'config_max_N_cell_types': self.max_N_cell_types
        }
        
        # Add cell type names if provided
        if cell_type_names is not None:
            if len(cell_type_names) != embeddings.shape[0]:
                print(f"Warning: Number of cell type names ({len(cell_type_names)}) "
                      f"doesn't match number of embeddings ({embeddings.shape[0]})")
            else:
                save_dict['cell_type_names'] = cell_type_names
        
        # Save the embeddings
        torch.save(save_dict, save_path)
        print(f"Cell type embeddings saved to {save_path}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding statistics - Mean: {embeddings.mean().item():.4f}, Std: {embeddings.std().item():.4f}")
        
        return save_dict

    def load_cell_type_embeddings(self, load_path):
        """
        Load cell type embeddings from a saved file.
        
        Args:
            load_path (str): Path to the saved embeddings file
            
        Returns:
            dict: Dictionary containing the loaded embeddings and metadata
        """
        # Load the saved embeddings
        loaded_dict = torch.load(load_path, map_location='cpu')
        
        # Check if dimensions match
        if loaded_dict['embeddings'].shape != self.cell_type_embedding.weight.shape:
            raise ValueError(f"Loaded embedding shape {loaded_dict['embeddings'].shape} "
                           f"doesn't match model embedding shape {self.cell_type_embedding.weight.shape}")
        
        # Load the embeddings into the model
        self.cell_type_embedding.weight.data = loaded_dict['embeddings'].to(self.cell_type_embedding.weight.device)
        
        print(f"Cell type embeddings loaded from {load_path}")
        print(f"Embedding shape: {loaded_dict['embeddings'].shape}")
        
        return loaded_dict

    def get_cell_type_embeddings(self, cell_type_indices=None):
        """
        Get the current cell type embeddings.
        
        Args:
            cell_type_indices (list, optional): Specific indices to retrieve. If None, returns all embeddings.
            
        Returns:
            torch.Tensor: Cell type embeddings
        """
        embeddings = self.cell_type_embedding.weight.data
        
        if cell_type_indices is not None:
            embeddings = embeddings[cell_type_indices]
            
        return embeddings

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
        meta_input_ids=None,
        meta_attention_mask=None,
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

        # Assuming that desc_vectors.shape -> (B, N, hidden_size_desc), where N - number of cell types (a.k.a experiment descriptors)
        N = desc_vectors.shape[1]

        # Расширяем выход
        # (B, seq_len, hidden_size) -> (B, N, seq_len, hidden_size)      
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1) # B, N, seq_len, hidden_size

        # Прогоняем desc_vectors через MLP 
        # (B, N, hidden_size) -> (B*N, hidden_size)
        desc_vectors_2d = desc_vectors.reshape(B*N, desc_vectors.shape[-1])
        desc_fc_output = self.desc_fc(desc_vectors_2d)  # (B*N, hidden_size)
        # (B*N, hidden_size) -> (B, N, hidden_size)
        desc_fc_output = desc_fc_output.reshape(B, N, hidden_size)

        # Складываем desc_vectors с CLS
        # CLS-токен — seq_out_expanded[:, :, 0, :]  (B, N, hidden_size)
        seq_out_expanded = seq_out_expanded.contiguous()
        seq_out_expanded[:, :, 0, :] = seq_out_expanded[:, :, 0, :].clone() + desc_fc_output

        # (B, N, seq_len, hidden_size) -> (B*N, seq_len, hidden_size)
        seq_out_flat = seq_out_expanded.reshape(B*N, seq_len, hidden_size)

        # Прогоняем через Encoder
        encoder_output = self.transformer_encoder(seq_out_flat)  # (B*N, seq_len, hidden_size)

        # Classifier -> (B*N, seq_len, 1)
        logits = self.classifier(encoder_output)
        logits = self.activation(logits)

        # Loss
        losses: dict[str, torch.Tensor] = self.loss(
            logits = logits,
            labels = labels,
            labels_mask = labels_mask,
            dataset_mean = dataset_mean,
            dataset_deviation = dataset_deviation,
        )

        if not return_dict:
            return (losses["loss"], logits)

        return ExpressionCountsModelOutput(
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            loss=losses["loss"],
            cls_loss=losses["cls_loss"],
            mean_loss=losses["mean_loss"],
            other_loss=losses["other_loss"],
            deviation_loss=losses["deviation_loss"],
        )