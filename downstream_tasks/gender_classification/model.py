import torch
import torch.nn as nn


class GenderChunkedClassifier(torch.nn.Module):
    def __init__(self, model, num_query_vectors=4, layer_norm_eps=1e-05, attention_dropout=0.0,
                 pooler_type='cls'):
        super().__init__()
        self.model = model
        self.layer_norm_eps = layer_norm_eps
        self.pooler_type = pooler_type
        hidden_size = model.config.hidden_size

        # trainable query vectors
        self.query_vectors = nn.Parameter(torch.randn(num_query_vectors, hidden_size))
        self.query_vectors.data.normal_(mean=0.0, std=model.config.initializer_range)

        # dense layer for .model outputs
        self.chunk_embedding_proj = nn.Linear(hidden_size, hidden_size)

        # make q, k, v for cross-attention from trainable vectors to chunks
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

        self.ln_q = torch.nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)
        self.ln_kv = torch.nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)
        self.ln_mlp = torch.nn.LayerNorm(hidden_size * num_query_vectors, eps=self.layer_norm_eps)

        self.attention_dropout = nn.Dropout(attention_dropout)

        # classification head
        self.dense1 = nn.Linear(hidden_size * num_query_vectors, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.classifier = nn.Linear(hidden_size // 2, 1)  # Binary classifier

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        bs, n_chunks, n_tokens = input_ids.shape

        bs, n_chunks, n_tokens = input_ids.shape
        input_ids = input_ids.view(bs * n_chunks, n_tokens)
        attention_mask = attention_mask.view(bs * n_chunks, n_tokens)

        if self.pooler_type == 'cls':
            # last_hidden_state or pooler_output, but pooler_output weights should be trained to be used
            # for now take cls token embedding
            # todo: add inner batch size parameter to encode chunks in loop
            outputs = self.model(input_ids, attention_mask=attention_mask)['last_hidden_state'][:, 0, :]
        else:
            # other options could be:
            # - mean/attention pooling for tokens embeddings
            # it should help as model was not trained to produce good chunk embeddings
            raise NotImplementedError('only pooler_type==cls is supported')

        # apply dense layer to chunk embeddings
        outputs = self.activation(self.chunk_embedding_proj(outputs))
        chunk_embeddings = outputs.view(bs, n_chunks, -1)
        # apply layer norm
        query_vectors = self.ln_q(self.query_vectors)
        chunk_embeddings = self.ln_kv(chunk_embeddings)

        # compute queries, keys, and values
        queries = self.query_layer(query_vectors).unsqueeze(0)  # Shape: (1, num_query_vectors, hidden_size)
        keys = self.key_layer(chunk_embeddings)  # Shape: (bs, n_chunks, hidden_size)
        values = self.value_layer(chunk_embeddings)  # Shape: (bs, n_chunks, hidden_size)

        # Compute attention scores
        attention_scores = torch.matmul(  # Shape: (bs, num_query_vectors, n_chunks)
            queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(keys.size(-1), dtype=torch.float32))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # apply attention dropout
        attention_probs = self.attention_dropout(attention_probs)

        # compute weighted sum of values for each query vector
        attended_chunks = torch.matmul(attention_probs, values)  # Shape: (bs, num_query_vectors, hidden_size)

        # concat cross-attention results into one feature vector
        pooled_output = attended_chunks.view(bs, -1)  # Shape: (bs, num_query_vectors * hidden_size)

        # pass through dense layers and classifier
        x = self.dropout(self.activation(self.dense1(self.ln_mlp(pooled_output))))
        x = self.dropout(self.activation(self.dense2(x)))
        logits = self.classifier(x)
        predictions = self.sigmoid(logits)

        # Compute loss if labels are provided
        if labels is not None:
            labels = labels.float().unsqueeze(1)  # Ensure labels have the correct shape
            loss = self.loss_fn(logits, labels).mean()
            return {'loss': loss, 'predictions': predictions}

        return {'predictions': predictions, 'attention_scores': attention_scores}
