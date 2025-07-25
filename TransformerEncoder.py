from ImportStuff import *


class TransformerEncoder(nn.Module):
    def __init__(self, alphabet_size, encoder, embedding_dim, max_sequence_length, num_encoder_layers, num_encoder_heads, dim_encoder_feedforward, num_ffnn_layers, ffnn_hidden_dim):
        super(TransformerEncoder, self).__init__()

        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim

        encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, num_encoder_heads, dim_feedforward=dim_encoder_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.embedding = encoder    # nn.Embedding(self.alphabet_size + 1, self.embedding_dim)
        self.class_token_embedding = nn.Embedding(1, self.embedding_dim)
        self.positional_embedding = self._get_positional_embeddings(max_sequence_length, self.embedding_dim)
        self.positional_embedding.requires_grad = False

        self.ffnn = nn.ModuleList()
        prev_size = embedding_dim
        for i in range(num_ffnn_layers):
            self.ffnn.append(self._get_ffnn_layer(prev_size, ffnn_hidden_dim))
            prev_size = ffnn_hidden_dim
        self.ffnn.append(nn.Linear(prev_size, 1))
        self.ffnn.append(nn.Sigmoid())

        self.model_type = "TransformerEncoder"

    @staticmethod
    def _get_ffnn_layer(prev_size, next_size):
        return nn.Sequential(
            nn.Linear(prev_size, next_size),
            nn.ReLU(),
        )

    @staticmethod
    def _get_positional_embeddings(sequence_length, d):
        result = torch.ones(sequence_length, d, dtype=torch.float, device=DEVICE)

        i = torch.arange(sequence_length, dtype=torch.float, device=DEVICE)[:, None]
        j = torch.arange(d, dtype=torch.float, device=DEVICE)[None]

        result[:, ::2] = torch.sin(i / torch.pow(10000, j[:, ::2] / d))
        result[:, 1::2] = torch.cos(i / torch.pow(10000, (j[:, 1::2] - 1) / d))

        return result

    def forward(self, x, lengths):
        batch_size = x.shape[0]

        # Add the classification token.
        embedding = self.embedding(x)
        class_token = self.class_token_embedding(torch.zeros(1, dtype=torch.int, device=DEVICE)).reshape(1, 1, -1).expand(batch_size, -1, -1)
        embedding = torch.cat((class_token, embedding), dim=1)
        embedding[:, 1:] += self.positional_embedding[None, :len(x[0])]

        # This is correct due to mask indices starting at -1.
        mask = torch.arange(-1, x.shape[1], dtype=torch.int, device=DEVICE, requires_grad=False)[None, :] >= lengths[:, None]
        encoding = self.encoder(embedding, src_key_padding_mask=mask)

        value = encoding[:, 0]
        for layer in self.ffnn:
            value = layer(value)

        return value.squeeze()
