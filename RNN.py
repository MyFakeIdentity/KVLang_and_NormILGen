import torch
from ImportStuff import *


class RNN(nn.Module):
    def __init__(self, alphabet_size, encoder, embedding_dim, num_rnn_layers, rnn_hidden_dim, num_mlp_layers, mlp_hidden_size):
        super(RNN, self).__init__()

        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = rnn_hidden_dim

        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, num_rnn_layers, batch_first=True)

        self.embedding = encoder    # nn.Embedding(self.alphabet_size, self.embedding_dim)

        self.ffnn = nn.ModuleList()
        prev_size = self.hidden_dim
        for i in range(num_mlp_layers):
            self.ffnn.append(self._get_ffnn_layer(prev_size, mlp_hidden_size))
            prev_size = mlp_hidden_size
        self.ffnn.append(nn.Linear(prev_size, 1))
        self.ffnn.append(nn.Sigmoid())

        self.model_type = "RNN"

    @staticmethod
    def _get_ffnn_layer(prev_size, next_size):
        return nn.Sequential(
            nn.Linear(prev_size, next_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        batch_size = x.shape[0]

        x = self.embedding(x)

        outputs, _ = self.rnn(x)

        indices = torch.maximum(lengths - 1, torch.zeros(len(lengths), dtype=torch.int, device=DEVICE))
        value = outputs[torch.arange(batch_size, dtype=torch.int, device=DEVICE), indices]
        value[lengths == 0] = 1

        for layer in self.ffnn:
            value = layer(value)

        return value.squeeze()

