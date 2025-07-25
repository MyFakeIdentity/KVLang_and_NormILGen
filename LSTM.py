import torch
from ImportStuff import *


class LSTM(nn.Module):
    def __init__(self, alphabet_size, encoder, embedding_dim, num_lstm_layers, lstm_hidden_size, num_mlp_layers, mlp_hidden_size):
        super(LSTM, self).__init__()

        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_size, num_lstm_layers, batch_first=True)
        self.embedding = encoder

        self.ffnn = nn.ModuleList()
        prev_size = lstm_hidden_size
        for i in range(num_mlp_layers):
            self.ffnn.append(self._get_ffnn_layer(prev_size, mlp_hidden_size))
            prev_size = mlp_hidden_size
        self.ffnn.append(nn.Linear(prev_size, 1))
        self.ffnn.append(nn.Sigmoid())

        self.model_type = "LSTM"

    @staticmethod
    def _get_ffnn_layer(prev_size, next_size):
        return nn.Sequential(
            nn.Linear(prev_size, next_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        batch_size = len(x)
        embedding = self.embedding(x)

        outputs, _ = self.lstm(embedding)

        # Need a special case when sequence lengths is 0.
        indices = torch.maximum(lengths - 1, torch.zeros(len(lengths), dtype=torch.int, device=DEVICE))
        value = outputs[torch.arange(batch_size, dtype=torch.int, device=DEVICE), indices]
        value[lengths == 0] = 0
        for layer in self.ffnn:
            value = layer(value)

        return value.squeeze()
