from ImportStuff import *


class GenericSequenceModel(nn.Module):
    def __init__(self, alphabet_size: int, encoder, embedding_dim: int, sub_model_hidden_dim: int,
                 sub_model: nn.Module, num_mlp_layers: int = 2, mlp_hidden_size: int = 16):
        super(GenericSequenceModel, self).__init__()

        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim

        self.embedding = encoder    # nn.Embedding(self.alphabet_size, self.embedding_dim)

        self.sub_model = sub_model

        self.ffnn = nn.ModuleList()
        prev_size = sub_model_hidden_dim
        for i in range(num_mlp_layers):
            self.ffnn.append(self._get_ffnn_layer(prev_size, mlp_hidden_size))
            prev_size = mlp_hidden_size
        self.ffnn.append(nn.Linear(prev_size, 1))
        self.ffnn.append(nn.Sigmoid())

    @staticmethod
    def _get_ffnn_layer(prev_size, next_size):
        return nn.Sequential(
            nn.Linear(prev_size, next_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        batch_size = x.shape[0]

        x = self.embedding(x)
        h = self.sub_model(x)

        indices = torch.maximum(lengths - 1, torch.zeros(len(lengths), dtype=torch.int, device=DEVICE))
        value = h[torch.arange(batch_size, dtype=torch.int, device=DEVICE), indices]
        value[lengths == 0] = 1

        for layer in self.ffnn:
            value = layer(value)

        return value.squeeze()
