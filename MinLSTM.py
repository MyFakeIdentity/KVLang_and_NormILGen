import torch
from ImportStuff import *
from ParallelScan import parallel_scan_log, log_g


class MinLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MinLSTMLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear_f = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_i = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_h = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)

        batch_size = x.shape[0]

        h_0 = torch.ones((batch_size, 1, self.hidden_dim), dtype=x.dtype, device=DEVICE)

        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = torch.log(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1))

        return h


class MinLSTM(nn.Module):
    def __init__(self, alphabet_size, encoder, embedding_dim, num_lstm_layers, lstm_hidden_size, num_mlp_layers, mlp_hidden_size):
        super(MinLSTM, self).__init__()

        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = lstm_hidden_size

        self.lstm_layers = nn.ModuleList()
        prev_size = self.embedding_dim
        for i in range(num_lstm_layers):
            self.lstm_layers.append(MinLSTMLayer(prev_size, lstm_hidden_size))
            prev_size = lstm_hidden_size

        # self.linear_f = nn.Linear(self.embedding_dim, self.hidden_dim)
        # self.linear_i = nn.Linear(self.embedding_dim, self.hidden_dim)
        # self.linear_h = nn.Linear(self.embedding_dim, self.hidden_dim)

        self.embedding = encoder

        self.ffnn = nn.ModuleList()
        prev_size = lstm_hidden_size
        for i in range(num_mlp_layers):
            self.ffnn.append(self._get_ffnn_layer(prev_size, mlp_hidden_size))
            prev_size = mlp_hidden_size
        self.ffnn.append(nn.Linear(prev_size, 1))
        self.ffnn.append(nn.Sigmoid())

        self.model_type = "minLSTM"

    @staticmethod
    def _get_ffnn_layer(prev_size, next_size):
        return nn.Sequential(
            nn.Linear(prev_size, next_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)

        batch_size = x.shape[0]

        x = self.embedding(x)

        h = x
        for lstm_layer in self.lstm_layers:
            h = lstm_layer(h)

        '''
        h_0 = torch.ones((batch_size, 1, self.hidden_dim), dtype=x.dtype, device=DEVICE)

        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = torch.log(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        '''

        indices = torch.maximum(lengths - 1, torch.zeros(len(lengths), dtype=torch.int, device=DEVICE))
        value = h[torch.arange(batch_size, dtype=torch.int, device=DEVICE), indices]
        value[lengths == 0] = 1

        for layer in self.ffnn:
            value = layer(value)

        return value.squeeze()

