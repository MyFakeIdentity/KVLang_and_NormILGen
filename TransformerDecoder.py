import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_encoder_layers, num_encoder_heads, num_ffnn_layers, ffnn_hidden_dim):
        super(TransformerClassifier, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_encoder_heads, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.ffnn = nn.ModuleList()
        prev_size = embedding_dim
        for i in range(num_ffnn_layers):
            self.ffnn.append(self._get_ffnn_layer(prev_size, ffnn_hidden_dim))
            prev_size = ffnn_hidden_dim
        self.ffnn.append(nn.Linear(prev_size, 1))
        self.ffnn.append(nn.Sigmoid())

    @staticmethod
    def _get_ffnn_layer(prev_size, next_size):
        return nn.Sequential(
            nn.Linear(prev_size, next_size),
            nn.ReLU(),
        )

    def forward(self, x):
        encoding = self.encoder(x)

        value = encoding[:, 0]
        for layer in self.ffnn:
            value = layer(value)

        return value.squeeze()
