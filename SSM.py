from mambapy.mamba import Mamba, MambaConfig
from GenericSequenceModel import GenericSequenceModel
from ImportStuff import *


class SSM(nn.Module):
    def __init__(self, alphabet_size, encoder, embedding_dim: int, d_model: int = 16, num_layers: int = 4, d_state: int = 16,
                 expand_factor: int = 2, d_conv: int = 4):
        super(SSM, self).__init__()

        self.config = MambaConfig(d_model=d_model, n_layers=num_layers, d_state=d_state, expand_factor=expand_factor, d_conv=d_conv)
        self.sub_model = Mamba(self.config)
        self.model = GenericSequenceModel(alphabet_size, encoder, embedding_dim, embedding_dim, self.sub_model)

        self.model_type = "Mamba"

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        return self.model(x, lengths)
