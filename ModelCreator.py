import torch

from RNN import RNN
from LSTM import LSTM
from TransformerEncoder import TransformerEncoder
from MinLSTM import MinLSTM
from SSM import SSM
from ImportStuff import *
from Languages.Language import Language


def gen_repeat_encoder(encoding, embedding_dim):
    encoding_tensor = torch.tensor(encoding, device=DEVICE).reshape(-1, 1)
    encoding_tensor = encoding_tensor.expand(-1, embedding_dim)

    def encode(x):
        return encoding_tensor[x]

    return encode


def create_encoder(language, embedding_dim):
    encoder_type = language.encoding_type

    if encoder_type == Language.LEARNABLE_ENCODINGS:
        return nn.Embedding(language.alphabet_size, embedding_dim)

    elif type(encoder_type) is list:
        if encoder_type[0] == Language.REPEAT_ENCODINGS:
            return gen_repeat_encoder(encoder_type[1], embedding_dim)
        else:
            raise ValueError()

    else:
        raise ValueError()


def create_model(model_type, alphabet_size, encoder, model_args):
    match model_type:
        case "RNN":
            return RNN(alphabet_size, encoder, **model_args)

        case "LSTM":
            return LSTM(alphabet_size, encoder, **model_args)

        case "TransformerEncoder":
            return TransformerEncoder(alphabet_size, encoder, **model_args)

        case "MinLSTM":
            return MinLSTM(alphabet_size, encoder, **model_args)

        case "Mamba":
            return SSM(alphabet_size, encoder, **model_args)

        case _:
            raise ValueError(f"Unrecognised model type: {model_type}.")


def create_optimiser(model, learning_rate, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_models(arguments, language):
    result = []

    for model_data in arguments["Models Data"]:
        embedding_dim = model_data["Model Args"]["embedding_dim"]
        encoder = create_encoder(language, embedding_dim)

        model = create_model(model_data["Model Type"], language.alphabet_size, encoder, model_data["Model Args"])
        model.to(DEVICE)

        training_args = model_data["Training Args"]

        optimiser = create_optimiser(model, training_args["Learning Rate"], training_args["Weight Decay"])

        all_model_data = {
            "model": model,
            "model_name": model.model_type if model_data["Model Name"] is None else model_data["Model Name"],
            "optimiser": optimiser,
            "num_epochs": training_args["Num Epochs"],
            "batch_size": training_args["Batch Size"],
        }

        result.append(all_model_data)

    return result
