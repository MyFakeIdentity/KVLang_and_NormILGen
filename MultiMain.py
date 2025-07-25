from Main import run_multi_experiment


experiments = {}

model_sizes = {
    f"{depth_str}-{width_str}-{emb_dim_str}": [
        # RNN
        {
            "embedding_dim": emb_dim,
            "num_rnn_layers": depth,
            "rnn_hidden_dim": width,
        },

        # LSTM
        {
            "embedding_dim": emb_dim,
            "num_lstm_layers": depth,
            "lstm_hidden_size": width,
        },

        # SSM
        {
            "embedding_dim": emb_dim,
            "d_model": 8 * (emb_dim // 8),
            "num_layers": depth,
            "d_state": 4 * (emb_dim // 8),  # 8
            "expand_factor": 2,
            "d_conv": 4
        },

        # Trans
        {
            "embedding_dim": emb_dim,
            "num_encoder_layers": depth,
            "num_encoder_heads": 4,
            "dim_encoder_feedforward": width
        },
    ]

    for depth_str, depth in [("SD", 1), ("MD", 2), ("LD", 4)]
    for width_str, width in [("SW", 32), ("MW", 64), ("LW", 128)]
    for emb_dim_str, emb_dim in [("SE", 8), ("ME", 16), ("LE", 32)]
}

SKIP_COUNT = 0

language_parameters = [(4, 1)]  # , (4, 2), (8, 1)

for model_size in list(model_sizes.keys())[SKIP_COUNT:]:
    model_args = model_sizes[model_size]

    for key_count, value_length in language_parameters:
        changes = [
            (("Training Dataset",), f"Key-Value Up {key_count}_{value_length} len 50"),
            (("Test Datasets",), [
                f"Key-Value Up {key_count}_{value_length} len 100",
                f"Key-Value Up {key_count}_{value_length} len 250"
            ],)
        ]

        for model_index, model in enumerate(model_args):
            for arg in model:
                changes.append((("Models Data", model_index, "Model Args", arg), model[arg]))
            changes.append((("Models Data", model_index, "Training Args", "Num Epochs"), 100))

        experiments[f"KV({key_count},{value_length})-{model_size}"] = changes

run_multi_experiment("Depth-Width-Embedding Dim", "Experiments/KeyValueExperiment.json", experiments)
