import os
import csv
import numpy as np
from matplotlib import pyplot as plt


def read_csv(path):
    result = []

    with open(path, newline='') as csvfile:
        file_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in file_data:
            result.append(row)

    return result


def read_results_file(folder_path):
    data = read_csv(os.path.join(folder_path, "results.csv"))
    dictionary = {}

    for row in data[1:]:
        model_name, model_data = row[0], row[1:]
        dictionary[model_name] = np.array([float(item) for item in model_data], np.float32)

    return data[0][1:], dictionary


def read_results_folder(folder_path):
    header = None
    table = {}

    for file in os.listdir(folder_path):
        header, data = read_results_file(os.path.join(folder_path, file))
        _, a, b, c = file.split("-")
        c, *_ = c.split("_")

        table[f"{a}_{b}_{c}"] = data

    return header, table


def to_tensor(table: dict, min_size: int, num_models: int, depth: int):
    delta = 3 - min_size
    tensor = np.zeros((num_models, delta, delta, delta, depth), np.float32)
    mapping = {"S": 0, "M": 1, "L": 2}
    model_order = None

    for sub_table_params in table:
        a, b, c = sub_table_params.split("_")
        sub_table = table[sub_table_params]
        p1 = mapping[a[0]] - min_size
        p2 = mapping[b[0]] - min_size
        p3 = mapping[c[0]] - min_size

        model_order = list(sub_table.keys())

        for model_num, model_name in enumerate(sub_table):
            data = sub_table[model_name]
            tensor[model_num, p1, p2, p3] = data

    return model_order, tensor


def to_tensor_2(table: dict, min_sizes: tuple[int], max_sizes: tuple[int], num_models: int, depth: int):
    delta = np.array(max_sizes) - min_sizes
    tensor = np.zeros((num_models, delta[0], delta[1], delta[2], depth), np.float32)
    mapping = {"S": 0, "M": 1, "L": 2}
    model_order = None

    for sub_table_params in table:
        a, b, c = sub_table_params.split("_")
        sub_table = table[sub_table_params]
        p1 = mapping[a[0]] - min_sizes[0]
        p2 = mapping[b[0]] - min_sizes[1]
        p3 = mapping[c[0]] - min_sizes[2]

        model_order = list(sub_table.keys())

        for model_num, model_name in enumerate(sub_table):
            data = sub_table[model_name]
            tensor[model_num, p1, p2, p3] = data

    return model_order, tensor


def get_row_string(elements):
    return "      " + " & ".join(elements) + " \\\\ \\hline"


def print_table(table_name, row_names, column_names, array, rounding=3, bold_max=True, caption=""):
    text = []

    column_string = "|" + "|".join(["c" for _ in range(len(column_names) + 1)]) + "|"

    top_text = f'''\\begin{{table}}[ht]
    \\centering
    \\begin{{tabular}}{{{column_string}}} \\hline'''

    text.append(top_text)
    text.append(get_row_string([row_names[0]] + column_names))

    for row_num, (row, row_name) in enumerate(zip(array, row_names[1:])):
        entry_strings = []

        for column_num, entry in enumerate(row):
            entry_string = str(round(entry, rounding)).ljust(rounding + 2, "0")
            # entry_string = f"{entry:.3f}"

            if bold_max and np.all(entry >= array[:, column_num]):
                entry_string = f"\\mathbf{{{entry_string}}}"

            entry_string = f"${entry_string}$"

            entry_strings.append(entry_string)

        text.append(get_row_string([row_name] + entry_strings))

    bottom_text = f'''    \\end{{tabular}}
    \\caption{{{caption}}}
    \\label{{tab:{table_name}}}
\\end{{table}}'''

    text.append(bottom_text)

    print()
    for print_data in text:
        print(print_data)
    print()


def print_dual_table(table_name, row_names, column_names, array_1, array_2, rounding=3, bold_max=True, caption=""):
    text = []

    column_string = "|" + "|".join(["c" for _ in range(len(column_names) + 1)]) + "|"

    top_text = f'''\\begin{{table}}[ht]
    \\centering
    \\begin{{tabular}}{{{column_string}}} \\hline'''

    text.append(top_text)
    text.append(get_row_string([row_names[0]] + column_names))

    for row_num, (row_1, row_2, row_name) in enumerate(zip(array_1, array_2, row_names[1:])):
        entry_strings = []

        for column_num, (entry_1, entry_2) in enumerate(zip(row_1, row_2)):
            entry_string_1 = str(round(entry_1, rounding)).ljust(rounding + 2, "0")
            if bold_max and np.all(entry_1 >= array_1[:, column_num]):
                entry_string_1 = f"\\mathbf{{{entry_string_1}}}"

            entry_string_2 = str(round(entry_2, rounding)).ljust(rounding + 2, "0")
            if bold_max and np.all(entry_2 >= array_2[:, column_num]):
                entry_string_2 = f"\\mathbf{{{entry_string_2}}}"

            entry_string = f"${entry_string_1}$-${entry_string_2}$"

            entry_strings.append(entry_string)

        text.append(get_row_string([row_name] + entry_strings))

    bottom_text = f'''    \\end{{tabular}}
    \\caption{{{caption}}}
    \\label{{tab:{table_name}}}
\\end{{table}}'''

    text.append(bottom_text)

    print()
    for print_data in text:
        print(print_data)
    print()


def get_depth(array):
    if type(array) in [list, tuple]:
        return get_depth(array[0]) + 1
    else:
        return 0


def print_table_advanced(table_name, row_names, hyper_column_names, column_names, array, rounding=3, bold_max=True, caption=""):
    """
    :param table_name: str
    :param row_names: [[str, ...], []]
    :param column_names: [(str, [str, ...]), (str, [])]
    :param array: np_array[row_dims][column_dims]
    :param rounding: int
    :param bold_max: bool
    :return:
    """

    text = []

    column_string = "|" + "|".join(["c" for _ in range(len(column_names))]) + "|"

    top_text = f'''\\begin{{table}}[ht]
    \\centering
    \\begin{{tabular}}{{{column_string}}} \\hline \\toprule'''

    text.append(top_text)

    hyper_column_text = [[], ""]
    current_column = 0
    for hyper_column_name, size in hyper_column_names:
        hyper_column_text[0].append(f"\\multicolumn{{{size}}}{{c}}{{\\textbf{{{hyper_column_name}}}}}")
        hyper_column_text[1] += f"\\cmidrule(r){{{current_column + 1}-{current_column + size}}}"
        current_column += size
    text.append("&".join(hyper_column_text[0]) + "\\\\")
    text.append(hyper_column_text[1])
    text.append("&".join(column_names) + "\\\\ \\midrule")

    for row_num, (row, row_name) in enumerate(zip(array, row_names)):
        entry_strings = []

        for column_num, entry in enumerate(row):
            entry_string = str(round(entry, rounding)).ljust(rounding + 2, "0")

            if bold_max and np.all(entry >= array[:, column_num]):
                entry_string = f"\\mathbf{{{entry_string}}}"

            entry_string = f"${entry_string}$"

            entry_strings.append(entry_string)

        text.append(get_row_string(row_name + entry_strings))

    bottom_text = f'''    \\end{{tabular}}
    \\caption{{{caption}}}
    \\label{{tab:{table_name}}}
\\end{{table}}'''

    text.append(bottom_text)

    print()
    for print_data in text:
        print(print_data)
    print()


header_4_1, table_4_1 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\4_1 Depth-Width-Embedding Dim__2025_3_26__9_34_13")
header_8_1, table_8_1 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\8_1 Depth-Width-Embedding Dim__2025_3_31__9_12_10")
header_4_2, table_4_2 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\4_2 Depth-Width-Embedding Dim__2025_3_31__11_8_24")
header_8_4_2_1, table_8_4_2_1 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\8_4_2_1 Depth-Width-Embedding Dim__2025_4_1__8_31_7")

mamba_header_4_1, mamba_table_4_1 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\4_1 DWE MAMBA")
mamba_header_8_1, mamba_table_8_1 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\8_1 DWE MAMBA")
mamba_header_4_2, mamba_table_4_2 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\4_2 DWE MAMBA")
mamba_header_8_4_2_1, mamba_table_8_4_2_1 = read_results_folder(r"C:\Users\Henry\PycharmProjects\Machine Learning\Results\8_4_2_1 DWE MAMBA")

model_order_4_1, tensor_4_1 = to_tensor(table_4_1, 0, 4, 3)
model_order_8_1, tensor_8_1 = to_tensor(table_8_1, 0, 4, 3)
model_order_4_2, tensor_4_2 = to_tensor(table_4_2, 1, 4, 3)
model_order_8_4_2_1, tensor_8_4_2_1 = to_tensor(table_8_4_2_1, 1, 4, 2)

# mamba_model_order_4_1, mamba_tensor_4_1 = to_tensor_2(mamba_table_4_1, [0, 1, 0], [3, 2, 3], 1, 3)
# mamba_model_order_8_1, mamba_tensor_8_1 = to_tensor_2(mamba_table_8_1, [0, 1, 0], [3, 2, 3], 1, 3)
# mamba_model_order_4_2, mamba_tensor_4_2 = to_tensor_2(mamba_table_4_2, [1, 1, 1], [3, 2, 3], 1, 3)
# mamba_model_order_8_4_2_1, mamba_tensor_8_4_2_1 = to_tensor_2(mamba_table_8_4_2_1, [1, 1, 1], [3, 2, 3], 1, 2)

# tensor_4_1[2] = mamba_tensor_4_1
# tensor_8_1[2] = mamba_tensor_8_1
# tensor_4_2[2] = mamba_tensor_4_2
# tensor_8_4_2_1[2] = mamba_tensor_8_4_2_1

tables = [(["4"], "1", ["50", "100", "250"], tensor_4_1), (["8"], "1", ["50", "100", "250"], tensor_8_1), (["4"], "2", ["50", "100", "250"], tensor_4_2), (["8", "4", "2"], "1", ["100", "250"], tensor_8_4_2_1)]
for key_params, value_params, dataset_lengths, table in tables:
    int_keys = list(map(int, key_params))
    print_dual_table(
        f"results_{'_'.join(key_params)}_{value_params}",
        ["Model", "RNN", "LSTM", "MAMBA", "TransEncoder"],
        [f"$\le {dataset_lengths[0]}$"] + dataset_lengths[1:],
        np.min(table, axis=(1, 2, 3)), np.max(table, axis=(1, 2, 3)),
        caption=f"Results table for model min-max performance on unseen test sets drawn from KVLangE({int_keys}, {value_params}). Models were trained on sequences of length $\le{dataset_lengths[0]}$. For details on what parameters were used for each model see the results section.",
    )

# models x hyperparameters x datasets
average_delta_table = np.zeros((3, 3, 4), np.float32)
for model_num, model_id in enumerate([0, 1, 3]):
    for table_num, table in enumerate([tensor_4_1, tensor_8_1, tensor_4_2, tensor_8_4_2_1]):
        for parameter in range(3):
            indexing = [model_id, slice(None), slice(None), slice(None), slice(1, None)]
            indexing[parameter + 1] = table.shape[parameter + 1] - 1
            max_data = table[tuple(indexing)]
            indexing[parameter + 1] = 0
            min_data = table[tuple(indexing)]
            average_delta_table[model_num, parameter, table_num] = np.average(max_data - min_data)

mamba_delta_table = np.zeros((2, 4), np.float32)
for table_num, table in enumerate([tensor_4_1, tensor_8_1, tensor_4_2, tensor_8_4_2_1]):
    for parameter in range(2):
        indexing = [2, slice(None), slice(None), slice(None), slice(1, None)]
        indexing[parameter * 2 + 1] = table.shape[parameter * 2 + 1] - 1
        max_data = table[tuple(indexing)]
        indexing[parameter * 2 + 1] = 0
        min_data = table[tuple(indexing)]
        mamba_delta_table[parameter, table_num] = np.average(max_data - min_data)

sections = [("rnn", "RNN", average_delta_table[0]), ("lstm", "LSTM", average_delta_table[1]), ("mamba", "MAMBA", mamba_delta_table), ("trans", "Transformer Encoder", average_delta_table[2])]

for model_id, model_name, table in sections:
    print_table(
        f"{model_id}_delta_table",
        ["Hyperparamter", "Depth", "Width", "Embedding Size"],
        ["KVE([4], 1)", "KVE([8], 1)", "KVE([4], 2)", "KVE([8, 4, 2], 1)"],
        table,
        caption=f"This table shows the average difference between a {model_name} model with a hyperparameter maximised verses minimised across the 4 datasets (only the length generalisation sections of the datasets are used due to the models superior performance on these sections). The range of the hyperparameter values depends on the dataset and the ranges can be found in the results section.",
    )

tables = [(["4"], "1", ["50", "100", "250"], tensor_4_1), (["8"], "1", ["50", "100", "250"], tensor_8_1), (["4"], "2", ["50", "100", "250"], tensor_4_2), (["8", "4", "2"], "1", ["100", "250"], tensor_8_4_2_1)]
models = [
    ("rnn", "RNN", "The depth column shows the number of layers in each RNN model, the width column the hidden dimension and the embedding dim shows the dimensionality of the embedding used."),
    ("lstm", "LSTM", "The depth column shows the number of layers in each LSTM model, the width column the hidden dimension and the embedding dim shows the dimensionality of the embedding used."),
    ("mamba", "MAMBA model", "The depth column shows the number of layers in each MAMBA model, the width column the hidden dimension and the embedding dim shows the dimensionality of the embedding used."),
    ("trans", "Transformer Encoder", "The depth column shows the number of layers in each Transformer model, the width column the hidden dimension of the feed-forward networks and the embedding dim shows the dimensionality of the embedding used."),
]
param_combs = {
    "basic": [[depth, width, e_dim] for depth in ["1", "2", "4"] for width in ["32", "64", "128"] for e_dim in ["8", "16", "32"]],
    "only_big": [[depth, width, e_dim] for depth in ["2", "4"] for width in ["64", "128"] for e_dim in ["16", "32"]],
    "mamba_basic": [[depth, str(int(e_dim) * int(e_dim) // 2), e_dim] for depth in ["1", "2", "4"] for _ in range(3) for e_dim in ["8", "16", "32"]],
    "mamba_only_big": [[depth, str(int(e_dim) * int(e_dim) // 2), e_dim] for depth in ["2", "4"] for _ in range(2) for e_dim in ["16", "32"]],
}
params = [["basic", "basic", "mamba_basic", "basic"], ["basic", "basic", "mamba_basic", "basic"],
          ["only_big", "only_big", "mamba_only_big", "only_big"], ["only_big", "only_big", "mamba_only_big", "only_big"]]

for table_num, (key_params, value_params, dataset_lengths, table) in enumerate(tables):
    for model_num, (model_id, model_name, caption_middle) in enumerate(models):
        size_strings = [f"$\le{dataset_lengths[0]}$"] + dataset_lengths[1:]

        caption_start = f"This table shows the test accuracies for {model_name}s trained on KVLangE({list(map(int, key_params))}, {value_params}). The models were tested on 3 datasets of varying string size {', '.join(size_strings)} (note that the models were trained on a disjoint dataset of strings of size $\le{dataset_lengths[0]}$)."
        caption_end = "Bold values are the largest in their column."

        hyperparameters = param_combs[params[table_num][model_num]]

        print_table_advanced(
            f"{model_id}_{'_'.join(key_params + [value_params])}",
            hyperparameters,
            [("Hyperparameters", 3), ("Test Accuracies", len(dataset_lengths))],
            ["Depth", "Width", "Embedding Dim"] + size_strings,
            table[model_num, :, :, :].reshape((-1, len(dataset_lengths))),
            caption=caption_start + " " + caption_middle + " " + caption_end
        )

'''
figure, plots = plt.subplots(2, 2, squeeze=False, constrained_layout=True, figsize=(10, 6.5))
for i in range(4):
    plot = plots[i // 2][i % 2]

    plot.set_ylim([0, 1])
    plot.set_title(["RNN", "LSTM", "SSM", "TE"][i])
    plot.scatter(np.arange(27), tensor_4_1[i, :, :, :, 0].flatten(), s=[((j // 3) % 3) * 10 + 10 for j in range(27)], c=[(j % 3) / 2 for j in range(27)])
    # plot.scatter(np.arange(8), tensor_4_2[i, :, :, :, 0].flatten(), s=[((j // 2) % 2) * 10 + 10 for j in range(8)], c=[(j % 2) / 2 for j in range(8)])
plt.show()
'''
