import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from ResultsReader import read_results_folder, to_tensor


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
section = sections[2]
_, model_name, section = section
section = np.round(section, 3)

column_group_names = ("KVE([4], 1)", "KVE([8], 1)", "KVE([4], 2)", "KVE([8, 4, 2], 1)")
column_data = {
    'Depth': section[0],
    'Width': section[1],
    'Embedding Size': section[2],
}

x = np.arange(len(column_group_names))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in column_data.items():
    offset = width * multiplier
    print(x, offset, measurement, width, attribute)
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Avg Accuracy Delta')
ax.set_title(f'Change in {model_name} performance by hyperparameter')
ax.set_xticks(x + width, column_group_names)
ax.legend(loc='upper left', ncols=4)
# ax.set_ylim(0, 250)

plt.show()
