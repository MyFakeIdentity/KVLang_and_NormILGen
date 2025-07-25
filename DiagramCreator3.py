import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from ResultsReader import read_results_folder, to_tensor
import matplotlib as mpl
from cycler import cycler


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

tables = [tensor_4_1, tensor_8_1, tensor_4_2]
for table_num, table in enumerate(tables):
    table[..., 1] = np.pow(table[..., 1], 1 / 100)
    table[..., 2] = np.pow(table[..., 2], 1 / 250)
    tables[table_num] = np.average(table, axis=(1, 2, 3))
tables = np.stack(tables, axis=1)
print(tables.shape)

swapped = [tables[i, :, 1] > tables[i, :, 2] for i in range(4)]
values = [[np.minimum(tables[i, :, 1], tables[i, :, 2]), np.maximum(tables[i, :, 1], tables[i, :, 2])] for i in range(4)]
# values = [[tables[i, :, 1], tables[i, :, 2]] for i in range(4)]
print(swapped)
print(tables)

column_group_names = ("KVE([4], 1)", "KVE([8], 1)", "KVE([4], 2)")
column_data = {
    'RNN': values[0],
    'LSTM': values[1],
    'MAMBA': values[2],
    'TransEncoder': values[3],
}

x = np.arange(len(column_group_names))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

colour_map = mpl.colormaps["tab20"]
plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)

fig, ax = plt.subplots(layout='constrained')

for attr_num, (attribute, (min_measurement, max_measurement)) in enumerate(column_data.items()):
    offset = width * multiplier

    colours = ([colour_map.colors[attr_num*2] for i in range(3)], [colour_map.colors[attr_num*2+1] for i in range(3)])
    labellings = ([attribute+"-100" for i in range(3)], [attribute+"-250" for i in range(3)])
    for i in range(3):
        if swapped[attr_num][i]:
            colours[0][i] = colour_map.colors[attr_num*2+1]
            colours[1][i] = colour_map.colors[attr_num*2]
            labellings[0][i] = attribute+"-250"
            labellings[1][i] = attribute+"-100"

    rects_1 = ax.bar(x + offset, min_measurement, width, label=attribute+"-100", color=colours[0])    # , color=colour_map.colors
    rects = ax.bar(x + offset, max_measurement - min_measurement, width, label=attribute+"-250", bottom=min_measurement, color=colours[1])
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length Normalised Accuracy')
ax.set_title(f'Model Length Generalisation Ability')
ax.set_xticks(x + width, column_group_names)
ax.legend(loc='lower left', ncols=4, labels=["RNN-100", "RNN-250", "LSTM-100", "LSTM-250", "MAMBA-100", "MAMBA-250", "TransEncoder-100", "TransEncoder-250"])
ax.set_ylim(0.994, 1.0005)

plt.show()
