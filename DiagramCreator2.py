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

table = tensor_8_1
table_name = "KVLangE([8], 1)"
min_vals = np.min(table, axis=(1, 2, 3))
max_vals = np.max(table, axis=(1, 2, 3))

column_group_names = ("Length ≤50", "Length 100", "Length 250")
column_data = {
    'RNN': (min_vals[0], max_vals[0]),
    'LSTM': (min_vals[1], max_vals[1]),
    'MAMBA': (min_vals[2], max_vals[2]),
    'TransEncoder': (min_vals[3], max_vals[3]),
}

x = np.arange(len(column_group_names))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

colour_map = mpl.colormaps["tab20"]
plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)

fig, ax = plt.subplots(layout='constrained')

for attribute, (min_measurement, max_measurement) in column_data.items():
    offset = width * multiplier
    rects_1 = ax.bar(x + offset, min_measurement, width, label=attribute+"-min")    # , color=colour_map.colors
    rects = ax.bar(x + offset, max_measurement - min_measurement, width, label=attribute+"-max", bottom=min_measurement)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# tab20 colour map
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title(f'{table_name} performance by model')
ax.set_xticks(x + width, column_group_names)
ax.legend(loc='lower left', ncols=4)
# ax.set_ylim(0, 250)

plt.show()
