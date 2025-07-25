import csv
import os
import datetime
import shutil

import matplotlib.pyplot as plt

from ImportStuff import *


def get_time_string():
    current_time = datetime.datetime.now()
    return f"{current_time.year}_{current_time.month}_{current_time.day}__{current_time.hour}_{current_time.minute}_{current_time.second}"


def save_results(test_languages, results, experiment_file, experiment_data, saved_model_paths):
    directory_path = os.path.join(ROOT_DIR, r"Results", f"{experiment_data['Experiment Name']}__{get_time_string()}")

    os.mkdir(directory_path)

    csv_data = [["Dataset"] + [language.name for language in test_languages]] + results
    file = open(os.path.join(directory_path, "results.csv"), 'w', newline='')
    writer = csv.writer(file)
    writer.writerows(csv_data)

    shutil.copyfile(os.path.join(ROOT_DIR, experiment_file), os.path.join(directory_path, "experiment.json"))

    for i, model_path in enumerate(saved_model_paths):
        if model_path is None:
            continue

        shutil.copyfile(os.path.join(ROOT_DIR, model_path), os.path.join(directory_path, f"saved_model_{i}.pth"))

    plt.savefig(os.path.join(directory_path, "figure.png"))
    plt.close()

    return directory_path
