import copy
import os.path
import time

from SaveResults import get_time_string
import torch.cuda
from typing import Any
from ModelCreator import get_models
from Languages.LoadedDataset import LoadedDataset
import Training
from Plots import Plots
import json
from SaveResults import save_results
from ImportStuff import *
import shutil


def print_memory_usage():
    print(f"Memory allocated: {torch.cuda.memory_allocated() // 1e6}MB, Max memory allocated: {torch.cuda.max_memory_allocated() // 1e6}MB")


def run_experiment(experiment_file: str, random_orderings=False):
    with open(os.path.join(ROOT_DIR, experiment_file)) as json_data:
        experiment_data = json.load(json_data)

    if random_orderings:
        experiment_data["Experiment Name"] += " - No Ordering"

    language = LoadedDataset.load(experiment_data["Training Dataset"], random_orderings=random_orderings)
    models = get_models(experiment_data, language)

    plots = Plots([model_data["model"].model_type for model_data in models])

    saved_models_path = []
    for i, model_data in enumerate(models):
        model = model_data["model"]
        optimiser = model_data["optimiser"]
        batch_size = model_data["batch_size"]
        num_epochs = model_data["num_epochs"]

        language.set_batch_size(batch_size)

        if "Load From" in experiment_data and i < len(experiment_data["Load From"]):
            save_file = experiment_data["Load From"][i]
        else:
            save_file = None

        _, _, _, _, most_recent_save = Training.train(model, optimiser, num_epochs, language.train_dataloader, language.val_dataloader,
                                                      lambda a, b, c: plots.update(i, a, b, c), print_incorrect=False, save_file=save_file,
                                                      early_termination_threshold=100)  # 20
        saved_models_path.append(most_recent_save)

    print_memory_usage()
    test_languages = [language] + [LoadedDataset.load(dataset_name, 0, 0, 1) for dataset_name in experiment_data["Test Datasets"]]

    results = [[model["model_name"]] for model in models]
    for test_language in test_languages:
        for i, model_data in enumerate(models):
            model = model_data["model"]
            batch_size = experiment_data["Test Batch Size"]

            test_language.set_batch_size(batch_size)
            value, _, _, _, _ = Training.evaluate(model, test_language.test_dataloader)
            results[i].append(value)

            print(f"Test {model_data['model_name']} on {test_language.name} completed: {round(value * 100, 2)}%")
            print_memory_usage()

    output_path = save_results(test_languages, results, experiment_file, experiment_data, saved_models_path)
    print("Results Saved")

    return output_path


def run_multi_experiment(bundle_name: str, experiment_file: str, experiments: dict[str, list[tuple[tuple, Any]]], random_orderings: bool = False):
    with open(os.path.join(ROOT_DIR, experiment_file)) as json_data:
        base_experiment_data = json.load(json_data)

    directory_path = os.path.join(ROOT_DIR, r"Results", f"{bundle_name}__{get_time_string()}")
    os.mkdir(directory_path)

    t_total = 0

    for experiment_num, experiment_name in enumerate(experiments):
        experiment_changes = experiments[experiment_name]
        experiment_data = copy.deepcopy(base_experiment_data)

        experiment_data["Experiment Name"] = experiment_name

        for change_path, updated_value in experiment_changes:
            section = experiment_data
            for part in change_path[:-1]:
                section = section[part]
            section[change_path[-1]] = updated_value

        with open(os.path.join(ROOT_DIR, "Experiments/temp_experiment_file.json"), "w") as json_data:
            json.dump(experiment_data, json_data, indent=2)

        print(f"\nStarting Experiment {experiment_num + 1}/{len(experiments)}\n")

        start = timer()
        output_path = run_experiment("Experiments/temp_experiment_file.json", random_orderings)

        while True:
            time.sleep(0.25)
            try:
                try:
                    shutil.move(output_path, directory_path)
                except shutil.Error:
                    break
                break
            except PermissionError:
                pass

        delta = timer() - start
        t_total += delta

        print(f"\nExperiment {experiment_num + 1}/{len(experiments)} Completed in: {round(delta)}s")

        percentage_complete = (experiment_num + 1) / len(experiments)
        eta = t_total * (1 / percentage_complete) - t_total
        print(f"ETA: {eta}s\n")
        time.sleep(1)

    return directory_path


if __name__ == "__main__":
    run_multi_experiment("Test Bundle", "Experiments/Test Experiment.json", {
        "An Experiment": [(("Models Data", 0, "Training Args", "Num Epochs"), 0), (("Models Data", 1, "Training Args", "Num Epochs"), 0)],
        "An Experiment 2": [(("Models Data", 0, "Training Args", "Num Epochs"), 1), (("Models Data", 1, "Training Args", "Num Epochs"), 1)]
    })
    # run_experiment("Experiments/Test Experiment2.json")
    # run_experiment("Experiments/Zero Centre Experiment.json")
    # run_experiment("Experiments/3SUM.json")
    # run_experiment("Experiments/KeyValueExperiment.json", False)
