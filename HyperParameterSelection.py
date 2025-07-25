import random
import numpy as np
from ImportStuff import *
from Plots import Plots


def _pick_hyperparameters(potential_hyperparameters):
    hyperparameters = {}

    for hyperparameter in potential_hyperparameters:
        hyperparameters[hyperparameter] = random.choice(potential_hyperparameters[hyperparameter])

    return hyperparameters


def select_optimal_hyperparameters(model_creator, training_func, optimiser_generator, potential_hyperparameters, num_attempts):
    best_performance = -np.inf
    best_hyperparameters = None
    plots = Plots([f"Model(0)"])

    for attempt in range(num_attempts):
        hyperparameters = _pick_hyperparameters(potential_hyperparameters)

        plots.plot_titles[0] = f"Model({attempt})"

        model = model_creator(hyperparameters)
        model.to(DEVICE)

        optimiser = optimiser_generator(model)

        print(f"Training model ({attempt+1}/{num_attempts}) with hyperparameters: {hyperparameters}")

        start = timer()
        performance = training_func(attempt, model, optimiser, plots)

        print(f"Validation performance: {performance}, trained in: {round(timer() - start, 2)}s")

        if performance > best_performance:
            best_performance = performance
            best_hyperparameters = hyperparameters

    print(f"Best hyperparameters: {best_hyperparameters}, with performance: {best_performance}")

    return best_hyperparameters
