import os
import random

import numpy as np
import torch

from ImportStuff import *
import datetime
import csv


def write_csv(path, train_accuracies, validation_accuracies, losses):
    writer = csv.writer(open(path, 'w', newline=''))
    writer.writerows([["Epoch"] + [i for i in range(len(train_accuracies))], ["Train Acc"] + train_accuracies,
                      ["Val Acc"] + validation_accuracies, ["Loss", "N/A"] + losses])


def print_memory_usage():
    print(f"Memory allocated: {torch.cuda.memory_allocated() // 1e6}MB, Max memory allocated: {torch.cuda.max_memory_allocated() // 1e6}MB")


def as_percentage(fraction):
    return f"{round(fraction * 100, 2)}%"


def load_saved_state(save_file: str, model: nn.Module):
    with open(os.path.join(save_file, "results.csv")) as file:
        results_data = list(csv.reader(file, delimiter=','))

    final_completed_epoch = int(results_data[0][-1])

    train_accuracies = list(map(float, results_data[1][1:]))
    val_accuracies = list(map(float, results_data[2][1:]))
    losses = list(map(float, results_data[3][2:]))

    most_recent_save = os.path.join(save_file, f"model_epoch_{final_completed_epoch - 1}.pth")

    model.load_state_dict(torch.load(most_recent_save, weights_only=True))

    return most_recent_save, final_completed_epoch, train_accuracies, val_accuracies, losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model: nn.Module, optimiser, num_epochs, train_dataloader, validation_dataloader, on_epoch_finished, print_incorrect=False,
          early_termination_threshold=10, save_file=None):

    loss_fn = nn.BCELoss()

    train_accuracies = [0.5]
    validation_accuracies = [0.5]
    losses = []

    if save_file is None:
        current_time = datetime.datetime.now()
        session_id = f"{model.model_type}__{current_time.year}_{current_time.month:02}_{current_time.day:02}__{current_time.hour:02}_{current_time.minute:02}_{current_time.second:02}__{random.randint(0, 999_999):06}"
        base_path = os.path.join(ROOT_DIR, "Saves", session_id)
        os.mkdir(base_path)
        most_recent_save = None
        starting_epoch = 0

    else:
        base_path = save_file
        most_recent_save, final_completed_epoch, train_accuracies, validation_accuracies, losses = load_saved_state(save_file, model)
        on_epoch_finished(train_accuracies, validation_accuracies, losses)
        starting_epoch = final_completed_epoch + 1

    result_store = torch.zeros(len(train_dataloader), dtype=torch.int32, device=DEVICE)
    losses_store = torch.zeros(len(train_dataloader), dtype=torch.float32, device=DEVICE)

    print(f"Beginning to train: {model.model_type}, with {round(count_parameters(model) / 1e3, 1)}k parameters.")

    start_time = timer()

    for epoch in range(starting_epoch, num_epochs):
        model.train()

        num_attempts = 0

        for batch_idx, (data, lengths, target) in enumerate(train_dataloader):
            optimiser.zero_grad()
            pred = model(data, lengths)
            loss = loss_fn(pred, target)
            loss.backward()
            optimiser.step()

            predictions = pred > 0.5
            truth = target > 0.5

            result_store[batch_idx] = torch.sum(predictions == truth)
            losses_store[batch_idx] = loss
            num_attempts += len(target)

        total_correct = torch.sum(result_store).item()
        total_loss = torch.sum(losses_store).item()

        train_accuracies.append(total_correct / num_attempts)
        validation_acc, fpr, fnr, ptr, ttr = evaluate(model, validation_dataloader, print_incorrect and (epoch == num_epochs - 1), record_rates=True)
        validation_accuracies.append(validation_acc)
        losses.append(total_loss / num_attempts)

        write_csv(os.path.join(base_path, "results.csv"), train_accuracies, validation_accuracies, losses)
        most_recent_save = os.path.join(base_path, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), most_recent_save)

        print(f"Epoch {epoch + 1}: Training accuracy = {as_percentage(train_accuracies[-1])}, "
              f"Validation accuracy = {as_percentage(validation_accuracies[-1])}, Loss = {losses[-1] :.3g}")
        print(f"FPR: {fpr.tolist()}")
        print(f"FNR: {fnr.tolist()}")
        print(f"PTR: {ptr.tolist()}")
        print(f"TTR: {ttr.tolist()}")
        print_memory_usage()

        percentage_complete = (epoch + 1) / num_epochs
        eta = (timer() - start_time) * (1 / percentage_complete) - (timer() - start_time)
        print(f"Completed: {round(percentage_complete * 100, 1)}%, ETA: {round(eta, 2)}s")
        print()

        on_epoch_finished(train_accuracies, validation_accuracies, losses)

        if epoch >= early_termination_threshold:
            prev_max = max(validation_accuracies[1:-early_termination_threshold])
            recent_max = max(validation_accuracies[-early_termination_threshold:])

            if prev_max >= recent_max:
                print(f"No improvement in {early_termination_threshold} epochs; terminating early.")
                if print_incorrect:
                    evaluate(model, validation_dataloader, print_incorrect)
                print()
                break

    return train_accuracies, validation_accuracies, losses, base_path, most_recent_save


def evaluate(model, dataloader, print_incorrect=False, record_rates=False):
    model.eval()

    total_correct = 0
    num_guesses = 0

    false_positive_counts = np.zeros(1000, dtype=np.int32)
    false_negative_counts = np.zeros(1000, dtype=np.int32)
    pred_positive_counts = np.zeros(1000, dtype=np.int32)
    pred_negative_counts = np.zeros(1000, dtype=np.int32)
    truth_positive_counts = np.zeros(1000, dtype=np.int32)
    length_counts = np.zeros(1000, dtype=np.int32)

    with torch.no_grad():
        for batch_idx, (data, lengths, target) in enumerate(dataloader):
            pred = model(data, lengths)
            predictions = pred > 0.5
            truths = target > 0.5

            total_correct += torch.sum(predictions == truths)
            num_guesses += len(target)

            if record_rates:
                predictions = predictions.cpu().numpy().tolist()
                truths = truths.cpu().numpy().tolist()
                lengths = lengths.cpu().numpy().tolist()

                for prediction, truth, length in zip(predictions, truths, lengths):
                    if prediction and not truth:
                        false_positive_counts[length] += 1
                    if not prediction and truth:
                        false_negative_counts[length] += 1

                    pred_positive_counts[length] += prediction
                    pred_negative_counts[length] += not prediction
                    truth_positive_counts[length] += truth
                    length_counts[length] += 1

            if print_incorrect:
                for i in np.argwhere((predictions != truths).cpu()).squeeze():
                    print(f"Incorrect (length = {lengths[i]}): {','.join(map(str, data[i].cpu()[:lengths[i]].detach().numpy()))} = {truths[i]} != {predictions[i]}.")

    if record_rates:
        amount = np.argwhere(length_counts != 0)[-1][0] + 1
        fpr = (false_positive_counts / pred_positive_counts)[:amount]
        fnr = (false_negative_counts / pred_negative_counts)[:amount]
        ptr = (pred_positive_counts / length_counts)[:amount]
        ttr = (truth_positive_counts / length_counts)[:amount]

    else:
        fpr = 0
        fnr = 0
        ptr = 0
        ttr = 0

    return (total_correct / num_guesses).item(), fpr, fnr, ptr, ttr
