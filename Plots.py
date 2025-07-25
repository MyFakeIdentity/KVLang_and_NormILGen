import matplotlib.pyplot as plt
import math
from ImportStuff import *


class Plots:
    def __init__(self, plot_titles, num_plots_per_row=7):
        num_plots = len(plot_titles)

        self.num_plots_per_row = num_plots_per_row

        self.figure, plots = plt.subplots(2 * math.ceil(num_plots / num_plots_per_row), min(num_plots, num_plots_per_row), squeeze=False, constrained_layout=True, figsize=(10, 6.5))

        self.plots = []
        self.plot_titles = plot_titles
        self.plot_data = [([], [], []) for _ in range(num_plots)]

        for plot_num in range(num_plots):
            acc_plot = plots[(plot_num // num_plots_per_row) * 2][plot_num % num_plots_per_row]
            loss_plot = plots[(plot_num // num_plots_per_row) * 2 + 1][plot_num % num_plots_per_row]
            self.plots.append((acc_plot, loss_plot))

    def _redraw_plots(self, plot_list=None):
        plot_list = range(len(self.plots)) if plot_list is None else plot_list

        for plot_num in plot_list:
            acc_plot, loss_plot = self.plots[plot_num]
            title = self.plot_titles[plot_num]
            train_acc, val_acc, loss = self.plot_data[plot_num]

            # for (acc_plot, loss_plot), title, (train_acc, val_acc, loss) in zip(self.plots, self.plot_titles, self.plot_data):
            acc_plot.clear()
            loss_plot.clear()

            acc_plot.set_xlabel("Epoch")
            acc_plot.set_ylabel("Accuracy")
            acc_plot.set_title(title)
            acc_plot.plot(np.arange(0, len(train_acc)), np.array(train_acc) * 100, label="Training")
            acc_plot.plot(np.arange(0, len(val_acc)), np.array(val_acc) * 100, label="Validation")
            acc_plot.legend(loc="upper left")
            acc_plot.set_ylim([0, 100])

            loss_plot.set_xlabel("Epoch")
            loss_plot.set_ylabel("Loss")
            loss_plot.set_yscale('log')
            loss_plot.set_title(title)
            loss_plot.plot(np.arange(1, len(loss) + 1), loss)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update(self, plot_num, train_accuracies, validation_accuracies, losses):
        self.plot_data[plot_num] = (train_accuracies, validation_accuracies, losses)
        self._redraw_plots([plot_num])
