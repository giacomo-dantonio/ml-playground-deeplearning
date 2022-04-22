import logging
import matplotlib.pyplot as plt
import pickle

logger = logging.getLogger(__name__)

def plot_histories(histories, height, width, filename):
    rates = [rate for rate in histories.keys() if rate != 1e-5]
    losses = [histories[rate]["val_loss"] for rate in rates]
    accuracies = [histories[rate]["val_accuracy"] for rate in rates]

        # plot the loss and the accuracy against the rates
    plt.figure(figsize=(width, height))
    plt.plot(rates, losses, label="loss")
    plt.plot(rates, accuracies, label="accuracy")
    plt.legend()

    plt.savefig(filename)


def exec_plot(args):
    try:
        with open(args.output, "rb") as f:
            histories = pickle.load(f)

        plot_histories(histories, args.width, args.height, args.filename)
    except:
        logger.error("Cannot load histories, aborting!")
