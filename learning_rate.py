import argparse
import keras.backend as K
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import subprocess

from datetime import datetime
from keras.datasets import cifar10
from tensorflow import keras

logger = logging.getLogger(__name__)

# TODO
# - load dataset from file
# - add option to choose a linear or a geometric search space
# - split the code in submodules
# - write README
# - add documentation

# basepath for all the serialized data and logs
basepath = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".."
    )
)
weights_path = os.path.join(basepath, "tf_models", "cifair10_weigths.h5")
root_logdir = os.path.join(basepath, "tf_logs")
model_path = os.path.join(basepath, "tf_models", "cifair10_model.h5")

def histories_path():
    return os.path.join(
        basepath,
        "tf_models",
        "histories.{timestamp}.p".format(timestamp=datetime.now().timestamp())
    )

def load_cifair10():
    "Import CIFAIR10 dataset and split into train, validation and test sets."
    data = cifar10.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = data
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    
    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def make_callbacks():
    "Create callbacks for the model."

    def get_run_logdir():
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=20)

    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)

    return (earlystopping_cb, tensorboard_cb, model_checkpoint_cb)


def train_model(model, learning_rate, weights_path):
    "Train the model with a given learning rate and return the history."

    (earlystopping_cb, tensorboard_cb, model_checkpoint_cb) = make_callbacks()
    (X_train, y_train, X_valid, y_valid, X_test, y_test) = load_cifair10()

    # reset initial weights
    model.load_weights(weights_path)

    K.set_value(model.optimizer.learning_rate, learning_rate)

    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_valid, y_valid),
        callbacks=[tensorboard_cb],
        verbose=2
        # callbacks=[tensorboard_cb, model_checkpoint_cb, earlystopping_cb]
    )

    (test_loss, test_acc) = model.evaluate(X_test, y_test)
    return history, test_loss, test_acc


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


def log_subprocess_output(subprocess_logger, pipe, level):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logging.log(level, line.decode("utf-8").strip())


def make_parser():
    parser = argparse.ArgumentParser(
        description=
            "Train a deep neural network with the Tensorflow Keras API "
            "for the CIFAIR10 dataset."
    )

    parser.add_argument(
        "model_path",
        help="The path to the keras model to explore, as an HDF5 file."
    )

    parser.add_argument(
        "-o",
        "--output",
        default=histories_path(),
        help="The path to the file to save the histories to."
    )

    parser.add_argument(
        "--weights_path",
        default=weights_path,
        help="The path to the file to save the weights to."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display verbose information on stdout."
    )

    parser.add_argument(
        "-l",
        "--logfile",
        help="The path to the logfile to write to."
    )

    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser(
        "train",
        help="Train the DDN with a given learning rate "
            "and serialize the history to a binary file."
    )

    parser_train.add_argument(
        "rate",
        type=float,
        help="The learning rate to train the model with."
    )

    parser_search = subparsers.add_parser(
        "search",
        help="Search for the optimal learning rate in a specified interval."
    )

    parser_search.add_argument(
        "from",
        type=float,
        help="The left end of search interval."
    )

    parser_search.add_argument(
        "to",
        type=float,
        help="The right end of search interval."
    )

    parser_search.add_argument(
        "-s",
        "--steps",
        type=int,
        default=10,
        help="Number of steps in the search interval."
    )

    parser_search.add_argument(
        "--linear",
        action="store_true",
        help="If set the a linear space between `from` and `to` is used. "
            "Otherwise a geometric space is used."
    )

    parser_search.add_argument(
        "--hide_train_output",
        action="store_true",
        help="Set the flag to prevent output of the training subprocesses from being logged."
    )

    parser_plot = subparsers.add_parser(
        "plot",
        help="Plot the learning curves of the model."
    )

    parser_plot.add_argument(
        "--width",
        default=20,
        type=int,
        help="The width of the plot."
    )

    parser_plot.add_argument(
        "--height",
        default=9,
        type=int,
        help="The height of the plot."
    )

    parser_plot.add_argument(
        "-f",
        "--filename",
        default="plot.png",
        help="The path to the file to save the plot to."
    )

    return parser


def exec_train(args):
    model = keras.models.load_model(args.model_path)
    history, _, _ = train_model(model, args.rate, weights_path=args.weights_path)
    val_loss = min(history.history["val_loss"])
    val_acc = max(history.history["val_accuracy"])

    try:
        with open(args.output, "rb") as f:
            histories = pickle.load(f)
    except:
        logging.warning("Cannot load histories, overwriting!")
        histories = {}

    histories[args.rate] = { "val_loss": val_loss, "val_accuracy": val_acc }

    with open(args.output, "wb") as f:
        pickle.dump(histories, f)


def exec_search(args):

    if args.linear:
        logger.info("Searching a linear space between %f and %f", getattr(args, "from"), args.to)
        rates = np.linspace(getattr(args, "from"), args.to, args.steps)
    else:
        logger.info("Searching a geometric space between %f and %f", getattr(args, "from"), args.to)
        rates = np.geomspace(getattr(args, "from"), args.to, args.steps)


    for rate in rates:
        logging.info(f"Training with learning rate {rate}")

        # start a new python process for each rate
        cli_args = [
            "python", __file__,
            args.model_path, "-o", args.output,
            "train", "%s" % rate
        ]

        subp = subprocess.Popen(
            cli_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

        if not args.hide_train_output:
            subprocess_logger = logging.getLogger(
                "{0} subprocess {1}".format(__name__, subp.pid))
            if subp.stdout is not None:
                with subp.stdout:
                    log_subprocess_output(subprocess_logger, subp.stdout, logging.INFO)

            if subp.stderr is not None:
                with subp.stderr:
                    log_subprocess_output(subprocess_logger, subp.stderr, logging.ERROR)

        subp.wait()

    with open(args.output, "rb") as f:
        histories = pickle.load(f)
        logging.debug("History keys: {0}", histories.keys())


def exec_plot(args):
    try:
        with open(args.output, "rb") as f:
            histories = pickle.load(f)

        plot_histories(histories, args.width, args.height, args.filename)
    except:
        logger.error("Cannot load histories, aborting!")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s\t%(message)s',
        level=logging.INFO if args.verbose else logging.WARNING
    )

    if args.logfile:
        # create a file handler to log to the file
        file_handler = logging.FileHandler(args.logfile)
        logging.root.addHandler(file_handler)

    if hasattr(args, "rate"):
        exec_train(args)

    elif hasattr(args, "from"):
        exec_search(args)

    elif hasattr(args, "filename"):
        exec_plot(args)

    else:
        parser.print_help()
