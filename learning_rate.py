from datetime import datetime
from keras.datasets import cifar10
from tensorflow import keras
import argparse
import keras.backend as K
import numpy as np
import os
import pickle

# TODO
# - add a command to plot the learning curves
# - add option to choose a linear or a geometric search space
# - log subprocess output to file instead of stdout

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
        callbacks=[tensorboard_cb]
        # callbacks=[tensorboard_cb, model_checkpoint_cb, earlystopping_cb]
    )

    (test_loss, test_acc) = model.evaluate(X_test, y_test)
    return history, test_loss, test_acc


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

    return parser

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if hasattr(args, "rate"):
        model = keras.models.load_model(args.model_path)
        history, _, _ = train_model(model, args.rate, weights_path=args.weights_path)
        val_loss = min(history.history["val_loss"])
        val_acc = max(history.history["val_accuracy"])

        try:
            with open(args.output, "rb") as f:
                histories = pickle.load(f)
        except:
            print("ACHTUNG! Cannot load histories, overwriting!")
            histories = {}

        histories[args.rate] = { "val_loss": val_loss, "val_accuracy": val_acc }

        with open(args.output, "wb") as f:
            pickle.dump(histories, f)

    elif hasattr(args, "from"):
        rates = np.geomspace(getattr(args, "from"), args.to, args.steps)
        for rate in rates:
            print(f"Training with learning rate {rate}")

            # start a new python process for each rate
            os.system(f"python {__file__} {args.model_path} -o {args.output} train {rate}")

        with open(args.output, "rb") as f:
            histories = pickle.load(f)
            print(histories.keys())

    else:
        parser.print_help()
