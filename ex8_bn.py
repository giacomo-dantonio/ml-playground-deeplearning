import argparse
import keras.backend as K
import numpy as np
import os
import pickle
from parso import parse

from tensorflow import keras
from keras.datasets import cifar10

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
histories_path = os.path.join(basepath, "tf_models", "histories.p")

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


# input_shape = X_train.shape[1:]
def make_model(input_shape):
    "Create a model with batch normalization."
    layers = [keras.layers.Flatten(input_shape=input_shape)]
    for _ in range(20):
        layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"))
    layers.append(keras.layers.BatchNormalization())
    layers.append(keras.layers.Dense(10, activation="softmax"))

    model = keras.models.Sequential(layers)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Nadam(),
        metrics=["accuracy"]
    )

    model.save_weights(weights_path)

    return model


def train_model(model, learning_rate):
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

    parser_train.add_argument(
        "-o",
        "--output",
        default=histories_path
    )

    parser_explore = subparsers.add_parser(
        "search",
        help="Search for the optimal learning rate in a specified interval."
    )

    parser_explore.add_argument(
        "from",
        type=float,
        help="The left end of search interval."
    )

    parser_explore.add_argument(
        "to",
        type=float,
        help="The right end of search interval."
    )

    parser_explore.add_argument(
        "-o",
        "--output",
        default=histories_path
    )

    return parser

# rates = np.geomspace(1e-6, 1, 10)
# histories = []
# for learning_rate in rates:
#     # FIXME: spawn another process for this, because this shit keeps dying
#     # all the time

#     histories.append((learning_rate, history.history))

#     K.clear_session()

# with open("histories.p", "wb") as f:
#     pickle.dump(histories, f)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if hasattr(args, "rate"):
        model = make_model()
        history, test_loss, test_acc = train_model(model, args.rate)

        with open(args.output, "wb") as f:
            histories = pickle.load(f)
    else:
        print("SEARCH")
