import h5py
import logging
import os
import pickle

from learningrate import utils
from tensorflow import keras
from keras import backend as K


def load_dataset(filepath):
    with h5py.File(filepath, "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        X_val = f["X_val"][:]
        y_val = f["y_val"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def make_callbacks():
    "Create callbacks for the model."

    def get_run_logdir():
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(utils.root_logdir, run_id)

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=20)

    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    # FIXME: this overwrite the input model!
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(utils.model_path, save_best_only=True)

    return (earlystopping_cb, tensorboard_cb, model_checkpoint_cb)


def train_model(model, dataset, learning_rate, weights_path):
    "Train the model with a given learning rate and return the history."

    (_, tensorboard_cb, _) = make_callbacks()
    (X_train, y_train, X_valid, y_valid, X_test, y_test) = dataset

    # reset initial weights
    if os.path.exists(weights_path):
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


def exec_train(args):
    model = keras.models.load_model(args.model_path)
    dataset = load_dataset(args.data_path)

    history, _, _ = train_model(model, dataset, args.rate, weights_path=args.weights_path)
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
