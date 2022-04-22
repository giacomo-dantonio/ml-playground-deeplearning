import logging
import os
import subprocess
import numpy as np
import pickle

from tensorflow import keras
from learningrate import train
from learningrate import utils

# FIXME: replace subprocess with multiprocessing and use
# the spawn start method to avoid cuda issues

logger = logging.getLogger(__name__)

def log_subprocess_output(subprocess_logger, pipe, level):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        subprocess_logger.log(level, line.decode("utf-8").strip())


def save_initial_weights(model_path, weights_path):
    # create weights_path directory, if doesn't exist
    weights_dir = os.path.dirname(weights_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # load keras model
    model = keras.models.load_model(model_path)
    model.save_weights(weights_path)

def exec_search(args):
    if args.linear:
        logger.info("Searching a linear space between %f and %f", getattr(args, "from"), args.to)
        rates = np.linspace(getattr(args, "from"), args.to, args.steps)
    else:
        logger.info("Searching a geometric space between %f and %f", getattr(args, "from"), args.to)
        rates = np.geomspace(getattr(args, "from"), args.to, args.steps)

    # save_initial_weights(args.model_path, args.weights_path)

    for rate in rates:
        logging.info(f"Training with learning rate {rate}")

        # start a new python process for each rate
        cli_args = [
            "python", "-m", "learningrate",
            args.data_path, args.model_path,
            "--weights_path", args.weights_path,
            "-o", args.output,
            "train", "%s" % rate
        ]

        subp = subprocess.Popen(
            cli_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        if not args.hide_train_output:
            subprocess_logger = logging.getLogger(
                "{0} subprocess {1}".format(__name__, subp.pid))
            if subp.stdout is not None:
                with subp.stdout:
                    log_subprocess_output(subprocess_logger, subp.stdout, logging.INFO)

            if subp.stderr is not None:
                with subp.stderr:
                    log_subprocess_output(subprocess_logger, subp.stderr, logging.ERROR)

    with open(args.output, "rb") as f:
        histories = pickle.load(f)
        logging.debug("History keys: {0}", histories.keys())
