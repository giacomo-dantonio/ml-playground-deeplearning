import argparse
import logging
import multiprocessing

from learningrate import plot
from learningrate import search
from learningrate import train
from learningrate import utils

logger = logging.getLogger(__name__)

# TODO
# - write README
# - add documentation

def make_parser():
    parser = argparse.ArgumentParser(
        description=
            "Train a deep neural network with the Tensorflow Keras API "
            "for the CIFAIR10 dataset."
    )

    parser.add_argument(
        "data_path",
        help="The path to the dataset."
    )

    parser.add_argument(
        "model_path",
        help="The path to the keras model to explore, as an HDF5 file."
    )

    parser.add_argument(
        "-o",
        "--output",
        default=utils.histories_path(),
        help="The path to the file to save the histories to."
    )

    parser.add_argument(
        "--weights_path",
        default=utils.weights_path,
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


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

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
        train.exec_train(args)

    elif hasattr(args, "from"):
        search.exec_search(args)

    elif hasattr(args, "filename"):
        plot.exec_plot(args)

    else:
        parser.print_help()
