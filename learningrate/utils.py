from datetime import datetime
import os

# basepath for all the serialized data and logs
basepath = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data"
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
