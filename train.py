from u2net import U2Net
import tensorflow as tf
from tensorflow.keras import layers

import getopt
import os
import pathlib
import sys
import numpy as np
import tensorflow as tf
from data import get_dataset
from metrics import muti_bce_loss_fusion
from model import IMG_HEIGHT, IMG_WIDTH
from model import deeplabv3_plus
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

train_output_dir = pathlib.Path("/content/drive/MyDrive/AIProjects/PortraitBackgroundRemover/TrainOutput")
checkpoint_dir = train_output_dir / "checkpoints"
csv_file_path = train_output_dir / "epoch_loss_metrics.csv"


def usage():
    print("Usage: train.py --initial-epoch=<some_int_value> | -i <some_int_value>")


def get_initial_epoch():
    try:
        arguments, values = getopt.getopt(sys.argv[1:], "i:", ["initial-epoch="])
        for argument, argument_value in arguments:
            if argument in ("-i", "--initial-epoch"):
                return int(argument_value)
        usage()
        sys.exit(2)
    except (getopt.GetoptError, ValueError) as e:
        print(e)
        usage()
        sys.exit(2)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    # create_dir(train_output_dir)
    # create_dir(checkpoint_dir)

    # Hyperparameters:
    batch_size = 12
    lr = 1e-03
    num_epochs = 20  # TODO Modify as needed. Must be greater than initial_epoch.

    # Dataset:
    train_dataset = get_dataset(directory="dataset/P3M-10k/train",
                                batch_size=batch_size)
    valid_dataset = get_dataset(directory="dataset/P3M-10k/validation/P3M-500-P",
                                batch_size=batch_size)

    # Model:
    initial_epoch = get_initial_epoch()
    print(f"Initial epoch: {initial_epoch}")

    if initial_epoch == 0:
        # Create a new model:
        model = deeplabv3_plus(
            (IMG_HEIGHT, IMG_WIDTH, 3))
        model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])
    else:
        # Load saved model:
        model = tf.keras.models.load_model(
            '/content/drive/MyDrive/AIProjects/PortraitBackgroundRemover'
            '/TrainOutput/checkpoints/.keras',  # TODO Point to the appropriate .keras file
            custom_objects={'dice_loss': dice_loss}
        )

    callbacks = [
        ModelCheckpoint(filepath=str(checkpoint_dir / "{epoch:02d}-{val_loss:.4f}.keras"),
                        # save_weights_only=True,
                        verbose=1,
                        # save_best_only=True  # TODO Enable
                        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(filename=csv_file_path, append=True),
        TensorBoard(log_dir=train_output_dir / "tensorboard",
                    # histogram_freq=1  # Enable only if histograms of weights of layers are needed.
                    ),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
