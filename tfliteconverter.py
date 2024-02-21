import tensorflow as tf
import pathlib
from u2net import U2Net
IMG_HEIGHT, IMG_WIDTH = U2Net.INPUT_IMAGE_HEIGHT, U2Net.INPUT_IMAGE_WIDTH
import numpy as np

keras_model_file_name = "40-0.467696.keras"
keras_model_path = pathlib.Path(
    "/home/vaishak/Downloads/checkpoints-20240221T053611Z-001/checkpoints") / keras_model_file_name

tflite_model_save_dir = pathlib.Path("/home/vaishak/Downloads/tflite/u2netlite")


def get_saved_model_from_keras_model():
    #  Load .keras model:
    keras_model = tf.keras.saving.load_model(keras_model_path)
    print(type(keras_model))
    # Build model so that input shape of all layers are known
    keras_model(np.ones(shape=(1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32))

    # Save as SavedModel:
    saved_model_dir_name = keras_model_file_name.split('-')[0]
    keras_model.save(tflite_model_save_dir / saved_model_dir_name, save_format="tf")

    return tf.keras.saving.load_model(tflite_model_save_dir / saved_model_dir_name)


if __name__ == "__main__":
    saved_model = get_saved_model_from_keras_model()
    print(type(saved_model))

