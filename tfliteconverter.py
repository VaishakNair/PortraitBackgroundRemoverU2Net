import tensorflow as tf
import pathlib
import numpy as np
from u2net import U2Net


IMG_HEIGHT, IMG_WIDTH = U2Net.INPUT_IMAGE_HEIGHT, U2Net.INPUT_IMAGE_WIDTH


keras_model_file_name = "40-0.467696.keras"
keras_model_path = pathlib.Path(
    "/home/vaishak/Downloads/checkpoints-20240221T053611Z-001/checkpoints") / keras_model_file_name

saved_model_dir = pathlib.Path("/home/vaishak/Downloads/tflite/u2netlite/saved_model")

tflite_model_save_dir = pathlib.Path("/home/vaishak/Downloads/tflite/u2netlite")


def get_saved_model_from_keras_model():
    #  Load .keras model:
    keras_model = tf.keras.saving.load_model(keras_model_path)

    # U2Net model produces 7 outputs but TF Lite model should only have 1 output So
    # create a new model with only the fused output of the U2Net model:
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
    outputs = keras_model(inputs)[0]
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="U2NetOneOutput")

    # Build model so that input shapes of all layers are known:
    # keras_model(np.ones(shape=(1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32))

    # Save as SavedModel:
    saved_model_name = keras_model_file_name.split('-')[0]
    keras_model.save(saved_model_dir / saved_model_name, save_format="tf")

    return tf.keras.saving.load_model(saved_model_dir / saved_model_name), saved_model_name


if __name__ == "__main__":
    saved_model, saved_model_name = get_saved_model_from_keras_model()

    # Convert the model to .tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_dir / saved_model_name))  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the .tflite model.
    with open(tflite_model_save_dir / f"{saved_model_name}.tflite", 'wb') as f:
        f.write(tflite_model)



