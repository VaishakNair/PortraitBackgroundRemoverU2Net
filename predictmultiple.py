import tensorflow as tf
import cv2
import numpy as np
import pathlib
from train import get_initial_epoch
from u2net import U2Net

IMG_HEIGHT, IMG_WIDTH = U2Net.INPUT_IMAGE_HEIGHT, U2Net.INPUT_IMAGE_WIDTH

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Read input image"""
    image = cv2.imread("PP.jpg", cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    x = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    checkpoints_dir = pathlib.Path("/content/drive/MyDrive/AIProjects/PortraitBackgroundRemoverU2Net"
                                   "/TrainOutput/checkpoints")  # TODO Point to appropriate location
    results_dir = pathlib.Path("/content/drive/MyDrive/AIProjects/PortraitBackgroundRemoverU2Net"
                               "/Results/512x512")  # TODO Point to appropriate location

    for model_path in checkpoints_dir.glob('*'):
        epoch_count = int(model_path.with_suffix('').name[:2])
        if epoch_count < get_initial_epoch():  # Skip all saved models until this epoch. Useful if resuming predictions.
            continue

        # Load saved model:
        model = tf.keras.models.load_model(model_path)

        """ Prediction """
        y = model.predict(x)[0][0]
        # print(f"Output shape: {y.shape}")
        # print(f"{np.max(y)}, {np.min(y)}")
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        """ Save the image """
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)
        cv2.imwrite(str(results_dir / f"{model_path.with_suffix('').name}.jpg"), cat_images)
