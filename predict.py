import os
import numpy as np
import cv2
import tensorflow as tf
from u2net import U2Net

IMG_HEIGHT, IMG_WIDTH = U2Net.INPUT_IMAGE_HEIGHT, U2Net.INPUT_IMAGE_WIDTH
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load saved model:
    model = tf.keras.models.load_model('/content/drive/MyDrive/AIProjects/PortraitBackgroundRemover'
                                       '/TrainOutput/checkpoints/.keras'  # TODO Point to appropriate .keras file
                                       )

    """ Read input image"""
    image = cv2.imread("PP.jpg", cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    x = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Prediction """
    y = model.predict(x)[0]
    # print(f"Output shape: {y.shape}")
    # print(f"{np.max(y)}, {np.min(y)}")
    y = cv2.resize(y, (w, h))
    y = np.expand_dims(y, axis=-1)

    """ Save the image """
    masked_image = image * y
    line = np.ones((h, 10, 3)) * 128
    cat_images = np.concatenate([image, line, masked_image], axis=1)
    cv2.imwrite("OPP.jpg", cat_images)
