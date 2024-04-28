import os
import numpy as np
import cv2
import tensorflow as tf
from u2net import U2Net


if __name__ == "__main__":
    # Seed random generators:
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load .keras model:
    model = tf.keras.models.load_model('path/to/.keras')

    # Input image should be of height x width: 512 x 512 with 3 color channels:
    INPUT_IMG_HEIGHT = U2Net.INPUT_IMAGE_HEIGHT
    INPUT_IMG_WIDTH = U2Net.INPUT_IMAGE_WIDTH
    INPUT_CHANNEL_COUNT = 3

    print(model.summary())

    # Read input image:
    image = cv2.imread("SampleImage.jpg", cv2.IMREAD_COLOR)   # png images are also supported
    h, w, channel_count = image.shape

    # Preprocess input image:
    if channel_count > INPUT_CHANNEL_COUNT:   # png images will have an alpha channel. Remove it:
        image = image[..., :INPUT_CHANNEL_COUNT]

    x = cv2.resize(image, (INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT)) # Resize input image to 512 x 512 x 3
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # Generate the pixel-wise probability mask:
    probability = model.predict(x)[0][0]

    # Produce output image:
    probability = cv2.resize(probability, dsize=(w, h))  # Resize the probability mask from (512, 512, 1) to (h, w)
    probability = np.expand_dims(probability, axis=-1)  # Reshape the probability mask from (h, w) to (h, w, 1)

    alpha_image = np.insert(image, 3, 255.0, axis=2)  # Add an opaque alpha channel to the input image

    PROBABILITY_THRESHOLD = 0.7  # Pixels with probability values less than or equal to the threshold belong to the background class.

    # Apply the probability mask by making pixels with probability value <= PROBABILITY_THRESHOLD transparent in the output image:
    masked_image = np.where(probability > PROBABILITY_THRESHOLD, alpha_image, 0.0)

    # Save output to a png file:
    cv2.imwrite("./output.png", masked_image)
