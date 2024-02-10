import pathlib

import tensorflow as tf
from tensorflow.keras import layers

from u2net import U2Net
IMG_HEIGHT, IMG_WIDTH = U2Net.INPUT_IMAGE_HEIGHT, U2Net.INPUT_IMAGE_WIDTH


def process_image(image_path):
    print(f"Image path: {str(image_path)}")
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = layers.Rescaling(scale=1. / 255)(image)  # Rescale (Divide by 255)
    return image


def process_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = layers.Rescaling(scale=1. / 255)(mask)  # Rescale (Divide by 255)
    return mask


def get_images(directory):
    images = tf.data.Dataset.list_files(
        file_pattern=str(pathlib.Path(directory) / 'blurred_image' / '*'),
        shuffle=False)
    images = images.map(process_image)
    print(f"Images dataset element spec: {images.element_spec}")
    return images


def get_masks(directory):
    masks = tf.data.Dataset.list_files(
        file_pattern=str(pathlib.Path(directory) / 'mask' / '*'),
        shuffle=False)
    masks = masks.map(process_mask)
    print(f"Masks dataset element spec: {masks.element_spec}")
    return masks


def get_dataset(directory, batch_size):
    images = get_images(directory)

    masks = get_masks(directory)

    dataset = tf.data.Dataset.zip((images, masks))

    # Configure dataset for performance:
    # dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Dataset element spec: {dataset.element_spec}")

    return dataset
