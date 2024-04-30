import tensorflow as tf
from tensorflow.keras import layers
from .Green import Green


@tf.keras.saving.register_keras_serializable(package="U2Net")
class Blue(layers.Layer):
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.downsample = layers.MaxPooling2D(pool_size=(2, 2))
        self.green = Green(output_channels=M)

    def call(self, inputs):
        x = self.downsample(inputs)
        return self.green(x)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "M": self.M
            }
        )
        return config
