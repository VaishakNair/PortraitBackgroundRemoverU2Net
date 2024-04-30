import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.saving.register_keras_serializable(package="U2Net")
class Green(layers.Layer):

    def __init__(self, output_channels, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.dilation_rate = dilation_rate
        self.conv = layers.Conv2D(filters=output_channels, kernel_size=3, padding="same", dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation("relu")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "output_channels": self.output_channels,
                "dilation_rate": self.dilation_rate
            }
        )
        return config
