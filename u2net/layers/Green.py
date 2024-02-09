from tensorflow.keras import layers


class Green(layers.Layer):

    def __init__(self, output_channels, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters=output_channels, kernel_size=3, padding="same", dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation("relu")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)
