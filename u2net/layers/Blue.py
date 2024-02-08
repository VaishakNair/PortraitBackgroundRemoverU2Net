from keras import layers
from Green import Green


class Blue(layers.Layer):
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)
        self.downsample = layers.MaxPooling2D(pool_size=(2, 2))  # TODO Originally uses 'MaxPool2D' class
        self.green = Green(output_channels=M)

    def call(self, inputs):
        x = self.downsample(inputs)
        return self.green(x)
