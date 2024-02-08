from keras import layers
from Green import Green


class Pink(layers.Layer):

    def __init__(self, M, **kwargs):
        super().__init(**kwargs)
        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.green = Green(output_channels=M)

    def call(self, inputs):
        x = self.upsample(inputs)
        return self.green(x)