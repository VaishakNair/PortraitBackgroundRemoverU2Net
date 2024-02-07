from keras import layers
import Green


class Pink(layers.Layer):

    def __init__(self, m, **kwargs):
        super().__init(**kwargs)
        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.green = Green(cout=m)

    def call(self, inputs):
        x = self.upsample(inputs)
        return self.green(x)
