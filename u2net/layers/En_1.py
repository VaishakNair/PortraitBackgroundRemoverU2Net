import tensorflow as tf
import keras
from keras import layers
from Green import Green
from Blue import Blue
from Pink import Pink


class En_1(layers.Layer):
    O = 64
    M = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.green_1 = Green(output_channels=En_1.O)
        self.green_2 = Green(output_channels=En_1.M)

        self.blue_1 = Blue(M=En_1.M)
        self.blue_2 = Blue(M=En_1.M)
        self.blue_3 = Blue(M=En_1.M)
        self.blue_4 = Blue(M=En_1.M)
        self.blue_5 = Blue(M=En_1.M)

        self.white = Green(output_channels=En_1.M, dilation_rate=2)
        self.green_3 = Green(output_channels=En_1.M)

        self.pink_1 = Pink(M=En_1.M)
        self.pink_2 = Pink(M=En_1.M)
        self.pink_3 = Pink(M=En_1.M)
        self.pink_4 = Pink(M=En_1.M)
        self.pink_5 = Pink(M=En_1.O)

    def call(self, inputs):
        green_1_output = self.green_1(inputs)

        green_2_output = self.green_2(green_1_output)

        blue_1_output = self.blue_1(green_2_output)
        blue_2_output = self.blue_2(blue_1_output)
        blue_3_output = self.blue_3(blue_2_output)
        blue_4_output = self.blue_4(blue_3_output)
        blue_5_output = self.blue_5(blue_4_output)

        white_output = self.white(blue_5_output)  # TODO Inspect logic from here:

        green_3_output = self.green_3(layers.Concatenate()(
            [white_output, blue_5_output]))  # TODO Check whether concatenation order is correct or not:

        pink_1_output = self.pink_1(layers.Concatenate()([layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(green_3_output), blue_4_output]))
        pink_2_output = self.pink_2(layers.Concatenate()([pink_1_output, blue_3_output]))
        pink_3_output = self.pink_3(layers.Concatenate()([pink_2_output, blue_2_output]))
        pink_4_output = self.pink_4(layers.Concatenate()([pink_3_output, blue_1_output]))
        pink_5_output = self.pink_5(layers.Concatenate()([pink_4_output, green_2_output]))

        return layers.Add()([pink_5_output, green_1_output])
