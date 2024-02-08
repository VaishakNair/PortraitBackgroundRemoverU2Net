from tensorflow.keras import layers
from Green import Green
from Blue import Blue
from Pink import Pink


class En_3(layers.Layer):
    O = 256
    M = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.green_1 = Green(output_channels=En_2.O)
        self.green_2 = Green(output_channels=En_2.M)

        self.blue_1 = Blue(M=En_2.M)
        self.blue_2 = Blue(M=En_2.M)
        self.blue_3 = Blue(M=En_2.M)

        self.white = Green(output_channels=En_2.M, dilation_rate=2)
        self.green_3 = Green(output_channels=En_2.M)

        self.pink_1 = Pink(M=En_2.M)
        self.pink_2 = Pink(M=En_2.M)
        self.pink_3 = Pink(M=En_2.O)

    def call(self, inputs):
        green_1_output = self.green_1(inputs)

        green_2_output = self.green_2(green_1_output)

        blue_1_output = self.blue_1(green_2_output)
        blue_2_output = self.blue_2(blue_1_output)
        blue_3_output = self.blue_3(blue_2_output)

        white_output = self.white(blue_3_output)

        green_3_output = self.green_3(layers.Concatenate()(
            [white_output, blue_3_output]))

        pink_1_output = self.pink_1([green_3_output, blue_2_output])
        pink_2_output = self.pink_2([pink_1_output, blue_1_output])

        pink_3_output = self.pink_3([pink_2_output, green_2_output])

        return layers.Add()([pink_3_output, green_1_output])
