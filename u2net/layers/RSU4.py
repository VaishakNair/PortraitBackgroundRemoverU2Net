from tensorflow.keras import layers
from .Green import Green
from .Blue import Blue
from .Pink import Pink


class RSU4(layers.Layer):

    def __init__(self, O, M, **kwargs):
        super().__init__(**kwargs)

        self.green_1 = Green(output_channels=O)
        self.green_2 = Green(output_channels=M)

        self.blue_1 = Blue(M=M)
        self.blue_2 = Blue(M=M)
  
        self.white = Green(output_channels=M, dilation_rate=2)
        self.green_3 = Green(output_channels=M)

        self.pink_1 = Pink(M=M)
        self.pink_2 = Pink(M=O)

    def call(self, inputs):
        green_1_output = self.green_1(inputs)

        green_2_output = self.green_2(green_1_output)

        blue_1_output = self.blue_1(green_2_output)
        blue_2_output = self.blue_2(blue_1_output)

        white_output = self.white(blue_2_output)

        green_3_output = self.green_3(layers.Concatenate()(
            [white_output, blue_2_output]))

        pink_1_output = self.pink_1([green_3_output, blue_1_output])

        pink_2_output = self.pink_2([pink_1_output, green_2_output])

        return layers.Add()([pink_2_output, green_1_output])
