import tensorflow as tf
from tensorflow.keras import layers
from .Green import Green


@tf.keras.saving.register_keras_serializable(package="U2Net")
class RSU4F(layers.Layer):
    def __init__(self, O, M, **kwargs):
        super().__init__(**kwargs)

        self.O = O
        self.M = M

        self.green_1 = Green(output_channels=O)
        self.green_2 = Green(output_channels=M)
        self.white_1 = Green(output_channels=M, dilation_rate=2)
        self.dashed_white_1 = Green(output_channels=M, dilation_rate=4)
        self.short_dashed_white = Green(output_channels=M, dilation_rate=8)
        self.dashed_white_2 = Green(output_channels=M, dilation_rate=4)
        self.white_2 = Green(output_channels=M, dilation_rate=2)
        self.green_3 = Green(output_channels=O)

    def call(self, inputs):
        green_1_output = self.green_1(inputs)

        green_2_output = self.green_2(green_1_output)

        white_1_output = self.white_1(green_2_output)

        dashed_white_1_output = self.dashed_white_1(white_1_output)

        short_dashed_white_output = self.short_dashed_white(dashed_white_1_output)

        dashed_white_2_output = self.dashed_white_2(layers.Concatenate()([short_dashed_white_output, dashed_white_1_output]))

        white_2_output = self.white_2(layers.Concatenate()([dashed_white_2_output, white_1_output]))

        green_3_output = self.green_3(layers.Concatenate()([white_2_output, green_2_output]))

        return layers.Add()([green_1_output, green_3_output])

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "O": self.O,
                "M": self.M
            }
        )
        return config
