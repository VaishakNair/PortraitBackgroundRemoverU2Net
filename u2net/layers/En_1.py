import tensorflow as tf
import keras
from keras import layers
from Green import Green
from Blue import Blue
from Pink import Pink


class En_1(layers.Layer):
    M = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        green_1 = Green(output_channels=64)
        green_2 = Green(output_channels=En_1.M)

        blue_1 = Blue(M=En_1.M)
        blue_2 = Blue(M=En_1.M)
        blue_3 = Blue(M=En_1.M)
        blue_4 = Blue(M=En_1.M)
        blue_5 = Blue(M=En_1.M)
