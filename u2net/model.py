import tensorflow as tf
from tensorflow.keras import layers
import layers as u2netlayers


class U2Net(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.En_1 = u2netlayers.RSU7(O=64, M=32)
        self.En_2 = u2netlayers.RSU6(O=128, M=32)
        self.En_3 = u2netlayers.RSU5(O=256, M=64)
        self.En_4 = u2netlayers.RSU4(O=512, M=128)
        self.En_5 = u2netlayers.RSU4F(O=512, M=256)
        self.En_6 = u2netlayers.RSU4F(O=512, M=256)

        self.De_5 = u2netlayers.RSU4F(O=512, M=256)
        self.De_4 = u2netlayers.RSU4(O=256, M=128)
        self.De_3 = u2netlayers.RSU5(O=128, M=64)
        self.De_2 = u2netlayers.RSU6(O=64, M=32)
        self.De_1 = u2netlayers.RSU7(O=64, M=16)

    def call(self, inputs, training=None, mask=None):
        En_1_output = self.En_1(inputs)
        En_2_output = self.En_2(layers.MaxPooling2D(pool_size=(2, 2))(En_1_output))
        En_3_output = self.En_3(layers.MaxPooling2D(pool_size=(2, 2))(En_2_output))
        En_4_output = self.En_4(layers.MaxPooling2D(pool_size=(2, 2))(En_3_output))
        En_5_output = self.En_5(layers.MaxPooling2D(pool_size=(2, 2))(En_4_output))
        En_6_output = self.En_6(layers.MaxPooling2D(pool_size=(2, 2))(En_5_output))

        De_5_output = self.De_5(layers.Concatenate()(
            [layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(En_6_output), En_5_output]))
        De_4_output = self.De_4(layers.Concatenate()(
            [layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(De_5_output), En_4_output]))
        De_3_output = self.De_3(layers.Concatenate()(
            [layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(De_4_output), En_3_output]))
        De_2_output = self.De_2(layers.Concatenate()(
            [layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(De_3_output), En_2_output]))
        De_1_output = self.De_1(layers.Concatenate()(
            [layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(De_2_output), En_1_output]))
