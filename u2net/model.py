import tensorflow as tf
from tensorflow.keras import layers
from . import layers as u2netlayers


class U2Net(tf.keras.Model):
    # TODO Modify as needed:
    INPUT_IMAGE_HEIGHT = 512
    INPUT_IMAGE_WIDTH = 512

    # Input/ output dimensions of RSU blocks for normal and lite
    # versions of U2Net:
    o = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    m = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]

    o_lite = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    m_lite = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    def __init__(self, is_lite=False):
        super().__init__()

        o = U2Net.o
        m = U2Net.m
        if is_lite:
            o = U2Net.o_lite
            m = U2Net.m_lite

        self.En_1 = u2netlayers.RSU7(O=o[0], M=m[0])
        self.En_2 = u2netlayers.RSU6(O=o[1], M=m[1])
        self.En_3 = u2netlayers.RSU5(O=o[2], M=m[2])
        self.En_4 = u2netlayers.RSU4(O=o[3], M=m[3])
        self.En_5 = u2netlayers.RSU4F(O=o[4], M=m[4])
        self.En_6 = u2netlayers.RSU4F(O=o[5], M=m[5])

        self.De_5 = u2netlayers.RSU4F(O=o[6], M=m[6])
        self.De_4 = u2netlayers.RSU4(O=o[7], M=m[7])
        self.De_3 = u2netlayers.RSU5(O=o[8], M=m[8])
        self.De_2 = u2netlayers.RSU6(O=o[9], M=m[9])
        self.De_1 = u2netlayers.RSU7(O=o[10], M=m[10])

        self.d0_conv2d = layers.Conv2D(filters=1, kernel_size=3, padding="same")
        self.d1_conv2d = layers.Conv2D(filters=1, kernel_size=3, padding="same")
        self.d2_conv2d = layers.Conv2D(filters=1, kernel_size=3, padding="same")
        self.d3_conv2d = layers.Conv2D(filters=1, kernel_size=3, padding="same")
        self.d4_conv2d = layers.Conv2D(filters=1, kernel_size=3, padding="same")
        self.d5_conv2d = layers.Conv2D(filters=1, kernel_size=3, padding="same")
        self.d6_conv2d = layers.Conv2D(filters=1, kernel_size=3, padding="same")

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

        d1 = self.d1_conv2d(De_1_output)
        d2 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(
            self.d2_conv2d(De_2_output))
        d3 = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(
            self.d3_conv2d(De_3_output))
        d4 = layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(
            self.d4_conv2d(De_4_output))
        d5 = layers.UpSampling2D(size=(16, 16), interpolation="bilinear")(
            self.d5_conv2d(De_5_output))
        d6 = layers.UpSampling2D(size=(32, 32), interpolation="bilinear")(
            self.d6_conv2d(En_6_output))

        d0 = self.d0_conv2d(layers.Concatenate()([d1, d2, d3, d4, d5, d6]))

        return [layers.Activation("sigmoid")(d0), layers.Activation("sigmoid")(d1), layers.Activation("sigmoid")(d2),
                layers.Activation("sigmoid")(d3), layers.Activation("sigmoid")(d4), layers.Activation("sigmoid")(d5),
                layers.Activation("sigmoid")(d6)]
