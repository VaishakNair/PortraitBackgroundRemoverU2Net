import tensorflow as tf
from tensorflow.keras import layers
import layers as u2netlayers


class U2Net(tf.keras.Model):
    # TODO Modify as needed:
    INPUT_IMAGE_HEIGHT = 1024
    INPUT_IMAGE_WIDTH = 1024

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

        d1 = layers.Conv2D(1, kernel_size=3, padding="same")(De_1_output)
        d2 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(
            layers.Conv2D(1, kernel_size=3, padding="same")(De_2_output))
        d3 = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(
            layers.Conv2D(1, kernel_size=3, padding="same")(De_3_output))
        d4 = layers.UpSampling2D(size=(8, 8), interpolation="bilinear")(
            layers.Conv2D(1, kernel_size=3, padding="same")(De_4_output))
        d5 = layers.UpSampling2D(size=(16, 16), interpolation="bilinear")(
            layers.Conv2D(1, kernel_size=3, padding="same")(De_5_output))
        d6 = layers.UpSampling2D(size=(32, 32), interpolation="bilinear")(
            layers.Conv2D(1, kernel_size=3, padding="same")(En_6_output))

        d0 = layers.Conv2D(filters=1, kernel_size=3, padding="same")(layers.Concatenate()([d1, d2, d3, d4, d5, d6]))

        return [layers.Activation("sigmoid")(d0), layers.Activation("sigmoid")(d1), layers.Activation("sigmoid")(d2),
                layers.Activation("sigmoid")(d3), layers.Activation("sigmoid")(d4), layers.Activation("sigmoid")(d5),
                layers.Activation("sigmoid")(d6)]
