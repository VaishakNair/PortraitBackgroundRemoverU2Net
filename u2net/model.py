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
        pass
