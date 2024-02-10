import keras
from tensorflow.keras.losses import BinaryCrossentropy

@keras.saving.register_keras_serializable()
def muti_bce_loss_fusion(y_true, y_pred):
    for i in range(y_true.shape[0]):



