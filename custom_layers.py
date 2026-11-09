# custom_layers.py
# نفس الكلاس اللي اتدرّب بيه الموديل في الـ notebook (package="oral" لازم يفضل زي ما هو)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="oral")
class BackbonePreprocess(layers.Layer):
    """Backbone-specific preprocessing (input = raw RGB 0..255)."""

    def __init__(self, kind="none", **kwargs):
        super().__init__(**kwargs)
        self.kind = kind

    def call(self, x):
        x = tf.cast(x, tf.float32)
        if self.kind == "resnet50":
            return keras.applications.resnet50.preprocess_input(x)
        if self.kind == "densenet121":
            return keras.applications.densenet.preprocess_input(x)
        if self.kind == "mobilenetv2":
            return keras.applications.mobilenet_v2.preprocess_input(x)
        return x  # EfficientNet rescales internally

    def get_config(self):
        cfg = super().get_config()
        cfg["kind"] = self.kind
        return cfg
