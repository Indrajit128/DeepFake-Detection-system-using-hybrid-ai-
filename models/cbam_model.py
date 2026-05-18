"""
CNN + CBAM (Convolutional Block Attention Module) for deepfake detection.

CBAM applies two sequential attention gates after each conv block:
  1. Channel Attention  -- "which feature maps matter?"
     Squeeze: both AvgPool and MaxPool over spatial dims
     Excite:  shared 2-layer MLP -> element-wise multiply on channels

  2. Spatial Attention  -- "where in the image matters?"
     Pool channels with Avg and Max -> concatenate -> 7x7 Conv -> sigmoid mask

Custom Keras Layer subclasses are used (not Lambda) for full Keras 3 compatibility.
"""

import keras
from keras import layers, ops
import tensorflow as tf
import config


# ---------------------------------------------------------------------------
# Serializable helper layers
# ---------------------------------------------------------------------------

class ChannelMeanPool(layers.Layer):
    """Reduces the channel axis with mean — fully serializable."""
    def call(self, x):
        return ops.mean(x, axis=-1, keepdims=True)


class ChannelMaxPool(layers.Layer):
    """Reduces the channel axis with max — fully serializable."""
    def call(self, x):
        return ops.max(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# CBAM as proper Keras Layer subclasses (Keras 3 compatible)
# ---------------------------------------------------------------------------

class ChannelAttention(layers.Layer):
    def __init__(self, ratio: int = config.CBAM_REDUCTION_RATIO, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense1 = layers.Dense(channels // self.ratio, activation="relu", use_bias=False)
        self.dense2 = layers.Dense(channels, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        # x: (B, H, W, C)
        avg = ops.mean(x, axis=[1, 2])              # (B, C)
        mx  = ops.max(x,  axis=[1, 2])              # (B, C)
        avg = self.dense2(self.dense1(avg))
        mx  = self.dense2(self.dense1(mx))
        scale = ops.sigmoid(avg + mx)               # (B, C)
        scale = ops.expand_dims(scale, axis=1)
        scale = ops.expand_dims(scale, axis=1)      # (B, 1, 1, C)
        return x * scale

    def get_config(self):
        return {**super().get_config(), "ratio": self.ratio}


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.mean_pool = ChannelMeanPool()
        self.max_pool = ChannelMaxPool()

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same",
                                  activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, x):
        avg = self.mean_pool(x)   # (B, H, W, 1)
        mx  = self.max_pool(x)    # (B, H, W, 1)
        scale = self.conv(ops.concatenate([avg, mx], axis=-1))  # (B, H, W, 1)
        return x * scale

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size}


class CBAMBlock(layers.Layer):
    def __init__(self, ratio: int = config.CBAM_REDUCTION_RATIO,
                 kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.channel_att = ChannelAttention(ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.channel_att.ratio,
                    "kernel_size": self.spatial_att.kernel_size})
        return cfg


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_cbam_cnn(input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
                   l2_reg: float = 1e-4) -> keras.Model:

    reg = keras.regularizers.l2(l2_reg)
    inp = layers.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = CBAMBlock(name="cbam_1")(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = CBAMBlock(name="cbam_2")(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = CBAMBlock(name="cbam_3")(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = CBAMBlock(name="cbam_4")(x)
    x = layers.MaxPooling2D(2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return keras.Model(inp, out, name="CNN_CBAM")


def compile_cbam_cnn(model: keras.Model,
                     lr: float = config.LEARNING_RATE) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
