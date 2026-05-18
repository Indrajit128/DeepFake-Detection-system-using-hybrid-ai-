"""
CNN + CBAM (Convolutional Block Attention Module) for deepfake detection.

CBAM applies two sequential attention gates after each conv block:
  1. Channel Attention  — "which feature maps matter?"
     Squeeze: both AvgPool and MaxPool over spatial dims
     Excite:  shared 2-layer MLP → element-wise multiply on channels

  2. Spatial Attention  — "where in the image matters?"
     Pool channels with Avg and Max → concatenate → 7x7 Conv → sigmoid mask

This forces the network to focus on forgery artefacts (blending boundaries,
frequency inconsistencies) rather than unrelated scene content.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import config


# ---------------------------------------------------------------------------
# CBAM building blocks
# ---------------------------------------------------------------------------

def channel_attention(x: tf.Tensor, ratio: int = config.CBAM_REDUCTION_RATIO) -> tf.Tensor:
    channels = x.shape[-1]
    shared_dense_1 = layers.Dense(channels // ratio, activation="relu", use_bias=False)
    shared_dense_2 = layers.Dense(channels, use_bias=False)

    # Average-pool path
    avg = layers.GlobalAveragePooling2D()(x)              # (B, C)
    avg = shared_dense_1(avg)
    avg = shared_dense_2(avg)

    # Max-pool path
    mx  = layers.GlobalMaxPooling2D()(x)
    mx  = shared_dense_1(mx)
    mx  = shared_dense_2(mx)

    scale = layers.Activation("sigmoid")(avg + mx)        # (B, C)
    scale = layers.Reshape((1, 1, channels))(scale)       # (B, 1, 1, C)
    return layers.Multiply()([x, scale])


def spatial_attention(x: tf.Tensor, kernel_size: int = 7) -> tf.Tensor:
    # Reduce along channel axis
    avg = tf.reduce_mean(x, axis=-1, keepdims=True)       # (B, H, W, 1)
    mx  = tf.reduce_max(x, axis=-1, keepdims=True)        # (B, H, W, 1)
    concat = layers.Concatenate(axis=-1)([avg, mx])       # (B, H, W, 2)

    scale = layers.Conv2D(1, kernel_size, padding="same",
                          activation="sigmoid", use_bias=False)(concat)
    return layers.Multiply()([x, scale])


def cbam_block(x: tf.Tensor) -> tf.Tensor:
    x = channel_attention(x)
    x = spatial_attention(x)
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_cbam_cnn(input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
                   l2_reg: float = 1e-4) -> tf.keras.Model:

    reg = regularizers.l2(l2_reg)
    inp = layers.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D(2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inp, out, name="CNN_CBAM")
    return model


def compile_cbam_cnn(model: tf.keras.Model,
                     lr: float = config.LEARNING_RATE) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
