"""
Baseline CNN for deepfake detection.

Architecture (fits in <2 GB VRAM at batch=32, input=128x128):
  4 convolutional blocks (Conv → BN → ReLU → MaxPool)
  Global Average Pooling
  Dense(256) → Dropout(0.5) → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import config


def build_cnn(input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
              l2_reg: float = 1e-4) -> tf.keras.Model:

    reg = regularizers.l2(l2_reg)
    inp = layers.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inp, out, name="CNN_Baseline")
    return model


def compile_cnn(model: tf.keras.Model,
                lr: float = config.LEARNING_RATE) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
