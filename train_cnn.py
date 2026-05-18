"""
Train the baseline CNN model.

Usage:
    python train_cnn.py
"""

import os
import numpy as np
import tensorflow as tf

from utils.dataset import load_dataset, split_dataset, make_tf_datasets
from utils.evaluation import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_training_history,
)
from models.cnn_model import build_cnn, compile_cnn
import config


def main():
    tf.random.set_seed(config.SEED)

    # 1. Data
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    train_ds, val_ds, test_ds = make_tf_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # 2. Model
    model = build_cnn()
    model = compile_cnn(model)
    model.summary()

    # 3. Callbacks
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.MODELS_DIR, "cnn_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    # 4. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_CNN,
        callbacks=callbacks,
    )

    plot_training_history(history, "CNN")

    # 5. Evaluate on test set
    best_model = tf.keras.models.load_model(ckpt_path)
    y_prob = best_model.predict(X_test, batch_size=config.BATCH_SIZE).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics("CNN (test set)", metrics)
    plot_confusion_matrix(y_test, y_pred, "CNN")

    return metrics, y_test, y_prob


if __name__ == "__main__":
    main()
