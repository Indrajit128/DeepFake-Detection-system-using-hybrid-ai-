"""
Dataset loading, preprocessing, and splitting utilities.
Targets the 2,041-image real/fake corpus (placed in data/real and data/fake).
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

import config


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def load_dataset(real_dir: str = config.REAL_DIR,
                 fake_dir: str = config.FAKE_DIR):
    """
    Returns (images, labels) as float32 arrays.
    Labels: 0 = real, 1 = fake.
    """
    images, labels = [], []

    for label, folder in enumerate([real_dir, fake_dir]):
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Directory not found: {folder}\n"
                "Place real images in data/real/ and fake images in data/fake/"
            )
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXT:
                continue
            fpath = os.path.join(folder, fname)
            try:
                images.append(_load_image(fpath))
                labels.append(label)
            except Exception as e:
                print(f"[WARN] Skipping {fpath}: {e}")

    if not images:
        raise RuntimeError("No images loaded. Check data/real and data/fake directories.")

    X = np.stack(images, axis=0)   # (N, H, W, C)
    y = np.array(labels, dtype=np.int32)
    print(f"Loaded {len(y)} images — real: {(y==0).sum()}, fake: {(y==1).sum()}")
    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray):
    """70 / 15 / 15  train / val / test split (stratified)."""
    test_size  = 1.0 - config.TRAIN_SPLIT
    val_ratio  = config.VAL_SPLIT / test_size   # val fraction of the (val+test) pool

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size, random_state=config.SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1.0 - val_ratio,
        random_state=config.SEED, stratify=y_tmp
    )

    print(f"Split → train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_tf_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """Wrap numpy arrays in tf.data.Dataset with augmentation on the training set."""

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(y_train), seed=config.SEED)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds
