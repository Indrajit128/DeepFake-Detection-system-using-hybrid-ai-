import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(base_dir="dataset", img_size=(128, 128)):
    """Loads images and labels from the specified directory."""
    images = []
    labels = []
    
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError(f"Dataset directories not found. Expected {real_dir} and {fake_dir}")

    # Load real images (Label: 0)
    for file in os.listdir(real_dir):
        img_path = os.path.join(real_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(0)

    # Load fake images (Label: 1)
    for file in os.listdir(fake_dir):
        img_path = os.path.join(fake_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(1)

    return np.array(images), np.array(labels)

def prepare_data(base_dir="dataset", img_size=(128, 128), test_size=0.2, flatten_for_ml=False):
    """Prepares train/test split. Can optionally flatten images for ML models."""
    X, y = load_data(base_dir, img_size)
    
    # Normalize images to [0, 1] for both ML and CNN
    X = X.astype('float32') / 255.0

    if flatten_for_ml:
        # Flatten images from (N, H, W, C) to (N, H*W*C)
        X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test
