"""
Hand-crafted feature extraction for the classical ML pipeline.
HOG descriptors + PCA keep the feature vector compact enough for SVM/Ridge/Lasso.
"""

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

import config


def extract_hog(images: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from a batch of images (N, H, W, C) in [0,1].
    Returns (N, D) float32 array.
    """
    feats = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        h = hog(
            img_uint8,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            channel_axis=-1,
            feature_vector=True,
        )
        feats.append(h)
    return np.array(feats, dtype=np.float32)


def flatten_pixels(images: np.ndarray) -> np.ndarray:
    """Simple pixel flattening as an alternative low-level feature."""
    return images.reshape(len(images), -1)


class FeaturePipeline:
    """
    StandardScaler + PCA wrapper that can be fit on train and applied to val/test.
    Serialises to outputs/saved_models/ for reuse at inference time.
    """

    def __init__(self, n_components: int = config.PCA_COMPONENTS):
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=n_components, random_state=config.SEED)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_scaled)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

    def save(self, tag: str = "hog"):
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(config.MODELS_DIR, f"{tag}_scaler.pkl"))
        joblib.dump(self.pca,    os.path.join(config.MODELS_DIR, f"{tag}_pca.pkl"))
        print(f"Feature pipeline saved ({tag})")

    def load(self, tag: str = "hog"):
        self.scaler = joblib.load(os.path.join(config.MODELS_DIR, f"{tag}_scaler.pkl"))
        self.pca    = joblib.load(os.path.join(config.MODELS_DIR, f"{tag}_pca.pkl"))


def prepare_ml_features(X_train, X_val, X_test):
    """
    Full pipeline: HOG extraction → StandardScaler → PCA.
    Returns transformed train/val/test arrays and the fitted pipeline.
    """
    print("Extracting HOG features...")
    F_train = extract_hog(X_train)
    F_val   = extract_hog(X_val)
    F_test  = extract_hog(X_test)

    print(f"HOG raw dim: {F_train.shape[1]}  →  reducing to {config.PCA_COMPONENTS} via PCA")
    pipe = FeaturePipeline()
    F_train_r = pipe.fit_transform(F_train)
    F_val_r   = pipe.transform(F_val)
    F_test_r  = pipe.transform(F_test)
    pipe.save("hog")

    return F_train_r, F_val_r, F_test_r, pipe
