"""
Single-image or directory inference using the best-saved model.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model cnn    # or cbam / SVM / Ridge / Lasso
    python predict.py --dir path/to/folder/
"""

import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf

import config
from models.ml_models import load_ml_model, get_predict_proba, ALL_ML_MODELS
from utils.feature_extractor import FeaturePipeline


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def predict_image(image_path: str, model_tag: str = "cbam"):
    """
    Returns (label_str, confidence).
    label_str: "FAKE" or "REAL"
    confidence: probability that the image is fake (0-1)
    """
    img = load_image(image_path)

    if model_tag in ("cnn", "cbam"):
        ckpt_name = "cnn_best.keras" if model_tag == "cnn" else "cbam_best.keras"
        ckpt_path = os.path.join(config.MODELS_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No trained model at {ckpt_path}. Run the training script first.")
        model  = tf.keras.models.load_model(ckpt_path)
        prob   = float(model.predict(img[None], verbose=0).ravel()[0])
    else:
        # ML model
        if model_tag not in ALL_ML_MODELS:
            raise ValueError(f"Unknown model '{model_tag}'. Choose from: cnn, cbam, "
                             + ", ".join(ALL_ML_MODELS.keys()))
        from skimage.feature import hog
        img_u = (img * 255).astype("uint8")
        feat  = hog(img_u, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), channel_axis=-1, feature_vector=True)
        pipe = FeaturePipeline()
        pipe.load("hog")
        feat_r = pipe.transform(feat[None])
        model  = load_ml_model(model_tag)
        prob   = float(get_predict_proba(model, feat_r)[0])

    label = "FAKE" if prob >= 0.5 else "REAL"
    return label, prob


def main():
    parser = argparse.ArgumentParser(description="Deepfake image predictor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--dir",   type=str, help="Path to a directory of images")
    parser.add_argument(
        "--model", type=str, default="cbam",
        choices=["cnn", "cbam"] + list(ALL_ML_MODELS.keys()),
        help="Which model to use for inference (default: cbam)",
    )
    args = parser.parse_args()

    if args.image:
        label, conf = predict_image(args.image, args.model)
        print(f"\n{'-'*40}")
        print(f"  File   : {os.path.basename(args.image)}")
        print(f"  Model  : {args.model.upper()}")
        print(f"  Result : {label}  (confidence: {conf:.4f})")
        print(f"{'-'*40}\n")
    else:
        paths = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ]
        print(f"\nRunning {args.model.upper()} on {len(paths)} images in {args.dir}\n")
        for p in paths:
            try:
                label, conf = predict_image(p, args.model)
                print(f"  {os.path.basename(p):<40} {label}  ({conf:.4f})")
            except Exception as e:
                print(f"  {os.path.basename(p):<40} ERROR: {e}")


if __name__ == "__main__":
    main()
