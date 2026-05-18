# Deepfake Image Detection System

A deepfake detection pipeline that benchmarks classical ML models (SVM, Ridge, Lasso) against a baseline CNN and a CNN augmented with **CBAM** (Convolutional Block Attention Module) on a dataset of **2,041 images**.

---

## Architecture Overview

```
data/
├── real/          ← place real images here
└── fake/          ← place deepfake images here

models/
├── cnn_model.py   ← 4-block CNN baseline
├── cbam_model.py  ← CNN + CBAM attention
└── ml_models.py   ← SVM, Ridge (L2), Lasso (L1)

utils/
├── dataset.py         ← loading, augmentation, tf.data pipeline
├── feature_extractor.py ← HOG + PCA for ML models
└── evaluation.py      ← metrics, ROC curves, confusion matrix, comparison table

train_ml.py    ← train SVM / Ridge / Lasso
train_cnn.py   ← train baseline CNN
train_cbam.py  ← train CNN + CBAM
evaluate.py    ← compare all saved models on the test set
predict.py     ← single-image or folder inference
pipeline.py    ← full end-to-end run (trains all + comparison)
config.py      ← all hyper-parameters in one place
```

---

## CBAM: Convolutional Block Attention Module

After each convolutional block, CBAM applies two sequential attention gates:

```
Input feature map  (B, H, W, C)
        │
   ┌────▼────────────────┐
   │  Channel Attention  │   GlobalAvgPool + GlobalMaxPool → shared MLP → sigmoid
   └────────────────────┘
        │ (refined channels)
   ┌────▼────────────────┐
   │  Spatial Attention  │   AvgPool(C) + MaxPool(C) → concat → Conv7x7 → sigmoid
   └────────────────────┘
        │
   Attended feature map
```

This steers the network toward forgery artefacts — blending boundaries,
compression noise, and GAN-frequency patterns — rather than scene content.

---

## Setup

```bash
pip install -r requirements.txt
```

### Prepare the dataset

Copy your images into:
```
data/real/   ← 1 021 genuine images
data/fake/   ← 1 020 deepfake images
```
Any `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` file is accepted.

---

## Usage

### Train everything at once

```bash
python pipeline.py
```

### Train models individually

```bash
python train_ml.py    # SVM, Ridge, Lasso
python train_cnn.py   # Baseline CNN
python train_cbam.py  # CNN + CBAM
```

### Evaluate saved models

```bash
python evaluate.py
```

Generates `outputs/plots/` and `outputs/results/comparison.csv`.

### Predict a single image

```bash
python predict.py --image path/to/face.jpg
python predict.py --image path/to/face.jpg --model cnn
python predict.py --image path/to/face.jpg --model SVM
python predict.py --dir   path/to/folder/
```

Available `--model` values: `cbam` (default), `cnn`, `SVM`, `Ridge`, `Lasso`.

---

## Model Details

| Model      | Input          | Key component                     |
|------------|----------------|-----------------------------------|
| SVM        | HOG + PCA(128) | RBF kernel, Platt probability      |
| Ridge      | HOG + PCA(128) | L2 linear classifier              |
| Lasso      | HOG + PCA(128) | L1 logistic regression            |
| CNN        | 128×128×3      | 4× Conv-BN-ReLU-MaxPool, GAP, FC  |
| CNN+CBAM   | 128×128×3      | Same + CBAM after every block     |

All CNN models target **<2 GB VRAM** (128×128 input, batch=32).

---

## Evaluation Metrics

Accuracy · Precision · Recall · F1-score · ROC-AUC

Results are written to `outputs/results/comparison.csv` and `comparison.json`.
Plots are saved to `outputs/plots/`.

---

## Configuration

All parameters live in [config.py](config.py):

| Parameter          | Default  | Description                        |
|--------------------|----------|------------------------------------|
| `IMG_SIZE`         | 128      | Input resolution (px)              |
| `BATCH_SIZE`       | 32       | Training batch size                |
| `EPOCHS_CNN`       | 30       | Max epochs (early-stop applies)    |
| `EPOCHS_CBAM`      | 30       | Max epochs for CBAM model          |
| `LEARNING_RATE`    | 1e-4     | Adam initial LR                    |
| `CBAM_REDUCTION_RATIO` | 8   | Channel compression in CBAM MLP   |
| `PCA_COMPONENTS`   | 128      | PCA dims for ML feature vectors    |

---

## Tech Stack

- **TensorFlow / Keras** — CNN, CBAM, training loop
- **scikit-learn** — SVM, Ridge, Lasso, PCA, metrics
- **scikit-image** — HOG feature extraction
- **Pillow** — image I/O
- **Matplotlib** — plots
# DeepFake-Detection-system-using-hybrid-ai-
