import os
import tensorflow as tf
import numpy as np

# Apply VRAM constraint BEFORE importing other models
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("VRAM Optimization: tf.config.experimental.set_memory_growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Running on CPU.")

from utils.data_loader import prepare_data
from utils.metrics import evaluate_model, plot_roc_curves
from models.ml_models import get_svm_model, get_ridge_model, get_lasso_model
from models.cnn_cbam import build_base_cnn, build_cnn_cbam

def main():
    BASE_DIR = "dataset"
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 16  # Small batch size to keep VRAM < 2GB
    EPOCHS = 10
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: Directory '{BASE_DIR}' not found. Please run 'python generate_dummy_data.py' first.")
        return

    print("Loading and preparing data...")
    # Load for CNN (Not flattened)
    X_train_cnn, X_test_cnn, y_train, y_test = prepare_data(BASE_DIR, IMG_SIZE, flatten_for_ml=False)
    
    # Load for ML models (Flattened)
    X_train_ml = X_train_cnn.reshape(X_train_cnn.shape[0], -1)
    X_test_ml = X_test_cnn.reshape(X_test_cnn.shape[0], -1)

    print(f"CNN Data Shape: {X_train_cnn.shape}, ML Data Shape: {X_train_ml.shape}")

    results_dict = {}
    y_true_dict = {}

    # --- 1. Traditional ML Models ---
    print("\nTraining SVM...")
    svm = get_svm_model()
    svm.fit(X_train_ml, y_train)
    svm_preds = svm.predict(X_test_ml)
    svm_probs = svm.predict_proba(X_test_ml)[:, 1]
    evaluate_model(y_test, svm_preds, svm_probs, "SVM")
    results_dict["SVM"] = svm_probs
    y_true_dict["SVM"] = y_test

    print("\nTraining Ridge Classifier...")
    ridge = get_ridge_model()
    ridge.fit(X_train_ml, y_train)
    ridge_preds = ridge.predict(X_test_ml)
    # Ridge doesn't support predict_proba natively, using decision_function
    ridge_scores = ridge.decision_function(X_test_ml)
    # Normalize decision scores to [0, 1] for AUC approximation
    ridge_probs = (ridge_scores - ridge_scores.min()) / (ridge_scores.max() - ridge_scores.min())
    evaluate_model(y_test, ridge_preds, ridge_probs, "Ridge")
    results_dict["Ridge"] = ridge_probs
    y_true_dict["Ridge"] = y_test

    print("\nTraining Lasso (L1 Logistic Regression)...")
    lasso = get_lasso_model()
    lasso.fit(X_train_ml, y_train)
    lasso_preds = lasso.predict(X_test_ml)
    lasso_probs = lasso.predict_proba(X_test_ml)[:, 1]
    evaluate_model(y_test, lasso_preds, lasso_probs, "Lasso")
    results_dict["Lasso"] = lasso_probs
    y_true_dict["Lasso"] = y_test

    # --- 2. Deep Learning Models ---
    print("\nTraining Base CNN...")
    cnn_model = build_base_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)
    
    cnn_probs = cnn_model.predict(X_test_cnn).flatten()
    cnn_preds = (cnn_probs > 0.5).astype(int)
    evaluate_model(y_test, cnn_preds, cnn_probs, "Base CNN")
    results_dict["Base CNN"] = cnn_probs
    y_true_dict["Base CNN"] = y_test

    print("\nTraining CNN + CBAM...")
    cbam_model = build_cnn_cbam(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    cbam_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cbam_model.fit(X_train_cnn, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)
    
    cbam_probs = cbam_model.predict(X_test_cnn).flatten()
    cbam_preds = (cbam_probs > 0.5).astype(int)
    evaluate_model(y_test, cbam_preds, cbam_probs, "CNN + CBAM")
    results_dict["CNN + CBAM"] = cbam_probs
    y_true_dict["CNN + CBAM"] = y_test

    # --- 3. Evaluate and Plot ---
    print("\nPlotting ROC Curves...")
    plot_roc_curves(results_dict, y_true_dict)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
