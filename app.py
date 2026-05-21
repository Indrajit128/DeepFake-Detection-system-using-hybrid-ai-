import os
import json
import random
import time
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Try to import predict_image from predict.py
try:
    from predict import predict_image
    HAS_ML_MODEL = True
except Exception:
    HAS_ML_MODEL = False

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/metrics")
def metrics():
    """Return model comparison metrics from comparison.json."""
    metrics_path = os.path.join(
        os.path.dirname(__file__), "outputs", "results", "comparison.json"
    )
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "metrics": data})
    else:
        # Return mock metrics for Vercel
        mock = {
            "SVM":   {"accuracy": 0.9935, "precision": 1.0000, "recall": 0.9871, "f1": 0.9935, "roc_auc": 0.9957},
            "Ridge": {"accuracy": 0.9935, "precision": 1.0000, "recall": 0.9871, "f1": 0.9935, "roc_auc": 0.9955},
            "Lasso": {"accuracy": 0.9839, "precision": 0.9808, "recall": 0.9871, "f1": 0.9839, "roc_auc": 0.9965},
            "CNN":   {"accuracy": 0.9700, "precision": 0.9710, "recall": 0.9680, "f1": 0.9695, "roc_auc": 0.9800},
            "CBAM":  {"accuracy": 0.9850, "precision": 0.9860, "recall": 0.9840, "f1": 0.9850, "roc_auc": 0.9920},
        }
        return jsonify({"success": True, "metrics": mock, "mocked": True})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Supported: jpg, jpeg, png, bmp, webp"}), 400

    model_tag = request.form.get('model', 'cbam').lower()
    VALID_MODELS = ['cbam', 'cnn', 'svm', 'ridge', 'lasso']
    if model_tag not in VALID_MODELS:
        model_tag = 'cbam'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if HAS_ML_MODEL:
            label, conf = predict_image(filepath, model_tag)
            conf = float(conf)
            mocked = False
        else:
            # Mock prediction for Vercel
            time.sleep(1.2)
            mock_prob = random.uniform(0.55, 0.99)
            if random.random() > 0.5:
                label = "FAKE"
                conf = mock_prob
            else:
                label = "REAL"
                conf = 1.0 - mock_prob
            mocked = True

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "label": label,
            "confidence": conf,
            "model": model_tag.upper(),
            "mocked": mocked
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
