import os
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
except ImportError:
    HAS_ML_MODEL = False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if HAS_ML_MODEL:
                # Use actual model
                label, conf = predict_image(filepath, "cbam")
                # Ensure conf is a normal float (not numpy float)
                conf = float(conf)
            else:
                # Mock prediction for Vercel
                time.sleep(1.5) # Simulate processing time
                mock_prob = random.random()
                label = "FAKE" if mock_prob >= 0.5 else "REAL"
                conf = mock_prob
            
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify({
                "label": label,
                "confidence": conf,
                "mocked": not HAS_ML_MODEL
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
