# python/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Загружаем модель один раз при старте
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель {model_path} не найдена. Запустите train.py сначала.")

model = joblib.load(model_path)
print("✅ Модель загружена")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Ожидался JSON: {\"features\": [...]}"}), 400
        
        features = np.array(data['features'], dtype=np.float32)
        if features.shape[0] != 8:
            return jsonify({"error": f"Ожидалось 8 фич, получено {features.shape[0]}"}), 400

        pred = model.predict(features.reshape(1, -1))[0]
        return jsonify({
            "prediction": float(pred),
            "model": "RandomForest (Python)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
