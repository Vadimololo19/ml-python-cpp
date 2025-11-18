import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import time
import psutil
import os
import json
import traceback
import warnings

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

app = Flask(__name__)

print("Загрузка артефактов...")

METADATA_PATH = os.path.join(BASE_DIR, "../models/metadata.json")
if not os.path.exists(METADATA_PATH):
    raise RuntimeError("Метаданные не найдены! Выполните сначала train_nids_csv.py")

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names"]
numerical_features = metadata["numerical_features"]
categorical_features = metadata["categorical_features"]
input_shape = metadata["input_shape"]

print(f"Загружены метаданные:")
print(f"- Всего признаков: {len(feature_names)}")
print(f"- Числовых признаков: {len(numerical_features)}")
print(f"- Категориальных признаков: {len(categorical_features)}")

PREPROCESSORS_PATH = os.path.join(BASE_DIR, "../models/preprocessors.pkl")
if not os.path.exists(PREPROCESSORS_PATH):
    raise RuntimeError("Предобработчики не найдены! Выполните сначала train_nids_csv.py")

preprocessors = joblib.load(PREPROCESSORS_PATH)
num_imputer = preprocessors['num_imputer']
scaler = preprocessors['scaler']

if categorical_features:
    cat_imputer = preprocessors['cat_imputer']
    cat_encoder = preprocessors['cat_encoder']

MODEL_PATH = os.path.join(BASE_DIR, "../models/rf_nids_csv.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Модель не найдена! Выполните сначала train_nids_csv.py")

model = joblib.load(MODEL_PATH)
print(f"Модель загружена. Память: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

REQUEST_COUNT = 0
TOTAL_INFERENCE_TIME = 0.0

@app.route('/health', methods=['GET'])
def health():
    avg_time = TOTAL_INFERENCE_TIME / REQUEST_COUNT if REQUEST_COUNT > 0 else 0
    return jsonify({
        "status": "ok",
        "model": "RandomForest NIDS (CSV) - FINAL",
        "requests_processed": REQUEST_COUNT,
        "avg_inference_time_ms": avg_time,
        "memory_mb": psutil.Process().memory_info().rss / 1024**2,
        "feature_count": len(feature_names)
    })

@app.route('/predict', methods=['POST'])
def predict():
    global REQUEST_COUNT, TOTAL_INFERENCE_TIME
    
    try:
        start_time = time.perf_counter()
        
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                "error": "Требуется JSON в формате: {'features': {<имена_признаков>: <значения>}}"
            }), 400
        
        input_features = data['features']
        
        num_values = []
        for feature in numerical_features:
            value = input_features.get(feature, 0.0)
            try:
                if isinstance(value, str):
                    value = float(value.replace(',', '.'))
                elif isinstance(value, bool):
                    value = 1.0 if value else 0.0
                num_values.append(float(value))
            except (ValueError, TypeError):
                num_values.append(0.0)
        
        cat_values = []
        if categorical_features:
            for feature in categorical_features:
                value = input_features.get(feature, "Missing")
                cat_values.append(str(value))
            
            cat_array = np.array(cat_values).reshape(1, -1)
            cat_imputed = cat_imputer.transform(cat_array)
            cat_encoded = cat_encoder.transform(cat_imputed)
        
        num_array = np.array(num_values).reshape(1, -1)
        num_imputed = num_imputer.transform(num_array)
        num_scaled = scaler.transform(num_imputed)
        
        if categorical_features:
            features_processed = np.hstack([num_scaled, cat_encoded])
        else:
            features_processed = num_scaled
        
        if features_processed.shape[1] != input_shape[1]:
            return jsonify({
                "error": f"Несоответствие количества признаков: {features_processed.shape[1]} вместо {input_shape[1]}",
                "expected_features": feature_names
            }), 400
        
        pred_proba = model.predict_proba(features_processed)[0][1]  # вероятность атаки
        inference_time = (time.perf_counter() - start_time) * 1000  # мс
        
        global REQUEST_COUNT, TOTAL_INFERENCE_TIME
        REQUEST_COUNT += 1
        TOTAL_INFERENCE_TIME += inference_time
        
        return jsonify({
            "prediction": float(pred_proba),
            "is_attack": bool(pred_proba > 0.5),
            "inference_time_ms": inference_time,
            "model_version": "rf_nids_csv_v3_final",
            "features_used": len(feature_names)
        })
    
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Ошибка инференса: {error_msg}")
        return jsonify({
            "error": str(e),
            "traceback": error_msg[:500]
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)
