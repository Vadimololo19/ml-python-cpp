# python/train.py
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import sys

# === 1. Загрузка данных ===
print("📥 Загружаем California Housing...")
data = fetch_california_housing(as_frame=True)
df = data.frame

print("📊 Shape:", df.shape)
print(df.head(3))

# === 2. Разделение ===
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Обучение модели ===
print("🧠 Обучаем RandomForest...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# === 4. Оценка ===
y_pred = model.predict(X_test)

# Совместимость: проверяем наличие root_mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
    rmse = root_mean_squared_error(y_test, y_pred)
except ImportError:
    # Для старых версий sklearn (<1.4)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R²:   {r2:.4f}")

# === 5. Сохранение ===
joblib.dump(model, "model.pkl")
print("💾 Модель сохранена как model.pkl")

# === 6. Экспорт в ONNX ===
print("📤 Экспортируем в ONNX...")
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ ONNX модель сохранена как model.onnx")
