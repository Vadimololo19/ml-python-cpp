import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import time
import json
import warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("="*50)
print("ЭТАП ОБУЧЕНИЯ МОДЕЛИ (NF-UNSW-NB15.csv)")
print("="*50)

DATA_PATH = "../data/NF-UNSW-NB15-v2.csv"

print(f"Загружаем CSV данные из {DATA_PATH}...")
start_load = time.time()

df = pd.read_csv(DATA_PATH)
load_time = time.time() - start_load
print(f"Данные загружены за {load_time:.2f} секунд")
print(f"Исходный shape: {df.shape}")

print("\nРеальные имена колонок в датасете:")
print(df.columns.tolist())

target_col = 'Label'
if target_col not in df.columns:
    possible_targets = ['label', 'target', 'class']
    found = False
    for alt in possible_targets:
        if alt in df.columns:
            target_col = alt
            found = True
            break
    if not found:
        raise ValueError("Не найдена целевая колонка. Проверьте имена колонок в датасете.")

print(f"\nЦелевая переменная: '{target_col}'")
print(f"Целевые классы:\n{df[target_col].value_counts(normalize=True)}")

print("\n" + "="*50)
print("АНАЛИЗ И ОТБОР ПРИЗНАКОВ")
print("="*50)

cols_to_drop = []
for col in df.columns:
    if df[col].nunique() > df.shape[0] * 0.5:
        cols_to_drop.append(col)
        continue
    
    if df[col].isnull().mean() > 0.3:
        cols_to_drop.append(col)
        continue
    
    if col.lower() in ['timestamp', 'date', 'time', 'src_ip', 'dst_ip', 'attack']:
        cols_to_drop.append(col)

print(f"Удаляем {len(cols_to_drop)} колонок: {cols_to_drop}")

feature_cols = [col for col in df.columns if col not in cols_to_drop + [target_col]]
print(f"Остаётся {len(feature_cols)} признаков для обучения.")

numerical_cols = []
categorical_cols = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        numerical_cols.append(col)
    else:
        try:
            pd.to_numeric(df[col].iloc[:100], errors='raise')
            numerical_cols.append(col)
        except:
            categorical_cols.append(col)

print(f"- Числовых признаков: {len(numerical_cols)}")
print(f"- Категориальных признаков: {len(categorical_cols)}")

if numerical_cols:
    print("\nПримеры числовых признаков:")
    print(numerical_cols[:5])
if categorical_cols:
    print("\nПримеры категориальных признаков:")
    print(categorical_cols[:5])

SAMPLE_SIZE = 50_000  
if len(df) > SAMPLE_SIZE:
    print(f"\nПрименение стратифицированной выборки. Исходный размер: {len(df)}")
    
    class_counts = df[target_col].value_counts()
    print(f"Распределение классов перед выборкой:\n{class_counts}")
    
    if class_counts.min() < 1000:  
        print("Обнаружен сильный дисбаланс классов. Применяем undersampling...")
        
        minority_class = class_counts.idxmin()
        minority_count = class_counts.min()
        majority_class = class_counts.idxmax()
        
        df_minority = df[df[target_col] == minority_class]
        
        df_majority = df[df[target_col] == majority_class]
        df_majority_sampled = df_majority.sample(
            n=min(len(df_majority), minority_count * 20),
            random_state=42
        )
        
        df_balanced = pd.concat([df_minority, df_majority_sampled])
        df_sampled = df_balanced
        print(f"После undersampling: {len(df_sampled)} записей")
    else:
        _, df_sampled = train_test_split(
            df,
            test_size=SAMPLE_SIZE / len(df),
            random_state=42,
            stratify=df[target_col]
        )
    
    df = df_sampled
    print(f"Новый размер данных: {len(df)}")
    print(f"Распределение классов после выборки:\n{df[target_col].value_counts(normalize=True)}")

y = df[target_col]
X = df[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

print("\n" + "="*50)
print("СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ")
print("="*50)

print("Предобработка данных...")
num_imputer = SimpleImputer(strategy='median')
X_train_num = num_imputer.fit_transform(X_train[numerical_cols])
X_test_num = num_imputer.transform(X_test[numerical_cols])

if categorical_cols:
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test_cat = cat_imputer.transform(X_test[categorical_cols])
    
    X_train_cat_encoded = cat_encoder.fit_transform(X_train_cat)
    X_test_cat_encoded = cat_encoder.transform(X_test_cat)
    
    X_train_processed = np.hstack([X_train_num, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_num, X_test_cat_encoded])
    
    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        categories = cat_encoder.categories_[i]
        for category in categories:
            cat_feature_names.append(f"{col}_{category}")
    
    feature_names = numerical_cols + cat_feature_names
else:
    X_train_processed = X_train_num
    X_test_processed = X_test_num
    feature_names = numerical_cols

print(f"Количество признаков после предобработки: {X_train_processed.shape[1]}")

print("Обучение RandomForest...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

start_time = time.time()
model.fit(X_train_processed, y_train)
train_time = time.time() - start_time
print(f"Обучение завершено за {train_time:.2f} секунд")

y_pred = model.predict(X_test_processed)
print("\nОтчёт по качеству:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n" + "="*50)
print("ЭКСПОРТ МОДЕЛИ")
print("="*50)

#os.makedirs("models", exist_ok=True)
#os.makedirs("load_test", exist_ok=True)

preprocessors = {
    'num_imputer': num_imputer,
    'scaler': StandardScaler().fit(X_train_num),  
    'feature_names': feature_names,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols
}

if categorical_cols:
    preprocessors['cat_imputer'] = cat_imputer
    preprocessors['cat_encoder'] = cat_encoder

joblib.dump(preprocessors, "../models/preprocessors.pkl")
joblib.dump(model, "../models/rf_nids_csv.pkl")
print("Предобработчики и модель сохранены")

try:
    print("Экспорт в ONNX...")
    
    initial_type = [('float_input', FloatTensorType([None, X_train_processed.shape[1]]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15,
        options={id(model): {'zipmap': False}}
    )

    with open("../models/rf_nids_csv.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("ONNX модель успешно сохранена")
except Exception as e:
    print(f"Ошибка при экспорте в ONNX: {e}")
    print("Сохраняем только sklearn модель для Python сервиса")

print("\n" + "="*50)
print("ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ")
print("="*50)

sample_idx = 0
sample_data = {}

for col in numerical_cols:
    if col in X_test.columns:
        sample_data[col] = float(X_test[col].iloc[sample_idx])

for col in categorical_cols:
    if col in X_test.columns:
        sample_data[col] = str(X_test[col].iloc[sample_idx])

payload = {
    "features": sample_data
}
with open("../load_test/payload_nids.json", "w") as f:
    json.dump(payload, f, indent=2)
print("Пример payload сохранён. Содержимое:")
print(json.dumps(payload, indent=2))

metadata = {
    "feature_names": feature_names,
    "numerical_features": numerical_cols,
    "categorical_features": categorical_cols,
    "target": target_col,
    "classes": ["Normal", "Attack"],
    "model_type": "RandomForestClassifier",
    "input_shape": [1, X_train_processed.shape[1]]
}
with open("../models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Метаданные сохранены")

print("\n" + "="*60)
print("ГОТОВО! МОДЕЛЬ УСПЕШНО ОБУЧЕНА И СОХРАНЕНА")
print("="*60)
print(f"Для запуска Python сервиса: python app_nids.py")
print("="*60)
