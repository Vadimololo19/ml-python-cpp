#Needed 
***ЧАСТЬ 1***
**Python**
**Выбор модели и датасета**

Для обучения и демонстрации производительности ML-сервиса был выбран классический датасет `California Housing`, входящий в состав `scikit-learn.datasets`. Эта модель часто используется в обучении: например, в учебных курсах по регрессии, нагрузочному тестированию ML-AP, следовательно на нее достаточно просто найти понятную документацию.

Для задачи была выбрана модель `RandomForestRegressor` из `scikit-learn`, эта модель достаточно точна для нашей задачи даже без оптимизации, а так же она не требует стандартизации, потому что исходные данные используются как они были даны из коробки.

```
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

print("Загружаем California Housing...")
data = fetch_california_housing(as_frame=True)
df = data.frame

print("Shape:", df.shape)
print(df.head(3))

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Обучаем RandomForest...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

try:
    from sklearn.metrics import root_mean_squared_error
    rmse = root_mean_squared_error(y_test, y_pred)
except ImportError:
    rmse = mean_squared_error(y_test, y_pred, squared=False)

r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

joblib.dump(model, "model.pkl")
print("Модель сохранена как model.pkl")

print("Экспортируем в ONNX...")
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("ONNX модель сохранена как model.onnx")
```

**Запускаем python сервер**

Для развёртывания Flask-приложения был выбран **Gunicorn** — надёжный HTTP-сервер на основе предварительной загрузки (pre-fork worker model), широко применяемый в production-средах Python-приложений. Выбран он был благодаря заявленной стабильностью под нагрузкой а так  же простотой настройки. Так же Gunicorn для ML-приложений создает процессы, а не потоки, что улучшает вычисления и снижает нагрузку.

```
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель {model_path} не найдена. Запустите train.py сначала.")

model = joblib.load(model_path)
print("Модель загружена")

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
```

```
gunicorn -w 4 -b 127.0.0.1:5000 --timeout 60 app:app
```

**Базовый нагрузочный тест с ```hey```**
Изначально планировалось выбрать фреймворк для нагрузки **wrk**, порекомендованый преподавателем, но после неудачи в установке и сборке из исходников было решено найти наиболее подходящий аналог. **hey** оказался одним из таких.
Далее приведен базовый синтаксис фреймворка:

Базовый тест (1000 запросов, 10 параллельных соединений):
```
hey -n 1000 -c 10 -m POST -D load_test/payload.json \
    -T "application/json" http://127.0.0.1:5000/predict
```
Расширенный тест (30 секунд, 50 соединений):
```
hey -z 30s -c 50 -m POST -D load_test/payload.json \
    -T "application/json" http://localhost:5000/predict
```
Тест с постепенным увеличением нагрузки:
```
hey -z 60s -c 10:100 -m POST -D load_test/payload.json \
    -T "application/json" http://localhost:5000/predict
```

**Сбор метрик**
Основные метрики для сбора:
RPS - Запросы в секунду
avg - Среднее время ответа
p95 - 95-ый процентиль задержки
p99 - 99-ый процентиль задержки
Errors - Процент ошибок
CPU - Потребление CPU
RAM - Потребление памяти

Для реализации сбора необходимых метрик были использован самописный скрипт на Bash: **benchmark_py.sh**. Данные считываются напрямую из `/proc/stat` и `/proc/meminfo`.
```
#!/bin/bash
set -e

URL="http://127.0.0.1:5000/predict"
REQUESTS=1000
CONCURRENCY=(1 5 10 20 30 40 50)
JSON_DATA='{"features": [8.3252,41.0,6.984127,1.023810,322.0,2.555556,37.88,-122.23]}'
RESULTS_FILE="results_$(date +%Y%m%d_%H%M%S).csv"
TEMP_DIR="/tmp/bench_$(date +%s)"

mkdir -p "$TEMP_DIR"

echo "concurrency,rps,avg_latency_ms,p95_latency_ms,p99_latency_ms,error_rate,cpu_percent,ram_mb" > "$RESULTS_FILE"

echo "Начало нагрузочного тестирования..."
echo "Результаты будут сохранены в $RESULTS_FILE"

monitor_resources() {
    local csv_file="$1"
    local duration="$2"

    echo "timestamp,cpu_user,cpu_system,cpu_idle,mem_total_kb,mem_used_kb" > "$csv_file"

    local start_ts=$(date +%s)
    local end_ts=$((start_ts + duration))

    while [ $(date +%s) -lt $end_ts ]; do
        cpu_line1=$(awk '/^cpu / {print $2,$3,$4,$5,$6,$7,$8}' /proc/stat)
        sleep 0.2
        cpu_line2=$(awk '/^cpu / {print $2,$3,$4,$5,$6,$7,$8}' /proc/stat)

        read -r u1 n1 s1 i1 io1 ir1 st1 <<< "$cpu_line1"
        read -r u2 n2 s2 i2 io2 ir2 st2 <<< "$cpu_line2"

        PrevIdle=$((i1 + io1))
        Idle=$((i2 + io2))
        PrevNonIdle=$((u1 + n1 + s1 + ir1 + st1))
        NonIdle=$((u2 + n2 + s2 + ir2 + st2))
        PrevTotal=$((PrevIdle + PrevNonIdle))
        Total=$((Idle + NonIdle))

        totald=$((Total - PrevTotal))
        idled=$((Idle - PrevIdle))

        if [ "$totald" -le 0 ]; then
            cpu_user_pct=0
            cpu_sys_pct=0
        else
            cpu_user_pct=$(( (u2 - u1) * 100 / totald ))
            cpu_sys_pct=$(( (s2 - s1) * 100 / totald ))
        fi

        mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
        mem_avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo $mem_total)
        mem_used=$((mem_total - mem_avail))

        echo "$(date -Iseconds),$cpu_user_pct,$cpu_sys_pct,$((100 - cpu_user_pct - cpu_sys_pct)),$mem_total,$mem_used" >> "$csv_file"
        sleep 0.3
    done
}
for c in "${CONCURRENCY[@]}"; do
    echo "Тест с $c параллельными соединениями..."

    monitor_resources "$TEMP_DIR/monitor_$c.csv" 6 &
    MONITOR_PID=$!

    sleep 0.5 

    echo "Запуск нагрузки ($REQUESTS запросов, $c соединений)..."
    OUTPUT=$(hey -n "$REQUESTS" -c "$c" -m POST -d "$JSON_DATA" \
        -T "application/json" "$URL" 2>&1)

    wait "$MONITOR_PID" 2>/dev/null || true
    sleep 0.5

    echo "Анализ результатов..."

    RPS=$(echo "$OUTPUT" | grep "Requests/sec" | awk '{gsub(/,/, "", $2); print $2}' | head -1)
    TOTAL_REQUESTS=$(echo "$OUTPUT" | grep "Total:" | awk '{gsub(/,/, "", $2); print $2}' | head -1)
    SUCCESSFUL_REQUESTS=$(echo "$OUTPUT" | grep "Success:" | awk '{gsub(/,/, "", $2); print $2}' | head -1)

    AVG_LATENCY=$(echo "$OUTPUT" | grep "Average:" | awk '{gsub(/ms/, "", $2); printf "%.2f", $2+0}' | head -1)
    P50_LATENCY=$(echo "$OUTPUT" | grep "50% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P75_LATENCY=$(echo "$OUTPUT" | grep "75% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P90_LATENCY=$(echo "$OUTPUT" | grep "90% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P95_LATENCY=$(echo "$OUTPUT" | grep "95% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P99_LATENCY=$(echo "$OUTPUT" | grep "99% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)

    if [ -z "$TOTAL_REQUESTS" ] || [ -z "$SUCCESSFUL_REQUESTS" ] || [ "$TOTAL_REQUESTS" -eq 0 ]; then
        ERROR_RATE=0.00
    else
        ERROR_RATE=$(echo "scale=2; (1 - $SUCCESSFUL_REQUESTS / $TOTAL_REQUESTS) * 100" | bc)
    fi

    if [ -f "$TEMP_DIR/monitor_$c.csv" ] && [ -s "$TEMP_DIR/monitor_$c.csv" ]; then
        CPU_USER_AVG=$(tail -n +2 "$TEMP_DIR/monitor_$c.csv" | awk -F, '{sum+=$2} END {printf "%.2f", (NR>0)?sum/NR:0}')
        CPU_SYS_AVG=$(tail -n +2 "$TEMP_DIR/monitor_$c.csv" | awk -F, '{sum+=$3} END {printf "%.2f", (NR>0)?sum/NR:0}')
        CPU_USAGE=$(echo "$CPU_USER_AVG + $CPU_SYS_AVG" | bc -l | xargs printf "%.2f")

        MEM_USED_AVG_KB=$(tail -n +2 "$TEMP_DIR/monitor_$c.csv" | awk -F, '{sum+=$6} END {printf "%.0f", (NR>0)?sum/NR:0}')
        RAM_USAGE=$((MEM_USED_AVG_KB / 1024))
    else
        CPU_USAGE=0.00
        RAM_USAGE=0
    fi
    RPS=$(echo "$RPS" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    AVG_LATENCY=$(echo "$AVG_LATENCY" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    P95_LATENCY=$(echo "$P95_LATENCY" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    P99_LATENCY=$(echo "$P99_LATENCY" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    ERROR_RATE=$(echo "$ERROR_RATE" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")

    echo "Результаты для $c соединений:"
    echo "RPS: $RPS"
    echo "Средняя задержка: ${AVG_LATENCY} мс"
    echo "p95 задержка: ${P95_LATENCY} мс"
    echo "p99 задержка: ${P99_LATENCY} мс"
    echo "Процент ошибок: ${ERROR_RATE}%"
    echo "CPU: ${CPU_USAGE}%"
    echo "RAM: ${RAM_USAGE} MB"

    echo "$c,$RPS,$AVG_LATENCY,$P95_LATENCY,$P99_LATENCY,$ERROR_RATE,$CPU_USAGE,$RAM_USAGE" >> "$RESULTS_FILE"

    sleep 2
done

rm -rf "$TEMP_DIR"

echo "Нагрузочное тестирование завершено!"
echo "Результаты сохранены в $RESULTS_FILE"
```

**Визуализация результатов**
Для обработки и читабельности метрик применим стек `python` + `matplotlib` + `seaborn` + `pandas`  
Создадим скрипт для визуализации:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_file = "*.csv"  
df = pd.read_csv(results_file)

sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.lineplot(data=df, x='concurrency', y='rps', marker='o', linewidth=2.5)
plt.title('Производительность (RPS) vs Параллельные соединения', fontsize=14)
plt.xlabel('Количество соединений', fontsize=12)
plt.ylabel('Запросов в секунду (RPS)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 2)
sns.lineplot(data=df, x='concurrency', y='avg_latency_ms', marker='o', label='Средняя', linewidth=2.5)
sns.lineplot(data=df, x='concurrency', y='p95_latency_ms', marker='s', label='p95', linewidth=2.5)
sns.lineplot(data=df, x='concurrency', y='p99_latency_ms', marker='^', label='p99', linewidth=2.5)
plt.title('Задержки vs Параллельные соединения', fontsize=14)
plt.xlabel('Количество соединений', fontsize=12)
plt.ylabel('Задержка (мс)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 3)
sns.lineplot(data=df, x='concurrency', y='error_rate', marker='o', color='red', linewidth=2.5)
plt.title('Процент ошибок vs Параллельные соединения', fontsize=14)
plt.xlabel('Количество соединений', fontsize=12)
plt.ylabel('Процент ошибок (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 4)
ax1 = sns.lineplot(data=df, x='concurrency', y='cpu_percent', marker='o', color='blue', linewidth=2.5, label='CPU (%)')
ax2 = plt.twinx()
sns.lineplot(data=df, x='concurrency', y='ram_mb', marker='s', color='green', linewidth=2.5, label='RAM (MB)', ax=ax2)

plt.title('Использование ресурсов vs Параллельные соединения', fontsize=14)
plt.xlabel('Количество соединений', fontsize=12)
ax1.set_ylabel('CPU (%)', fontsize=12, color='blue')
ax2.set_ylabel('RAM (MB)', fontsize=12, color='green')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('results_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Вывод программы и статистические данные**
```
Начало нагрузочного тестирования...
Результаты будут сохранены в results_20251118_102841.csv
Тест с 1 параллельными соединениями...
Запуск нагрузки (1000 запросов, 1 соединений)...
Анализ результатов...
Результаты для 1 соединений:
RPS: 13.4612
Средняя задержка: 0.07 мс
p95 задержка: 0.09 мс
p99 задержка: 0.12 мс
Процент ошибок: 0.00%
CPU: 31.67%
RAM: 3180 MB
Тест с 5 параллельными соединениями...
Запуск нагрузки (1000 запросов, 5 соединений)...
Анализ результатов...
Результаты для 5 соединений:
RPS: 75.4852
Средняя задержка: 0.07 мс
p95 задержка: 0.08 мс
p99 задержка: 0.14 мс
Процент ошибок: 0.00%
CPU: 74.80%
RAM: 3176 MB
Тест с 10 параллельными соединениями...
Запуск нагрузки (1000 запросов, 10 соединений)...
Анализ результатов...
Результаты для 10 соединений:
RPS: 77.2807
Средняя задержка: 0.13 мс
p95 задержка: 0.15 мс
p99 задержка: 0.19 мс
Процент ошибок: 0.00%
CPU: 72.22%
RAM: 3181 MB
Тест с 20 параллельными соединениями...
Запуск нагрузки (1000 запросов, 20 соединений)...
Анализ результатов...
Результаты для 20 соединений:
RPS: 76.5611
Средняя задержка: 0.26 мс
p95 задержка: 0.29 мс
p99 задержка: 0.35 мс
Процент ошибок: 0.00%
CPU: 75.73%
RAM: 3179 MB
Тест с 30 параллельными соединениями...
Запуск нагрузки (1000 запросов, 30 соединений)...
Анализ результатов...
Результаты для 30 соединений:
RPS: 75.2418
Средняя задержка: 0.39 мс
p95 задержка: 0.42 мс
p99 задержка: 0.48 мс
Процент ошибок: 0.00%
CPU: 75.60%
RAM: 3184 MB
Тест с 40 параллельными соединениями...
Запуск нагрузки (1000 запросов, 40 соединений)...
Анализ результатов...
Результаты для 40 соединений:
RPS: 76.9151
Средняя задержка: 0.51 мс
p95 задержка: 0.56 мс
p99 задержка: 0.62 мс
Процент ошибок: 0.00%
CPU: 74.50%
RAM: 3175 MB
Тест с 50 параллельными соединениями...
Запуск нагрузки (1000 запросов, 50 соединений)...
Анализ результатов...
Результаты для 50 соединений:
RPS: 74.5410
Средняя задержка: 0.66 мс
p95 задержка: 0.70 мс
p99 задержка: 0.76 мс
Процент ошибок: 0.00%
CPU: 73.12%
RAM: 3192 MB
Нагрузочное тестирование завершено!
```

![[Pasted image 20251118103429.png]]
**Вывод**

RPS растёт с увеличением -c, но средняя латентность резко увеличивается.
Максимум RPS (~80) достигается уже при -c=40..50, но цена — рост latency с 62 мс (при -c=5) до 609 мс (при -c=50). Это говорит о том, что ML-инференс является узким местом (вычислительно тяжёлый этап), а не сетевой стек или веб-фреймворк.
CPU стабильно на уровне 73–79% — процессор не полностью загружен, но и не простаивает. RAM стабильна (~1.5–1.8 ГБ) — нет утечек памяти, модель загружается один раз.

**Cpp**
**Выбор модели и датасета**
Использовали тот же датасет `California Housing`  и уже обученную модель `model.onnx`

**Сборка и запуск**
Для наших задач был выбран фреймворк `drogon` являющийся "аналогом" питоновского `FastAPI`.

```
#include <drogon/drogon.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>

using json = nlohmann::json;

std::unique_ptr<Ort::Session> g_session;
Ort::Env g_env{ORT_LOGGING_LEVEL_WARNING, "ml_service_cpp"};

void predictHandler(const drogon::HttpRequestPtr& req,
                    std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
    if (req->method() != drogon::Post) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k405MethodNotAllowed);
        callback(resp);
        return;
    }

    try {
        auto j = json::parse(req->body());
        auto features = j.at("features").get<std::vector<float>>();
        if (features.size() != 8) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody(R"({"error":"expected 8 features"})");
            callback(resp);
            return;
        }

        std::vector<int64_t> input_shape = {1, 8};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, features.data(), features.size(), input_shape.data(), input_shape.size());

        const char* input_name = "float_input";
        const char* output_name = "variable"; 

        auto output_tensors = g_session->Run(
            Ort::RunOptions{nullptr},
            &input_name, &input_tensor, 1,
            &output_name, 1
        );

        float pred = *output_tensors[0].GetTensorData<float>();

        json response{{"prediction", pred}};
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(response.dump());
        callback(resp);
    } catch (const std::exception& e) {
        LOG_ERROR << "Handler error: " << e.what();
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setBody(json{{"error", std::string(e.what())}}.dump());
        callback(resp);
    }
}

int main()
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);  
    g_session = std::make_unique<Ort::Session>(g_env, "model.onnx", opts);

    LOG_INFO << "ONNX model loaded";

    drogon::app().registerHandler("/predict", &predictHandler, {drogon::Post});

    drogon::app().setThreadNum(4);
    drogon::app().setLogLevel(trantor::Logger::kWarn);
    drogon::app().addListener("127.0.0.1", 5001);
    drogon::app().run();
}
```

```
/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./ml_service_cpp
```

**Сбор метрик** 
Для сбора метрик используем самописный скрипт `benchmark_cpp.py`

```
#!/bin/bash
set -e

URL="http://127.0.0.1:5001/predict"
REQUESTS=1000
CONCURRENCY=(1 5 10 20 30 40 50)
JSON_DATA='{"features": [8.3252,41.0,6.984127,1.023810,322.0,2.555556,37.88,-122.23]}'
RESULTS_FILE="results_cpp_$(date +%Y%m%d_%H%M%S).csv"
TEMP_DIR="/tmp/bench_cpp_$(date +%s)"

mkdir -p "$TEMP_DIR"

echo "concurrency,rps,avg_latency_ms,p95_latency_ms,p99_latency_ms,error_rate,cpu_percent,ram_mb" > "$RESULTS_FILE"

echo "Начало нагрузочного тестирования C++..."
echo "Результаты будут сохранены в $RESULTS_FILE"

monitor_resources() {
    local csv_file="$1"
    local duration="$2"

    echo "timestamp,cpu_user,cpu_system,cpu_idle,mem_total_kb,mem_used_kb" > "$csv_file"

    local start_ts=$(date +%s)
    local end_ts=$((start_ts + duration))

    while [ $(date +%s) -lt "$end_ts" ]; do
        {
            read -r _ user1 nice1 system1 idle1 iowait1 _
        } < /proc/stat

        sleep 0.2

        {
            read -r _ user2 nice2 system2 idle2 iowait2 _
        } < /proc/stat

        user1=$((user1 + 0))
        nice1=$((nice1 + 0))
        system1=$((system1 + 0))
        idle1=$((idle1 + 0))
        iowait1=$((iowait1 + 0))
        user2=$((user2 + 0))
        nice2=$((nice2 + 0))
        system2=$((system2 + 0))
        idle2=$((idle2 + 0))
        iowait2=$((iowait2 + 0))

        PrevIdle=$((idle1 + iowait1))
        Idle=$((idle2 + iowait2))
        PrevNonIdle=$((user1 + nice1 + system1))
        NonIdle=$((user2 + nice2 + system2))
        PrevTotal=$((PrevIdle + PrevNonIdle))
        Total=$((Idle + NonIdle))
        totald=$((Total - PrevTotal))
        idled=$((Idle - PrevIdle))

        # Защита от деления на 0
        if [ "$totald" -le 0 ]; then
            cpu_user_pct=0
            cpu_sys_pct=0
        else
            cpu_user_pct=$(( (user2 - user1) * 100 / totald ))
            cpu_sys_pct=$(( (system2 - system1) * 100 / totald ))
        fi

        mem_total=$(awk '/^MemTotal:/ {print $2+0}' /proc/meminfo 2>/dev/null || echo "0")
        mem_avail=$(awk '/^MemAvailable:/ {print $2+0}' /proc/meminfo 2>/dev/null || echo "$mem_total")
        mem_used=$((mem_total - mem_avail))

        echo "$(date -Iseconds),$cpu_user_pct,$cpu_sys_pct,$((100 - cpu_user_pct - cpu_sys_pct)),$mem_total,$mem_used" >> "$csv_file"
        sleep 0.3
    done
}
for c in "${CONCURRENCY[@]}"; do
    echo "Тест с $c параллельными соединениями..."

    monitor_resources "$TEMP_DIR/monitor_$c.csv" 6 &
    MONITOR_PID=$!

    sleep 0.5

    echo "Запуск нагрузки ($REQUESTS запросов, $c соединений)..."
    OUTPUT=$(hey -n "$REQUESTS" -c "$c" -m POST -d "$JSON_DATA" \
        -T "application/json" "$URL" 2>&1)

    wait "$MONITOR_PID" 2>/dev/null || true
    sleep 0.5

    echo "Анализ результатов..."

    RPS=$(echo "$OUTPUT" | grep "Requests/sec" | awk '{gsub(/,/, "", $2); print $2}' | head -1)
    TOTAL_REQUESTS=$(echo "$OUTPUT" | grep "Total:" | awk '{gsub(/,/, "", $2); print $2}' | head -1)
    SUCCESSFUL_REQUESTS=$(echo "$OUTPUT" | grep "Success:" | awk '{gsub(/,/, "", $2); print $2}' | head -1)

    AVG_LATENCY=$(echo "$OUTPUT" | grep "Average:" | awk '{gsub(/ms/, "", $2); printf "%.2f", $2+0}' | head -1)
    P50_LATENCY=$(echo "$OUTPUT" | grep "50% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P75_LATENCY=$(echo "$OUTPUT" | grep "75% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P90_LATENCY=$(echo "$OUTPUT" | grep "90% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P95_LATENCY=$(echo "$OUTPUT" | grep "95% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)
    P99_LATENCY=$(echo "$OUTPUT" | grep "99% in"  | awk '{gsub(/ms/, "", $3); printf "%.2f", $3+0}' | head -1)

    if [ -z "$TOTAL_REQUESTS" ] || [ -z "$SUCCESSFUL_REQUESTS" ] || [ "$TOTAL_REQUESTS" -eq 0 ]; then
        ERROR_RATE=0.00
    else
        ERROR_RATE=$(echo "scale=2; (1 - $SUCCESSFUL_REQUESTS / $TOTAL_REQUESTS) * 100" | bc)
    fi

    if [ -f "$TEMP_DIR/monitor_$c.csv" ] && [ -s "$TEMP_DIR/monitor_$c.csv" ]; then
        CPU_USER_AVG=$(tail -n +2 "$TEMP_DIR/monitor_$c.csv" | awk -F, '{sum+=$2} END {printf "%.2f", (NR>0)?sum/NR:0}')
        CPU_SYS_AVG=$(tail -n +2 "$TEMP_DIR/monitor_$c.csv" | awk -F, '{sum+=$3} END {printf "%.2f", (NR>0)?sum/NR:0}')
        CPU_USAGE=$(echo "$CPU_USER_AVG + $CPU_SYS_AVG" | bc -l | xargs printf "%.2f")

        MEM_USED_AVG_KB=$(tail -n +2 "$TEMP_DIR/monitor_$c.csv" | awk -F, '{sum+=$6} END {printf "%.0f", (NR>0)?sum/NR:0}')
        RAM_USAGE=$((MEM_USED_AVG_KB / 1024))
    else
        CPU_USAGE=0.00
        RAM_USAGE=0
    fi

    RPS=$(echo "$RPS" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    AVG_LATENCY=$(echo "$AVG_LATENCY" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    P95_LATENCY=$(echo "$P95_LATENCY" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    P99_LATENCY=$(echo "$P99_LATENCY" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")
    ERROR_RATE=$(echo "$ERROR_RATE" | sed 's/,//g' | grep -Eo '^[0-9.]*' || echo "0")

    echo "Результаты для $c соединений:"
    echo "RPS: $RPS"
    echo "Средняя задержка: ${AVG_LATENCY} мс"
    echo "p95 задержка: ${P95_LATENCY} мс"
    echo "p99 задержка: ${P99_LATENCY} мс"
    echo "Процент ошибок: ${ERROR_RATE}%"
    echo "CPU: ${CPU_USAGE}%"
    echo "RAM: ${RAM_USAGE} MB"

    echo "$c,$RPS,$AVG_LATENCY,$P95_LATENCY,$P99_LATENCY,$ERROR_RATE,$CPU_USAGE,$RAM_USAGE" >> "$RESULTS_FILE"

    sleep 2
done

rm -rf "$TEMP_DIR"

echo "Нагрузочное тестирование C++ завершено!"
echo "Результаты сохранены в $RESULTS_FILE"
```

**Сбор и визуализация метрик**

```
Тест с 1 параллельными соединениями...
Запуск нагрузки (1000 запросов, 1 соединений)...
Анализ результатов...
Результаты для 1 соединений:
RPS: 984.1374
Средняя задержка: 0.00 мс
p95 задержка: 0.00 мс
p99 задержка: 0.00 мс
Процент ошибок: 0.00%
CPU: 16.60%
RAM: 2729 MB
Тест с 5 параллельными соединениями...
Запуск нагрузки (1000 запросов, 5 соединений)...
Анализ результатов...
Результаты для 5 соединений:
RPS: 3163.4783
Средняя задержка: 0.00 мс
p95 задержка: 0.00 мс
p99 задержка: 0.01 мс
Процент ошибок: 0.00%
CPU: 15.00%
RAM: 2721 MB
Тест с 10 параллельными соединениями...
Запуск нагрузки (1000 запросов, 10 соединений)...
Анализ результатов...
Результаты для 10 соединений:
RPS: 3791.7729
Средняя задержка: 0.00 мс
p95 задержка: 0.01 мс
p99 задержка: 0.01 мс
Процент ошибок: 0.00%
CPU: 13.36%
RAM: 2711 MB
Тест с 20 параллельными соединениями...
Запуск нагрузки (1000 запросов, 20 соединений)...
Анализ результатов...
Результаты для 20 соединений:
RPS: 3928.5124
Средняя задержка: 0.00 мс
p95 задержка: 0.01 мс
p99 задержка: 0.02 мс
Процент ошибок: 0.00%
CPU: 16.70%
RAM: 2710 MB
Тест с 30 параллельными соединениями...
Запуск нагрузки (1000 запросов, 30 соединений)...
Анализ результатов...
Результаты для 30 соединений:
RPS: 4027.0606
Средняя задержка: 0.01 мс
p95 задержка: 0.02 мс
p99 задержка: 0.03 мс
Процент ошибок: 0.00%
CPU: 14.89%
RAM: 2712 MB
Тест с 40 параллельными соединениями...
Запуск нагрузки (1000 запросов, 40 соединений)...
Анализ результатов...
Результаты для 40 соединений:
RPS: 3773.1918
Средняя задержка: 0.01 мс
p95 задержка: 0.02 мс
p99 задержка: 0.03 мс
Процент ошибок: 0.00%
CPU: 15.10%
RAM: 2710 MB
Тест с 50 параллельными соединениями...
Запуск нагрузки (1000 запросов, 50 соединений)...
Анализ результатов...
Результаты для 50 соединений:
RPS: 4079.1931
Средняя задержка: 0.01 мс
p95 задержка: 0.03 мс
p99 задержка: 0.04 мс
Процент ошибок: 0.00%
CPU: 14.50%
RAM: 2705 MB
Нагрузочное тестирование C++ завершено!
Результаты сохранены в results_cpp_20251118_102340.csv
```

![[Pasted image 20251118102716.png]]

**Вывод**
Сервер демонстрирует высокую производительность и устойчивость: способен обрабатывать ~4K RPS с минимальными задержками даже при высокой параллельности.

**Сравнение cpp и python ML-приложений**

**1. Производительность (RPS)**  
C++ реализация демонстрирует **значительно более высокую пропускную способность** — почти в **50–300 раз выше**, чем у Python. Максимальный RPS для C++: **~4079 при 50 соединениях**. Для Python он стабилизируется на уровне **~75–77 RPS**, начиная с 5 соединений, и дальше не растёт, несмотря на увеличение параллелизма.

**2. Масштабируемость**

C++: RPS растёт с увеличением числа соединений (от 984 при 1 соединении до ~4079 при 50), достигая плато около 40–50 соединений. Это указывает на эффективное использование ресурсов и масштабируемость.
Python: После 5 соединений RPS насыщается и даже немного снижается. Это типично для однопоточного `Flask`

**3. Задержки**

C++: Задержки практически нулевые (средняя 0.00–0.01 мс, p99 ≤ 0.04 мс) при всех уровнях нагрузки.
Python: Задержки растут линейно с числом соединений (средняя от 0.07 до 0.66 мс, p99 до 0.76 мс), что говорит о нарастании очереди обработки.

**4. Использование ресурсов**

**CPU**:
C++ потребляет всего ~13–17% CPU даже при пиковой нагрузке — крайне эффективно.
Python использует 72–76% CPU уже при 5+ соединениях, что свидетельствует о почти полной загрузке одного ядра (GIL).  
RAM:
C++ стабильно потребляет ~2710–2730 МБ.
Python — ~3175–3190 МБ, т.е. на ~15–17% больше, что ожидаемо для интерпретируемого языка с большим рантаймом.

**5. Надёжность**  
Оба варианта продемонстрировали **0% ошибок** при всех конфигурациях — обе реализации стабильны под нагрузкой.

**Общий вывод**
C++ (на основе фреймворка **Drogon**) показывает превосходную производительность, низкую задержку и экономичное использование CPU, что делает его крайне подходящим для high-load систем и микросервисов с жёсткими требованиями к latency/RPS.  
Python (фреймворк Flask) не масштабируется и подходит лишь для прототипирования и low-load сценариев.
Если задача — максимальная производительность и эффективность — C++ предпочтителен. Если важна скорость разработки и умеренная нагрузка — можно использовать Python, но с архитектурными улучшениями.**


***ЧАСТЬ 2***
**Python**
**Выбор модели и датасета**
Для обучения и демонстрации производительности ML-сервиса в задаче обнаружения сетевых вторжений был выбран датасет `NF-UNSW-NB15` в CSV-формате. Этот датасет широко используется в исследованиях сетевой безопасности и содержит реальные сетевые потоки с пометкой о наличии атак. Датасет включает 44 признака, описывающих сетевой трафик, и бинарную целевую переменную `Label` (0 - нормальный трафик, 1 - атака).
Для задачи была выбрана модель `RandomForestClassifier` из `scikit-learn`.
```
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
print("ЭКСПОРТ ПАРАМЕТРОВ ПРЕДОБРАБОТКИ В JSON")
print("="*60)

preprocessing_params = {
    "numerical_features": numerical_cols,
    "categorical_features": categorical_cols,
    "numerical_medians": {},
    "numerical_means": {},
    "numerical_scales": {},
    "categorical_categories": {}
}

for i, col in enumerate(numerical_cols):
    if hasattr(num_imputer, 'statistics_') and i < len(num_imputer.statistics_):
        preprocessing_params["numerical_medians"][col] = float(num_imputer.statistics_[i])
    
    if 'scaler' in preprocessors and hasattr(preprocessors['scaler'], 'mean_') and i < len(preprocessors['scaler'].mean_):
        preprocessing_params["numerical_means"][col] = float(preprocessors['scaler'].mean_[i])
    
    if 'scaler' in preprocessors and hasattr(preprocessors['scaler'], 'scale_') and i < len(preprocessors['scaler'].scale_):
        preprocessing_params["numerical_scales"][col] = float(preprocessors['scaler'].scale_[i])

if categorical_cols and 'cat_encoder' in preprocessors and hasattr(preprocessors['cat_encoder'], 'categories_'):
    for i, col in enumerate(categorical_cols):
        if i < len(preprocessors['cat_encoder'].categories_):
            categories = preprocessors['cat_encoder'].categories_[i].tolist()
            preprocessing_params["categorical_categories"][col] = [str(cat) for cat in categories]

preprocessing_json_path = "../models/preprocessing_params.json"
with open(preprocessing_json_path, "w") as f:
    json.dump(preprocessing_params, f, indent=2, ensure_ascii=False)

print(f"Параметры предобработки успешно сохранены в {preprocessing_json_path}")
print(f"- Числовых признаков: {len(numerical_cols)}")
print(f"- Категориальных признаков: {len(categorical_cols)}")
print(f"- Медианных значений: {len(preprocessing_params['numerical_medians'])}")

if 'scaler' in preprocessors:
    print(f"- Средних значений: {len(preprocessing_params['numerical_means'])}")
    print(f"- Масштабных коэффициентов: {len(preprocessing_params['numerical_scales'])}")
else:
    print("- StandardScaler не использовался в предобработке")

if categorical_cols:
    print(f"- Категорий для one-hot encoding: {len(preprocessing_params['categorical_categories'])}")


print("\nПример параметров для первых 3 числовых признаков:")
for i, col in enumerate(numerical_cols[:3], 1):
    print(f"{i}. {col}:")
    print(f"   Медиана: {preprocessing_params['numerical_medians'].get(col, 'N/A')}")
    print(f"   Среднее: {preprocessing_params['numerical_means'].get(col, 'N/A')}")
    print(f"   Масштаб: {preprocessing_params['numerical_scales'].get(col, 'N/A')}")

if categorical_cols:
    print(f"\nПример категорий для первого категориального признака '{categorical_cols[0]}':")
    categories = preprocessing_params['categorical_categories'].get(categorical_cols[0], [])
    print(f"   Всего категорий: {len(categories)}")
    print(f"   Первые 3 категории: {categories[:3]}")

print("\n" + "="*60)
print("ГОТОВО! МОДЕЛЬ УСПЕШНО ОБУЧЕНА И СОХРАНЕНА")
print("="*60)
print(f"Для запуска Python сервиса: python app_nids.py")
print("="*60)

```

**Запускаем python сервер**
Для развёртывания Flask-приложения был выбран `Gunicorn` с воркерами `gevent`. Такая комбинация обеспечивает асинхронную обработку запросов, что критично для ML-сервисов с высокой нагрузкой. `Gunicorn` с `gevent` позволяет эффективно обрабатывать сотни параллельных соединений с минимальными накладными расходами.
```
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

        if pred_proba > 0.5:
            result_type = "ATTACK"
        else:
            result_type = "NORMAL"
        
        print(f"Запрос обработан. Время инференса: {inference_time:.2f} мс. Результат: {result_type}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Запрос #{REQUEST_COUNT}: {result_type} (вероятность: {pred_proba:.4f}, время: {inference_time:.2f}мс)")

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

```

```
python3 app_nids.py
```

**Нагрузка**
Для нагрузочного тестирования был использован фреймворк **hey** как современная альтернатива wrk. Hey предоставляет подробную статистику по задержкам и позволяет гибко настраивать нагрузку.

Для нагрузки созданы скрипты под `.py` и `.cpp` сервисы: 

`benchmark_py.sh`
```
#!/bin/bash
set -e

echo "=== НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ PYTHON СЕРВИСА ==="

SERVICE_NAME="python"
URL="http://localhost:5000/predict"
PAYLOAD="../load_test/payload_nids.json"
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/${SERVICE_NAME}_benchmark_${TIMESTAMP}.csv"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$PAYLOAD" ]; then
    echo "Ошибка: payload файл не найден!"
    echo "Проверьте путь: $PAYLOAD"
    exit 1
fi

echo "Проверка доступности Python сервиса..."
if ! curl -s --max-time 5 --fail http://localhost:5000/health > /dev/null 2>&1; then
    echo "Python сервис не отвечает на порту 5000!"
    echo "Проверьте, запущен ли сервис командой:"
    echo "python3 app_nids.py"
    echo ""
    echo "Дополнительная диагностика:"
    echo "Проверьте, слушает ли порт 5000:"
    echo "sudo lsof -i :5000"
    echo "или"
    echo "netstat -tuln | grep 5000"
    echo ""
    echo "Попробуйте запросить health endpoint вручную:"
    echo "curl http://localhost:5000/health"
    exit 1
fi

echo "Python сервис доступен!"

echo "timestamp,concurrency,rps,latency_avg_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,error_rate,total_requests,success_requests,cpu_percent,mem_mb" > "$RESULTS_FILE"

get_service_pid() {
    pid=$(lsof -t -i :5000 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    
    pid=$(ps aux | grep 'python3 app_nids.py' | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    
    pid=$(sudo netstat -tulnp 2>/dev/null | grep ':5000' | awk '{print $7}' | cut -d'/' -f1 | head -1 2>/dev/null || echo "")
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    
    echo ""
    return 1
}

PID=$(get_service_pid)
if [ -z "$PID" ]; then
    echo "Не удалось определить PID Python сервиса. Мониторинг ресурсов отключен."
else
    echo "PID Python сервиса: $PID"
fi

monitor_resources() {
    local pid=$1
    local duration=$2
    local output_file=$3
    
    echo "timestamp,cpu_percent,mem_mb" > "$output_file"
    end_time=$((SECONDS + duration))
    
    while [ $SECONDS -lt $end_time ]; do
        if ps -p $pid > /dev/null 2>&1; then
            cpu_raw=$(ps -p $pid -o %cpu= 2>/dev/null || echo "0.0")
            cpu=$(echo "$cpu_raw" | sed 's/[^0-9.]//g' | awk '{printf "%.1f", $1+0}')
            
            mem_kb=$(ps -p $pid -o rss= 2>/dev/null || echo "0")
            mem_kb=$(echo "$mem_kb" | sed 's/[^0-9]//g')
            
            if [ -z "$mem_kb" ] || [ "$mem_kb" -eq 0 ] 2>/dev/null; then
                mem_mb=0
            else
                mem_mb=$((mem_kb / 1024))
            fi
        else
            cpu="0.0"
            mem_mb="0"
        fi
        
        echo "$(date +%s),$cpu,$mem_mb" >> "$output_file"
        sleep 0.5
    done
}

CONCURRENCY_LEVELS=(1 5 10 25 50 100)

echo "Запуск нагрузочного тестирования Python сервиса..."
echo "Результаты будут сохранены в: $RESULTS_FILE"
echo "Тестирование с уровнями нагрузки: ${CONCURRENCY_LEVELS[@]}"

for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
    echo "Тест с $concurrency параллельными соединениями..."
    
    RESOURCE_FILE="/tmp/${SERVICE_NAME}_resources_${concurrency}.csv"
    echo "timestamp,cpu_percent,mem_mb" > "$RESOURCE_FILE"
    
    if [ -n "$PID" ]; then
        monitor_resources "$PID" 70 "$RESOURCE_FILE" &
        MONITOR_PID=$!
        echo "Мониторинг ресурсов запущен (PID: $MONITOR_PID)"
        sleep 2
    fi
    
    echo "Прогрев сервиса (5 секунд)..."
    if ! hey -z 5s -c $concurrency -m POST -D "$PAYLOAD" -T "application/json" "$URL" > /dev/null 2>&1; then
        echo "Предупреждение: ошибка при прогреве, продолжаем тестирование"
    fi
    
    echo "Основной тест (60 секунд)..."
    echo "Команда: hey -z 60s -c $concurrency -m POST -D \"$PAYLOAD\" -T \"application/json\" \"$URL\""
    
    OUTPUT=$(hey -z 60s -c $concurrency -m POST -D "$PAYLOAD" -T "application/json" "$URL" 2>&1)
    
    echo "Результаты hey:"
    echo "$OUTPUT"
    
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null || true
        echo "Мониторинг ресурсов остановлен"
    fi
    
    RPS=$(echo "$OUTPUT" | grep "Requests/sec" | awk '{gsub(/,/, "", $2); print $2}' | head -1 || echo "0.0")
    LATENCY_AVG=$(echo "$OUTPUT" | grep "Average:" | awk '{gsub(/ms/, "", $2); print $2}' | head -1 || echo "0.0")
    LATENCY_P50=$(echo "$OUTPUT" | grep "50% in" | awk '{gsub(/ms/, "", $3); print $3}' | head -1 || echo "0.0")
    LATENCY_P95=$(echo "$OUTPUT" | grep "95% in" | awk '{gsub(/ms/, "", $3); print $3}' | head -1 || echo "0.0")
    LATENCY_P99=$(echo "$OUTPUT" | grep "99% in" | awk '{gsub(/ms/, "", $3); print $3}' | head -1 || echo "0.0")
    TOTAL_REQUESTS=$(echo "$OUTPUT" | grep "Total:" | awk '{gsub(/,/, "", $2); print $2}' | head -1 || echo "0")
    SUCCESS_REQUESTS=$(echo "$OUTPUT" | grep "Success:" | awk '{gsub(/,/, "", $2); print $2}' | head -1 || echo "0")
    
    if [ -z "$TOTAL_REQUESTS" ] || [ -z "$SUCCESS_REQUESTS" ] || [ "$TOTAL_REQUESTS" -eq 0 ]; then
        ERROR_RATE=0.00
    else
        ERROR_RATE=$(echo "scale=2; 100 * (1 - $SUCCESS_REQUESTS / $TOTAL_REQUESTS)" | bc 2>/dev/null || echo "0.00")
    fi
    
    CPU_AVG="0.0"
    MEM_AVG="0"
    if [ -s "$RESOURCE_FILE" ] && [ $(wc -l < "$RESOURCE_FILE") -gt 1 ]; then
        CPU_AVG=$(awk -F, 'NR>1 {sum+=$2; count++} END {if (count>0) printf "%.1f", sum/count; else print "0.0"}' "$RESOURCE_FILE")
        MEM_AVG=$(awk -F, 'NR>1 {sum+=$3; count++} END {if (count>0) printf "%.0f", sum/count; else print "0"}' "$RESOURCE_FILE")
    fi
    
    RPS=$(echo "$RPS" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_AVG=$(echo "$LATENCY_AVG" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_P50=$(echo "$LATENCY_P50" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_P95=$(echo "$LATENCY_P95" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_P99=$(echo "$LATENCY_P99" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    ERROR_RATE=$(echo "$ERROR_RATE" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    CPU_AVG=$(echo "$CPU_AVG" | sed 's/[^0-9.]//g' | awk '{printf "%.1f", $1+0}')
    
    echo "$(date +%s),$concurrency,$RPS,$LATENCY_AVG,$LATENCY_P50,$LATENCY_P95,$LATENCY_P99,$ERROR_RATE,$TOTAL_REQUESTS,$SUCCESS_REQUESTS,$CPU_AVG,$MEM_AVG" >> "$RESULTS_FILE"
    
    echo "$concurrency соединений: RPS=$RPS, avg_lat=${LATENCY_AVG}ms, cpu=${CPU_AVG}%, errors=${ERROR_RATE}%"
    echo "   └── Всего запросов: $TOTAL_REQUESTS, Успешно: $SUCCESS_REQUESTS"
    
    rm -f "$RESOURCE_FILE"
    
    sleep 5
done

echo ""
echo "Нагрузочное тестирование Python сервиса завершено!"
echo "Результаты сохранены в: $RESULTS_FILE"
echo ""
```

`benchmark_cpp.sh`
```
#!/bin/bash
set -e

echo "=== НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ C++ СЕРВИСА ==="

SERVICE_NAME="cpp"
URL="http://localhost:5001/predict"
PAYLOAD="../load_test/payload_nids.json"
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/${SERVICE_NAME}_benchmark_${TIMESTAMP}.csv"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$PAYLOAD" ]; then
    echo "Ошибка: payload файл не найден!"
    echo "Проверьте путь: $PAYLOAD"
    exit 1
fi

echo "C++ сервис доступен!"

echo "timestamp,concurrency,rps,latency_avg_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,error_rate,total_requests,success_requests,cpu_percent,mem_mb" > "$RESULTS_FILE"

get_service_pid() {
    pid=$(lsof -t -i :5001 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    
    pid=$(ps aux | grep 'ml_nids_cpp' | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi
    
    echo ""
    return 1
}

PID=$(get_service_pid)
if [ -z "$PID" ]; then
    echo "Не удалось определить PID C++ сервиса. Мониторинг ресурсов отключен."
else
    echo "PID C++ сервиса: $PID"
fi

monitor_resources() {
    local pid=$1
    local duration=$2
    local output_file=$3
    
    echo "timestamp,cpu_percent,mem_mb" > "$output_file"
    end_time=$((SECONDS + duration))
    
    while [ $SECONDS -lt $end_time ]; do
        if ps -p $pid > /dev/null 2>&1; then
            cpu_raw=$(ps -p $pid -o %cpu= 2>/dev/null || echo "0.0")
            cpu=$(echo "$cpu_raw" | sed 's/[^0-9.]//g' | awk '{printf "%.1f", $1+0}')
            
            mem_kb=$(ps -p $pid -o rss= 2>/dev/null || echo "0")
            mem_kb=$(echo "$mem_kb" | sed 's/[^0-9]//g')
            
            if [ -z "$mem_kb" ] || [ "$mem_kb" -eq 0 ] 2>/dev/null; then
                mem_mb=0
            else
                mem_mb=$((mem_kb / 1024))
            fi
        else
            cpu="0.0"
            mem_mb="0"
        fi
        
        echo "$(date +%s),$cpu,$mem_mb" >> "$output_file"
        sleep 0.5
    done
}

CONCURRENCY_LEVELS=(1 5 10 25 50 100)

echo "Запуск нагрузочного тестирования C++ сервиса..."
echo "Результаты будут сохранены в: $RESULTS_FILE"
echo "Тестирование с уровнями нагрузки: ${CONCURRENCY_LEVELS[@]}"

for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
    echo "Тест с $concurrency параллельными соединениями..."
    
    RESOURCE_FILE="/tmp/${SERVICE_NAME}_resources_${concurrency}.csv"
    echo "timestamp,cpu_percent,mem_mb" > "$RESOURCE_FILE"
    
    if [ -n "$PID" ]; then
        monitor_resources "$PID" 70 "$RESOURCE_FILE" &
        MONITOR_PID=$!
        echo "Мониторинг ресурсов запущен (PID: $MONITOR_PID)"
        sleep 2
    fi
    
    echo "Прогрев сервиса (5 секунд)..."
    if ! hey -z 5s -c $concurrency -m POST -D "$PAYLOAD" -T "application/json" "$URL" > /dev/null 2>&1; then
        echo "Предупреждение: ошибка при прогреве, продолжаем тестирование"
    fi
    
    echo "Основной тест (60 секунд)..."
    echo "Команда: hey -z 60s -c $concurrency -m POST -D \"$PAYLOAD\" -T \"application/json\" \"$URL\""
    
    if ! OUTPUT=$(timeout 65s hey -z 60s -c $concurrency -m POST -D "$PAYLOAD" -T "application/json" "$URL" 2>&1); then
        echo "Тест прерван по таймауту или с ошибкой"
    fi
    
    echo "Результаты hey:"
    echo "$OUTPUT"
    
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null || true
        echo "Мониторинг ресурсов остановлен"
    fi
    
    RPS=$(echo "$OUTPUT" | grep "Requests/sec" | awk '{gsub(/,/, "", $2); print $2}' | head -1 || echo "0.0")
    LATENCY_AVG=$(echo "$OUTPUT" | grep "Average:" | awk '{gsub(/ms/, "", $2); print $2}' | head -1 || echo "0.0")
    LATENCY_P50=$(echo "$OUTPUT" | grep "50% in" | awk '{gsub(/ms/, "", $3); print $3}' | head -1 || echo "0.0")
    LATENCY_P95=$(echo "$OUTPUT" | grep "95% in" | awk '{gsub(/ms/, "", $3); print $3}' | head -1 || echo "0.0")
    LATENCY_P99=$(echo "$OUTPUT" | grep "99% in" | awk '{gsub(/ms/, "", $3); print $3}' | head -1 || echo "0.0")
    TOTAL_REQUESTS=$(echo "$OUTPUT" | grep "Total:" | awk '{gsub(/,/, "", $2); print $2}' | head -1 || echo "0")
    SUCCESS_REQUESTS=$(echo "$OUTPUT" | grep "Success:" | awk '{gsub(/,/, "", $2); print $2}' | head -1 || echo "0")
    
    if [ -z "$TOTAL_REQUESTS" ] || [ -z "$SUCCESS_REQUESTS" ] || [ "$TOTAL_REQUESTS" -eq 0 ]; then
        ERROR_RATE=0.00
    else
        ERROR_RATE=$(echo "scale=2; 100 * (1 - $SUCCESS_REQUESTS / $TOTAL_REQUESTS)" | bc 2>/dev/null || echo "0.00")
    fi
    
    CPU_AVG="0.0"
    MEM_AVG="0"
    if [ -s "$RESOURCE_FILE" ] && [ $(wc -l < "$RESOURCE_FILE") -gt 1 ]; then
        CPU_AVG=$(awk -F, 'NR>1 {sum+=$2; count++} END {if (count>0) printf "%.1f", sum/count; else print "0.0"}' "$RESOURCE_FILE")
        MEM_AVG=$(awk -F, 'NR>1 {sum+=$3; count++} END {if (count>0) printf "%.0f", sum/count; else print "0"}' "$RESOURCE_FILE")
    fi
    
    RPS=$(echo "$RPS" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_AVG=$(echo "$LATENCY_AVG" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_P50=$(echo "$LATENCY_P50" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_P95=$(echo "$LATENCY_P95" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    LATENCY_P99=$(echo "$LATENCY_P99" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    ERROR_RATE=$(echo "$ERROR_RATE" | sed 's/[^0-9.]//g' | awk '{printf "%.2f", $1+0}')
    CPU_AVG=$(echo "$CPU_AVG" | sed 's/[^0-9.]//g' | awk '{printf "%.1f", $1+0}')
    
    echo "$(date +%s),$concurrency,$RPS,$LATENCY_AVG,$LATENCY_P50,$LATENCY_P95,$LATENCY_P99,$ERROR_RATE,$TOTAL_REQUESTS,$SUCCESS_REQUESTS,$CPU_AVG,$MEM_AVG" >> "$RESULTS_FILE"
    
    echo "$concurrency соединений: RPS=$RPS, avg_lat=${LATENCY_AVG}ms, cpu=${CPU_AVG}%, errors=${ERROR_RATE}%"
    echo "   └── Всего запросов: $TOTAL_REQUESTS, Успешно: $SUCCESS_REQUESTS"
    
    rm -f "$RESOURCE_FILE"
    
    sleep 5
done
```

**Визуализация**
Визуализация представлена в виде скрипта `compare.py`
Создавая `html` страничку со всеми метриками и сравнениями

**Cpp**
Использован тот же датасет `NF-UNSW-NB15` и уже обученная модель `rf_nids_csv.onnx`. Для C++ реализации выбран фреймворк **Drogon** — высокопроизводительный асинхронный веб-фреймворк на C++17, обеспечивающий нативную производительность и минимальные накладные расходы.
```
#include <drogon/drogon.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <unistd.h>
#include <limits.h>

using json = nlohmann::json;
namespace fs = std::filesystem;
std::unique_ptr<Ort::Session> g_session;
Ort::Env g_env{ORT_LOGGING_LEVEL_WARNING, "nids_service"};
std::vector<std::string> g_feature_names;

std::atomic<uint64_t> g_request_count{0};
double g_total_inference_time = 0.0;
std::mutex g_stats_mutex;

struct PreprocessingParams {
    std::unordered_map<std::string, double> numerical_medians;
    std::unordered_map<std::string, double> numerical_means;
    std::unordered_map<std::string, double> numerical_scales;
    std::unordered_map<std::string, std::vector<std::string>> categorical_categories;
    std::vector<std::string> numerical_features;
    std::vector<std::string> categorical_features;
};

PreprocessingParams g_preprocessing_params;

std::string get_executable_path() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return count > 0 ? std::string(result, count) : "";
}

void load_metadata(const std::string& metadata_path) {
    std::ifstream metadata_file(metadata_path);
    if (!metadata_file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл метаданных: " + metadata_path);
    }
    json metadata;
    metadata_file >> metadata;
    if (!metadata.contains("feature_names") || !metadata["feature_names"].is_array()) {
        throw std::runtime_error("Некорректный формат файла метаданных: отсутствует feature_names");
    }
    g_feature_names = metadata["feature_names"].get<std::vector<std::string>>();
    spdlog::info("Загружено {} признаков из {}", g_feature_names.size(), metadata_path);
}

void load_preprocessing_params(const std::string& params_path) {
    if (!fs::exists(params_path)) {
        spdlog::warn("Файл параметров предобработки не найден: {}", params_path);
        spdlog::warn("Будет использована упрощенная предобработка");
        return;
    }
    
    std::ifstream params_file(params_path);
    if (!params_file.is_open()) {
        spdlog::warn("Не удалось открыть файл параметров предобработки: {}", params_path);
        return;
    }
    
    try {
        json params;
        params_file >> params;
        
        if (params.contains("numerical_features") && params["numerical_features"].is_array()) {
            g_preprocessing_params.numerical_features = params["numerical_features"].get<std::vector<std::string>>();
            spdlog::info("Числовые признаки: {}", g_preprocessing_params.numerical_features.size());
        }
        
        if (params.contains("numerical_medians") && params["numerical_medians"].is_object()) {
            for (auto& [key, value] : params["numerical_medians"].items()) {
                if (value.is_number()) {
                    g_preprocessing_params.numerical_medians[key] = value.get<double>();
                }
            }
            spdlog::info("Загружено медианных значений: {}", g_preprocessing_params.numerical_medians.size());
        }
        
        if (params.contains("numerical_means") && params["numerical_means"].is_object()) {
            for (auto& [key, value] : params["numerical_means"].items()) {
                if (value.is_number()) {
                    g_preprocessing_params.numerical_means[key] = value.get<double>();
                }
            }
            spdlog::info("Загружено средних значений: {}", g_preprocessing_params.numerical_means.size());
        }
        
        if (params.contains("numerical_scales") && params["numerical_scales"].is_object()) {
            for (auto& [key, value] : params["numerical_scales"].items()) {
                if (value.is_number()) {
                    g_preprocessing_params.numerical_scales[key] = value.get<double>();
                }
            }
            spdlog::info("Загружено масштабных коэффициентов: {}", g_preprocessing_params.numerical_scales.size());
        }
        
        if (params.contains("categorical_features") && params["categorical_features"].is_array()) {
            g_preprocessing_params.categorical_features = params["categorical_features"].get<std::vector<std::string>>();
            spdlog::info("Категориальные признаки: {}", g_preprocessing_params.categorical_features.size());
        }
        
        if (params.contains("categorical_categories") && params["categorical_categories"].is_object()) {
            for (auto& [feature_name, categories] : params["categorical_categories"].items()) {
                if (categories.is_array()) {
                    std::vector<std::string> cat_list;
                    for (auto& cat : categories) {
                        if (cat.is_string()) {
                            cat_list.push_back(cat.get<std::string>());
                        }
                    }
                    if (!cat_list.empty()) {
                        g_preprocessing_params.categorical_categories[feature_name] = cat_list;
                    }
                }
            }
            spdlog::info("Загружено категорий для {} признаков", g_preprocessing_params.categorical_categories.size());
        }
        
        spdlog::info("Параметры предобработки успешно загружены");
    } catch (const std::exception& e) {
        spdlog::error("Ошибка при загрузке параметров предобработки: {}", e.what());
    }
}

float extract_numeric_value(const json& value) {
    if (value.is_number()) {
        return value.get<float>();
    } else if (value.is_string()) {
        std::string str_val = value.get<std::string>();
        str_val.erase(std::remove(str_val.begin(), str_val.end(), ','), str_val.end());
        str_val.erase(std::remove(str_val.begin(), str_val.end(), ' '), str_val.end());
        try {
            return std::stof(str_val);
        } catch (...) {
            return 0.0f;
        }
    } else if (value.is_boolean()) {
        return value.get<bool>() ? 1.0f : 0.0f;
    }
    return 0.0f;
}

std::vector<float> preprocess_input_features(const json& features_json) {
    std::vector<float> processed_features(g_feature_names.size(), 0.0f);
    
    for (const auto& feature_name : g_preprocessing_params.numerical_features) {
        float value = 0.0f;
        
        if (features_json.contains(feature_name)) {
            value = extract_numeric_value(features_json[feature_name]);
        } else if (g_preprocessing_params.numerical_medians.find(feature_name) != 
                   g_preprocessing_params.numerical_medians.end()) {
            value = static_cast<float>(g_preprocessing_params.numerical_medians[feature_name]);
            spdlog::debug("Использовано медианное значение для {}: {}", feature_name, value);
        }
        
        if (g_preprocessing_params.numerical_means.find(feature_name) != 
            g_preprocessing_params.numerical_means.end() &&
            g_preprocessing_params.numerical_scales.find(feature_name) != 
            g_preprocessing_params.numerical_scales.end()) {
            
            double mean = g_preprocessing_params.numerical_means[feature_name];
            double scale = g_preprocessing_params.numerical_scales[feature_name];
            
            if (scale > 1e-8) { 
                value = static_cast<float>((value - mean) / scale);
            }
        }
        
        auto it = std::find(g_feature_names.begin(), g_feature_names.end(), feature_name);
        if (it != g_feature_names.end()) {
            size_t idx = std::distance(g_feature_names.begin(), it);
            processed_features[idx] = value;
        }
    }
    
    for (const auto& feature_name : g_preprocessing_params.categorical_features) {
        std::string value = "Missing";
        
        if (features_json.contains(feature_name)) {
            if (features_json[feature_name].is_string()) {
                value = features_json[feature_name].get<std::string>();
            } else {
                try {
                    value = std::to_string(extract_numeric_value(features_json[feature_name]));
                } catch (...) {
                    value = "Missing";
                }
            }
        }
        
        if (g_preprocessing_params.categorical_categories.find(feature_name) != 
            g_preprocessing_params.categorical_categories.end()) {
            
            const auto& categories = g_preprocessing_params.categorical_categories[feature_name];
            for (const auto& category : categories) {
                std::string full_feature_name = feature_name + "_" + category;
                
                auto it = std::find(g_feature_names.begin(), g_feature_names.end(), full_feature_name);
                if (it != g_feature_names.end()) {
                    size_t idx = std::distance(g_feature_names.begin(), it);
                    processed_features[idx] = (value == category) ? 1.0f : 0.0f;
                }
            }
        }
    }
    
    return processed_features;
}

void predictHandler(const drogon::HttpRequestPtr& req,
                    std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        auto json_body = json::parse(req->body());
        if (!json_body.contains("features") || !json_body["features"].is_object()) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody(R"({"error": "Требуется {'features': {<имена>: <значения>}}"})");
            callback(resp);
            return;
        }

        auto features = json_body["features"];
        
        std::vector<float> input_values;
        if (!g_preprocessing_params.numerical_features.empty() || 
            !g_preprocessing_params.categorical_features.empty()) {
            input_values = preprocess_input_features(features);
        } else {
            input_values.reserve(g_feature_names.size());
            for (const auto& feature_name : g_feature_names) {
                float value = 0.0f;
                if (features.contains(feature_name)) {
                    value = extract_numeric_value(features[feature_name]);
                }
                input_values.push_back(value);
            }
        }

        if (input_values.size() != g_feature_names.size()) {
            spdlog::warn("Несоответствие количества признаков: {} вместо {}", 
                        input_values.size(), g_feature_names.size());
        }

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_values.size())};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());

        const char* input_name = "float_input";
        const char* output_names[] = {"probabilities", "variable", "output_label", "output_probability"};
        
        std::vector<Ort::Value> output_tensors;
        Ort::RunOptions run_options;
        float attack_prob = 0.0f;
        bool success = false;
        
        for (const auto* output_name : output_names) {
            try {
                output_tensors = g_session->Run(
                    run_options,
                    &input_name, &input_tensor, 1,
                    &output_name, 1
                );
                
                if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
                    auto& output_tensor = output_tensors[0];
                    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
                    auto shape = tensor_info.GetShape();
                    
                    auto output_data = output_tensor.GetTensorData<float>();
                    
                    if (shape.size() == 2 && shape[1] == 2) {
                        attack_prob = output_data[1];
                    } else if (shape.size() == 1 || (shape.size() == 2 && shape[1] == 1)) {
                        float value = output_data[0];
                        if (value < 0.0f || value > 1.0f) {
                            attack_prob = 1.0f / (1.0f + std::exp(-value));
                        } else {
                            attack_prob = value;
                        }
                    } else if (tensor_info.GetElementCount() >= 2) {
                        attack_prob = output_data[1];
                    }
                    
                    spdlog::debug("Использовано имя выхода '{}', вероятность атаки: {:.4f}", 
                                 output_name, attack_prob);
                    success = true;
                    break;
                }
            } catch (const Ort::Exception& ex) {
                spdlog::debug("Попытка с '{}' не удалась: {}", output_name, ex.what());
                continue;
            }
        }
        
        if (!success) {
            throw std::runtime_error("Не удалось получить корректный вывод от модели");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double inference_time_ms = duration.count() / 1000.0;

        bool is_attack = attack_prob > 0.5f;
        std::string result_str = is_attack ? "АТАКА" : "НОРМА";
        
        {
            uint64_t request_number = ++g_request_count;
            spdlog::info("[Запрос #{:06d}] {} (вероятность: {:.4f}, время: {:.2f}мс)", 
                        request_number, result_str, attack_prob, inference_time_ms);
        }

        {
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            g_total_inference_time += inference_time_ms;
        }

        json response = {
            {"prediction", attack_prob},
            {"is_attack", is_attack},
            {"inference_time_ms", inference_time_ms},
            {"model_version", "rf_nids_csv_v3_cpp_fixed"},
            {"features_used", static_cast<int>(g_feature_names.size())}
        };

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(response.dump());
        callback(resp);

    } catch (const std::exception& e) {
        spdlog::error("Ошибка инференса: {}", e.what());
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setBody(json{{"error", e.what()}, {"model_version", "rf_nids_csv_v3_cpp_fixed"}}.dump());
        callback(resp);
    }
}

void healthHandler(const drogon::HttpRequestPtr& req,
                  std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
    uint64_t request_count;
    double total_time;
    {
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        request_count = g_request_count.load();
        total_time = g_total_inference_time;
    }
    
    double avg_time = 0.0;
    if (request_count > 0) {
        avg_time = total_time / request_count;
    }
    
    json response = {
        {"status", "ok"},
        {"model", "ONNX NIDS (CSV)"},
        {"requests_processed", request_count},
        {"avg_inference_time_ms", avg_time},
        {"feature_count", static_cast<int>(g_feature_names.size())},
        {"model_version", "rf_nids_csv_v3_cpp_fixed"}
    };
    
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(response.dump());
    callback(resp);
}

int main()
{
    try {
        std::string exe_path = get_executable_path();
        std::string base_dir = fs::path(exe_path).parent_path().string();
        
        std::string metadata_path = base_dir + "/../models/metadata.json";
        std::string model_path = base_dir + "/../models/rf_nids_csv.onnx";
        std::string preprocessing_path = base_dir + "/../../models/preprocessing_params.json";
        
        try {
            if (fs::exists(metadata_path)) {
                metadata_path = fs::canonical(metadata_path).string();
            } else {
                std::vector<std::string> alt_paths = {
                    "../models/metadata.json",
                    "../../models/metadata.json",
                    "models/metadata.json"
                };
                bool found = false;
                for (const auto& path : alt_paths) {
                    if (fs::exists(path)) {
                        metadata_path = fs::canonical(path).string();
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("Файл метаданных не найден");
                }
            }
        } catch (const fs::filesystem_error& e) {
            spdlog::error("Ошибка поиска файла метаданных: {}", e.what());
            throw;
        }
        
        spdlog::info("Путь к метаданным: {}", metadata_path);
        load_metadata(metadata_path);
        
        if (fs::exists(preprocessing_path)) {
            spdlog::info("Путь к параметрам предобработки: {}", preprocessing_path);
            load_preprocessing_params(preprocessing_path);
        } else {
            spdlog::warn("Файл параметров предобработки не найден: {}", preprocessing_path);
        }
        
        try {
            if (fs::exists(model_path)) {
                model_path = fs::canonical(model_path).string();
            } else {
                std::vector<std::string> alt_paths = {
                    "../models/rf_nids_csv.onnx",
                    "../../models/rf_nids_csv.onnx",
                    "models/rf_nids_csv.onnx"
                };
                bool found = false;
                for (const auto& path : alt_paths) {
                    if (fs::exists(path)) {
                        model_path = fs::canonical(path).string();
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("ONNX модель не найдена");
                }
            }
        } catch (const fs::filesystem_error& e) {
            spdlog::error("Ошибка поиска ONNX модели: {}", e.what());
            throw;
        }
        
        spdlog::info("Путь к модели: {}", model_path);
        
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        g_session = std::make_unique<Ort::Session>(g_env, model_path.c_str(), opts);
        spdlog::info("ONNX модель успешно загружена");
        
        drogon::app().registerHandler("/predict", &predictHandler, {drogon::Post});
        drogon::app().registerHandler("/health", &healthHandler, {drogon::Get});

        drogon::app().setThreadNum(std::thread::hardware_concurrency());
        drogon::app().setLogLevel(trantor::Logger::kInfo);
        drogon::app().addListener("127.0.0.1", 5001);
        
        spdlog::info("Сервер запущен на http://127.0.0.1:5001");
        spdlog::info("Доступные эндпоинты: POST /predict, GET /health");
        spdlog::info("Числовые признаки: {}", g_preprocessing_params.numerical_features.size());
        spdlog::info("Категориальные признаки: {}", g_preprocessing_params.categorical_features.size());
        
        drogon::app().run();
    } catch (const std::exception& e) {
        spdlog::critical("Критическая ошибка: {}", e.what());
        return 1;
    }
}
```

**Заключение**
Для задач обнаружения сетевых вторжений с требованиями к производительности более 1000 RPS C++ реализация является технически и экономически оправданным выбором. Python может использоваться только для разработки и прототипирования, но не для production-развертывания в высоконагруженных средах.

