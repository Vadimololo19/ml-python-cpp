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
Если задача — максимальная производительность и эффективность — C++ предпочтителен. Если важна скорость разработки и умеренная нагрузка — можно использовать Python, но с архитектурными улучшениями.