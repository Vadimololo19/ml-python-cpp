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

