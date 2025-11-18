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

