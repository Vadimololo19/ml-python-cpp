#!/bin/bash
CSV_FILE="$1"
DURATION="$2"

echo "timestamp,cpu_user,cpu_system,cpu_idle,mem_total_kb,mem_used_kb,mem_free_kb,mem_available_kb" > "$CSV_FILE"

start_time=$(date +%s)
end_time=$((start_time + DURATION))

while [ $(date +%s) -lt $end_time ]; do
    read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
    sleep 0.2
    read cpu2 user2 nice2 system2 idle2 iowait2 irq2 softirq2 steal2 guest2 < /proc/stat

    PrevIdle=$((idle + iowait))
    Idle=$((idle2 + iowait2))
    PrevNonIdle=$((user + nice + system + irq + softirq + steal))
    NonIdle=$((user2 + nice2 + system2 + irq2 + softirq2 + steal2))
    PrevTotal=$((PrevIdle + PrevNonIdle))
    Total=$((Idle + NonIdle))

    totald=$((Total - PrevTotal))
    idled=$((Idle - PrevIdle))

    CPU_USAGE_USER=$(( (user2 - user) * 100 / totald ))
    CPU_USAGE_SYSTEM=$(( (system2 - system) * 100 / totald ))
    CPU_IDLE=$(( idled * 100 / totald ))

    mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    mem_free=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
    mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    mem_used=$((mem_total - mem_available))

    ts=$(date -Iseconds)
    echo "$ts,$CPU_USAGE_USER,$CPU_USAGE_SYSTEM,$CPU_IDLE,$mem_total,$mem_used,$mem_free,$mem_available" >> "$CSV_FILE"

    sleep 0.3
done
