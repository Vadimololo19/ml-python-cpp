import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_file = "results_20251118_102841.csv"  
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
