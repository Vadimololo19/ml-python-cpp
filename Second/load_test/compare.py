#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime
import json
import sys
from pathlib import Path
import matplotlib as mpl

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['figure.dpi'] = 150

def find_latest_results():
    results_dir = Path("results")
    
    py_files = list(results_dir.glob("python_benchmark_*.csv"))
    cpp_files = list(results_dir.glob("cpp_benchmark_*.csv"))
    
    if not py_files:
        print("Не найдены файлы результатов Python сервиса!")
        return None, None
    
    if not cpp_files:
        print("Не найдены файлы результатов C++ сервиса!")
        return None, None
    
    latest_py = max(py_files, key=lambda f: f.stat().st_mtime)
    latest_cpp = max(cpp_files, key=lambda f: f.stat().st_mtime)
    
    print(f"Найдены файлы результатов:")
    print(f"Python: {latest_py.name} (изменен: {datetime.fromtimestamp(latest_py.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"C++   : {latest_cpp.name} (изменен: {datetime.fromtimestamp(latest_cpp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
    
    return str(latest_py), str(latest_cpp)

def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Загружено {len(df)} записей из {Path(file_path).name}")
    except Exception as e:
        print(f"Ошибка загрузки {file_path}: {e}")
        return None
    
    numeric_cols = ['rps', 'latency_avg_ms', 'latency_p50_ms', 'latency_p95_ms', 
                    'latency_p99_ms', 'error_rate', 'cpu_percent', 'mem_mb']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                median_val = df[col].median()
                print(f"В колонке {col} найдены некорректные значения. Заполнение медианой ({median_val:.2f})")
                df[col] = df[col].fillna(median_val)
    
    df = df[df['rps'] > 0]
    df = df[df['latency_avg_ms'] >= 0]
    df = df[df['concurrency'] > 0]
    
    print(f"После очистки: {len(df)} записей")
    return df

def create_individual_plots(df, service_name, output_dir):
    service_color = '#e74c3c' if service_name == 'Python' else '#2ecc71'
    service_marker = 'o' if service_name == 'Python' else 's'
    
    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(data=df, x='concurrency', y='rps', 
                     marker=service_marker, linewidth=3, markersize=12,
                     color=service_color)
    
    plt.title(f'{service_name} сервис: RPS vs Параллельные соединения', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Количество параллельных соединений', fontsize=14)
    plt.ylabel('Запросов в секунду (RPS)', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    
    max_rps = df['rps'].max()
    max_concurrency = df.loc[df['rps'].idxmax(), 'concurrency']
    plt.annotate(f'Макс. RPS: {max_rps:.1f}\nпри {max_concurrency} соединениях',
                xy=(max_concurrency, max_rps),
                xytext=(0, 20), textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    rps_path = f"{output_dir}/{service_name.lower()}_rps_comparison.png"
    plt.savefig(rps_path, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    latency_df = df.melt(id_vars=['concurrency'], 
                        value_vars=['latency_avg_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms'],
                        var_name='metric', value_name='latency')
    
    latency_df['metric'] = latency_df['metric'].replace({
        'latency_avg_ms': 'Средняя',
        'latency_p50_ms': 'p50',
        'latency_p95_ms': 'p95',
        'latency_p99_ms': 'p99'
    })
    
    ax = sns.lineplot(data=latency_df, x='concurrency', y='latency', hue='metric',
                     marker=service_marker, linewidth=2.5, markersize=10,
                     palette='viridis')
    
    plt.title(f'{service_name} сервис: Задержки при разных уровнях нагрузки', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Количество параллельных соединений', fontsize=14)
    plt.ylabel('Задержка (мс)', fontsize=14)
    plt.yscale('log')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(title='Метрика задержки', title_fontsize=13, fontsize=12)
    
    plt.tight_layout()
    latency_path = f"{output_dir}/{service_name.lower()}_latency_comparison.png"
    plt.savefig(latency_path, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = 'tab:blue'
    ax1.set_xlabel('Количество параллельных соединений', fontsize=14)
    ax1.set_ylabel('CPU (%)', color=color, fontsize=14)
    sns.lineplot(data=df, x='concurrency', y='cpu_percent', 
                marker=service_marker, color=color,
                linewidth=3, markersize=12, ax=ax1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Память (MB)', color=color, fontsize=14)
    sns.lineplot(data=df, x='concurrency', y='mem_mb', 
                marker='D', color=color,
                linewidth=3, markersize=10, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'{service_name} сервис: Использование ресурсов', fontsize=18, fontweight='bold', pad=20)
    fig.tight_layout()
    
    resources_path = f"{output_dir}/{service_name.lower()}_resource_usage.png"
    plt.savefig(resources_path, bbox_inches='tight')
    plt.close()
    
    return {
        'rps_path': rps_path,
        'latency_path': latency_path,
        'resources_path': resources_path
    }

def create_comparison_plots(df_py, df_cpp, output_dir):
    df_py['service'] = 'Python'
    df_cpp['service'] = 'C++'
    df_combined = pd.concat([df_py, df_cpp])
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    ax = sns.lineplot(data=df_combined, x='concurrency', y='rps', hue='service',
                     style='service', markers=True, dashes=False,
                     linewidth=3, markersize=12, 
                     palette=['#e74c3c', '#2ecc71'])
    
    plt.title('Сравнение RPS: Python vs C++', fontsize=16, fontweight='bold')
    plt.xlabel('Количество параллельных соединений', fontsize=14)
    plt.ylabel('Запросов в секунду (RPS)', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    
    for service in ['Python', 'C++']:
        service_data = df_combined[df_combined['service'] == service]
        max_rps = service_data['rps'].max()
        max_conc = service_data.loc[service_data['rps'].idxmax(), 'concurrency']
        plt.annotate(f'{max_rps:.0f} RPS',
                    xy=(max_conc, max_rps),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    plt.subplot(2, 2, 2)
    latency_df = df_combined.melt(id_vars=['concurrency', 'service'], 
                                value_vars=['latency_avg_ms', 'latency_p95_ms', 'latency_p99_ms'],
                                var_name='metric', value_name='latency')
    
    latency_df['metric'] = latency_df['metric'].replace({
        'latency_avg_ms': 'Средняя',
        'latency_p95_ms': 'p95',
        'latency_p99_ms': 'p99'
    })
    
    ax = sns.lineplot(data=latency_df, x='concurrency', y='latency', hue='service', style='metric',
                     markers=True, dashes=False, linewidth=2.5, markersize=8,
                     palette=['#e74c3c', '#2ecc71'])
    
    plt.title('Сравнение задержек', fontsize=16, fontweight='bold')
    plt.xlabel('Количество параллельных соединений', fontsize=14)
    plt.ylabel('Задержка (мс)', fontsize=14)
    plt.yscale('log')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(title='Сервис / Метрика', title_fontsize=13)
    
    plt.subplot(2, 2, 3)
    ax = sns.lineplot(data=df_combined, x='concurrency', y='cpu_percent', hue='service',
                     style='service', markers=True, dashes=False,
                     linewidth=3, markersize=12, 
                     palette=['#e74c3c', '#2ecc71'])
    
    plt.title('Сравнение использования CPU', fontsize=16, fontweight='bold')
    plt.xlabel('Количество параллельных соединений', fontsize=14)
    plt.ylabel('CPU (%)', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    df_combined['efficiency'] = df_combined['rps'] / (df_combined['cpu_percent'] + 1)
    
    ax = sns.lineplot(data=df_combined, x='concurrency', y='efficiency', hue='service',
                     style='service', markers=True, dashes=False,
                     linewidth=3, markersize=12, 
                     palette=['#e74c3c', '#2ecc71'])
    
    plt.title('Сравнение эффективности (RPS / 1% CPU)', fontsize=16, fontweight='bold')
    plt.xlabel('Количество параллельных соединений', fontsize=14)
    plt.ylabel('RPS / 1% CPU', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=3.0)
    comparison_plot = f"{output_dir}/comparison_all_metrics.png"
    plt.savefig(comparison_plot, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    speedup_data = []
    for concurrency in sorted(df_combined['concurrency'].unique()):
        py_data = df_py[df_py['concurrency'] == concurrency]
        cpp_data = df_cpp[df_cpp['concurrency'] == concurrency]
        
        if not py_data.empty and not cpp_data.empty:
            py_rps = py_data['rps'].mean()
            cpp_rps = cpp_data['rps'].mean()
            
            if py_rps > 0:
                speedup = cpp_rps / py_rps
                speedup_data.append({
                    'concurrency': concurrency,
                    'speedup': speedup
                })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        
        ax = sns.lineplot(data=speedup_df, x='concurrency', y='speedup', 
                         marker='o', linewidth=3, markersize=12, color='#9b59b6')
        
        plt.title('Коэффициент ускорения C++ относительно Python', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Количество параллельных соединений', fontsize=14)
        plt.ylabel('Коэффициент ускорения (раз)', fontsize=14)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='Без ускорения')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        
        max_speedup = speedup_df['speedup'].max()
        max_concurrency = speedup_df.loc[speedup_df['speedup'].idxmax(), 'concurrency']
        plt.annotate(f'Макс. ускорение: {max_speedup:.1f}x\nпри {max_concurrency} соединениях',
                    xy=(max_concurrency, max_speedup),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        speedup_plot = f"{output_dir}/speedup_comparison.png"
        plt.savefig(speedup_plot, bbox_inches='tight')
        plt.close()
    else:
        speedup_plot = None
    
    return comparison_plot, speedup_plot

def generate_summary(df_py, df_cpp):
    summary_data = []
    
    for service, df in [('Python', df_py), ('C++', df_cpp)]:
        if df.empty:
            continue
            
        max_rps = df['rps'].max()
        best_conc = df.loc[df['rps'].idxmax(), 'concurrency']
        avg_lat = df.loc[df['concurrency'] == best_conc, 'latency_avg_ms'].mean()
        p99_lat = df.loc[df['concurrency'] == best_conc, 'latency_p99_ms'].mean()
        cpu = df.loc[df['concurrency'] == best_conc, 'cpu_percent'].mean()
        mem = df.loc[df['concurrency'] == best_conc, 'mem_mb'].mean()
        error_rate = df.loc[df['concurrency'] == best_conc, 'error_rate'].mean()
        
        summary_data.append({
            'Service': service,
            'Max RPS': f"{max_rps:.1f}",
            'Best Concurrency': f"{best_conc}",
            'Avg Latency (ms)': f"{avg_lat:.2f}",
            'p99 Latency (ms)': f"{p99_lat:.2f}",
            'CPU Usage (%)': f"{cpu:.1f}",
            'Memory (MB)': f"{mem:.0f}",
            'Error Rate (%)': f"{error_rate:.2f}"
        })
    
    return pd.DataFrame(summary_data)

def generate_html_report(summary_df, speedup, plots, output_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Сравнительный отчет: Python vs C++ ML сервисы</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #3498db;
                border-left: 4px solid #3498db;
                padding-left: 10px;
                margin-top: 30px;
                margin-bottom: 20px;
            }}
            .chart-container {{
                text-align: center;
                margin: 30px 0;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .chart-container img {{
                max-width: 100%;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 4px;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .summary-table th, .summary-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .summary-table th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            .summary-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .summary-table tr:hover {{
                background-color: #e3f2fd;
            }}
            .highlight {{
                background-color: #fffacd;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .speedup {{
                font-size: 28px;
                font-weight: bold;
                color: #27ae60;
                text-align: center;
                padding: 25px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 15px;
                margin: 30px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border: 2px solid #2ecc71;
            }}
            .conclusion {{
                background-color: #e3f2fd;
                padding: 25px;
                border-radius: 8px;
                margin: 30px 0;
                border-left: 4px solid #2196f3;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                color: #7f8c8d;
                font-style: italic;
                padding: 20px;
                border-top: 1px solid #eee;
            }}
            .metric-card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-label {{
                color: #7f8c8d;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <h1>Сравнительный отчет: Python vs C++ ML сервисы</h1>
        <p style="text-align: center; color: #7f8c8d; font-size: 14px;">
            Дата генерации: {timestamp} | Датасет: NF-UNSW-NB15 (обнаружение сетевых атак)
        </p>
        
        <div class="speedup">
            C++ сервис быстрее Python в <span class="highlight">{speedup:.1f} раз</span>
        </div>
        
        <h2>Сводные результаты</h2>
        <table class="summary-table">
            <thead>
                <tr>
    """
    
    for col in summary_df.columns:
        html_content += f"<th>{col}</th>"
    html_content += """
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in summary_df.iterrows():
        html_content += "<tr>"
        for i, value in enumerate(row.values):
            if i == 0:  
                color = "#e74c3c" if value == "Python" else "#2ecc71"
                html_content += f'<td style="color: {color}; font-weight: bold;">{value}</td>'
            else:
                html_content += f"<td>{value}</td>"
        html_content += "</tr>"
    
    html_content += """
            </tbody>
        </table>
        
        <h2>Сравнение производительности</h2>
        <div class="chart-container">
            <img src="comparison_all_metrics.png" alt="Сравнение всех метрик">
            <p style="color: #7f8c8d; margin-top: 5px;">Сравнение RPS, задержек, использования CPU и эффективности при разных уровнях нагрузки</p>
        </div>
        
        <h2>Коэффициент ускорения</h2>
        <div class="chart-container">
            <img src="speedup_comparison.png" alt="Коэффициент ускорения">
            <p style="color: #7f8c8d; margin-top: 5px;">Во сколько раз C++ сервис обрабатывает больше запросов при том же уровне нагрузки</p>
        </div>
    </body>
    </html>
    """
    
    html_path = f"{output_dir}/comparison_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

def main():
    print("="*60)
    print("СРАВНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ PYTHON vs C++ СЕРВИСОВ")
    print("="*60)
    
    py_file, cpp_file = find_latest_results()
    if not py_file or not cpp_file:
        sys.exit(1)
    
    print("\n" + "="*60)
    print("ЗАГРУЗКА И ОЧИСТКА ДАННЫХ")
    print("="*60)
    
    df_py = load_and_clean_data(py_file)
    df_cpp = load_and_clean_data(cpp_file)
    
    if df_py is None or df_cpp is None:
        print("Ошибка загрузки данных. Прерывание выполнения.")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/plots_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nСоздана директория для результатов: {output_dir}")
    
    print("\n" + "="*60)
    print("СОЗДАНИЕ ИНДИВИДУАЛЬНЫХ ГРАФИКОВ")
    print("="*60)
    
    py_plots = create_individual_plots(df_py, 'Python', output_dir)
    cpp_plots = create_individual_plots(df_cpp, 'C++', output_dir)
    
    print(f"Графики Python сохранены:")
    for name, path in py_plots.items():
        print(f"   • {name}: {path}")
    
    print(f"Графики C++ сохранены:")
    for name, path in cpp_plots.items():
        print(f"   • {name}: {path}")
    
    print("\n" + "="*60)
    print("СОЗДАНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ")
    print("="*60)
    
    comparison_plot, speedup_plot = create_comparison_plots(df_py, df_cpp, output_dir)
    
    print(f"Сравнительные графики сохранены:")
    print(f"Все метрики: {comparison_plot}")
    if speedup_plot:
        print(f"Коэффициент ускорения: {speedup_plot}")
    
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ СВОДКИ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    summary_df = generate_summary(df_py, df_cpp)
    
    print("\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    py_max_rps = df_py['rps'].max()
    cpp_max_rps = df_cpp['rps'].max()
    speedup = cpp_max_rps / py_max_rps if py_max_rps > 0 else 0
    
    print(f"\nКоэффициент ускорения C++ относительно Python: {speedup:.1f}x")
    print(f"Python максимальный RPS: {py_max_rps:.1f}")
    print(f"C++ максимальный RPS: {cpp_max_rps:.1f}")
    
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ HTML ОТЧЕТА")
    print("="*60)
    
    html_path = generate_html_report(summary_df, speedup, {
        'comparison_plot': comparison_plot,
        'speedup_plot': speedup_plot
    }, output_dir)
    
    print(f"HTML отчет успешно создан: {html_path}")
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("="*60)
    print(f"\nВсе результаты сохранены в: {output_dir}")
    print(f"\nДля просмотра отчета откройте в браузере:")
    print(f"file://{os.path.abspath(html_path)}")
    print(f"\nОсновные файлы:")
    print(f"HTML отчет: {os.path.basename(html_path)}")
    print(f"Сравнение всех метрик: {os.path.basename(comparison_plot)}")
    if speedup_plot:
        print(f"Коэффициент ускорения: {os.path.basename(speedup_plot)}")
    
if __name__ == "__main__":
    main()
