import os, csv

def save_metrics_to_csv(city, algo_name, exec_time, peak_mem_mb, score, parameters_str):
    """Saves the performance metrics of an algorithm run to a CSV file."""
    os.makedirs("results", exist_ok=True)
    csv_file = "results/metrics.csv"

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Dataset", "Algorithm", "Grid Size", "Score", "Time (s)", "Peak Memory (MB)", "Parameters"])

        grid_size = f"{city.H}x{city.W}"

        writer.writerow([
            city.dataset_name,
            algo_name,
            grid_size,
            score,
            f"{exec_time:.4f}",
            f"{peak_mem_mb:.4f}",
            parameters_str
        ])
