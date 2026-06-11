#!/usr/bin/env python3

import argparse
import json
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def extract_percentile_from_filename(filepath):
    match = re.search(r'_(\d+)th\.json$', Path(filepath).name)
    if not match:
        raise ValueError(f"Could not extract percentile from filename: {filepath}")
    return int(match.group(1))


def load_cluster_raw_values(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_histogram_with_error_bars(data_across_repeats, n_bins=30):
    all_data = np.concatenate(data_across_repeats)
    bin_edges = np.histogram_bin_edges(all_data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    histograms = []
    for data in data_across_repeats:
        counts, _ = np.histogram(data, bins=bin_edges)
        histograms.append((counts / len(data)) * 100)

    histograms = np.array(histograms)
    return (
        bin_centers,
        np.mean(histograms, axis=0),
        np.std(histograms, axis=0),
        bin_edges[1] - bin_edges[0],
    )


def calculate_summary_stats(data_across_repeats):
    all_data = np.concatenate(data_across_repeats)
    return {
        'mean':   np.mean(all_data),
        'median': np.median(all_data),
        'p90':    np.percentile(all_data, 90),
        'p99':    np.percentile(all_data, 99),
    }


def plot_namespace(raw_values, namespace, metrics, percentile, output_dir):
    n_metrics = len(metrics)
    if n_metrics <= 3:
        n_rows, n_cols = n_metrics, 1
    elif n_metrics == 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows = math.ceil(n_metrics / 2)
        n_cols = 2

    namespace_colors = {'BP': '#4472C4', 'MF': '#70AD47', 'CC': '#FFC000'}
    color = namespace_colors[namespace]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        namespace_data = raw_values[namespace]

        data_across_repeats = [
            namespace_data[key][metric]
            for key in sorted(namespace_data.keys())
            if metric in namespace_data[key]
        ]

        if not data_across_repeats:
            ax.set_visible(False)
            continue

        bin_centers, bin_means, bin_errors, bin_width = calculate_histogram_with_error_bars(
            data_across_repeats
        )

        ax.bar(bin_centers, bin_means, width=bin_width * 0.9,
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(bin_centers, bin_means, yerr=bin_errors,
                    fmt='none', ecolor='black', capsize=2, linewidth=1)

        stats = calculate_summary_stats(data_across_repeats)
        ax.axvline(stats['mean'],   color='red',       linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.3f}")
        ax.axvline(stats['median'], color='darkgreen', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.3f}")
        ax.axvline(stats['p90'],    color='purple',    linestyle=':',  linewidth=2, label=f"90th: {stats['p90']:.3f}")
        ax.axvline(stats['p99'],    color='orange',    linestyle=':',  linewidth=2, label=f"99th: {stats['p99']:.3f}")

        metric_display = metric.replace('_', ' ').title()
        ax.set_xlabel(metric_display, fontsize=12)
        ax.set_ylabel('Percentage %', fontsize=12)
        ax.set_title(f'{namespace} - {metric_display}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = Path(output_dir) / f"{namespace}_cluster_statistics_{percentile}th.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize cluster statistics distributions with error bars across repeats.'
    )
    parser.add_argument('--raw_values_file', type=str, required=True,
                        help='Path to cluster raw values JSON file (e.g., cluster_raw_values_90th.json)')
    parser.add_argument('--plot_statistics', type=str, required=True,
                        help='Space-separated list of metrics to plot')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory for plots')
    args = parser.parse_args()

    percentile = extract_percentile_from_filename(args.raw_values_file)
    metrics = args.plot_statistics.split()
    raw_values = load_cluster_raw_values(args.raw_values_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for namespace in tqdm(['BP', 'MF', 'CC'], desc='Plotting', unit='ontology'):
        plot_namespace(raw_values, namespace, metrics, percentile, output_dir)


if __name__ == '__main__':
    main()