#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from tqdm import tqdm

warnings.filterwarnings('ignore')


def _stat_from_raw(raw_data, stat_key):
    stat_map = {
        'mean':          lambda d: np.mean(d),
        'std':           lambda d: np.std(d),
        'median':        lambda d: np.median(d),
        'min':           lambda d: np.min(d),
        'max':           lambda d: np.max(d),
        'percentile_25': lambda d: np.percentile(d, 25),
        'percentile_75': lambda d: np.percentile(d, 75),
        'percentile_90': lambda d: np.percentile(d, 90),
        'percentile_95': lambda d: np.percentile(d, 95),
        'percentile_99': lambda d: np.percentile(d, 99),
        'n_observations': lambda d: len(d),
    }
    fn = stat_map.get(stat_key)
    return fn(raw_data) if fn else None


class BaselineInterpolator:

    def __init__(self, base_dir: str, signature_lengths: List[int]):
        self.base_dir = Path(base_dir)
        self.signature_lengths = sorted(signature_lengths)
        self.data = {}

    def load_all_data(self):
        for sig_length in tqdm(self.signature_lengths, desc='Loading baselines', unit='length'):
            baseline_dir = self.base_dir / f"baseline_{sig_length}"
            if not baseline_dir.exists():
                continue

            self.data[sig_length] = {}

            enrichment_file = baseline_dir / "enrichment_baseline_stats.json"
            if enrichment_file.exists():
                with open(enrichment_file, 'r') as f:
                    self.data[sig_length]['enrichment'] = json.load(f)

            for cluster_file in baseline_dir.glob("cluster_statistics_*th.json"):
                percentile = cluster_file.stem.split('_')[-1].replace('th', '')
                with open(cluster_file, 'r') as f:
                    self.data[sig_length].setdefault('cluster_stats', {})[percentile] = json.load(f)

            for raw_file in baseline_dir.glob("cluster_raw_values_*th.json"):
                percentile = raw_file.stem.split('_')[-1].replace('th', '')
                with open(raw_file, 'r') as f:
                    self.data[sig_length].setdefault('cluster_raw', {})[percentile] = json.load(f)

    def _make_interp_func(self, values: List[float], method: str):
        x = np.array(self.signature_lengths[:len(values)])
        y = np.array(values)

        valid = ~np.isnan(y)
        x, y = x[valid], y[valid]

        if len(x) < 2:
            return None
        if len(x) < 4 and method == 'cubic':
            method = 'linear'

        try:
            return interpolate.interp1d(x, y, kind=method, fill_value='extrapolate')
        except Exception:
            return None

    def _collect_repeat_mean(self, sig_length, percentile, namespace, repeat_key, metric, stat_key):
        try:
            return self.data[sig_length]['cluster_stats'][percentile][namespace][repeat_key][metric][stat_key]
        except KeyError:
            pass
        try:
            raw = self.data[sig_length]['cluster_raw'][percentile][namespace][repeat_key][metric]
            return _stat_from_raw(raw, stat_key)
        except KeyError:
            return None

    def interpolate_cluster_statistics(self, target_length: int, percentile: str, method: str = 'cubic') -> Dict:
        result = {}

        first_length = self.signature_lengths[0]
        if first_length not in self.data:
            return result

        namespaces = self.data[first_length]['cluster_stats'][percentile].keys()

        for namespace in namespaces:
            result[namespace] = {}
            repeat_keys = list(self.data[first_length]['cluster_stats'][percentile][namespace].keys())

            metrics = set(self.data[first_length]['cluster_stats'][percentile][namespace][repeat_keys[0]].keys())
            try:
                raw_ns = self.data[first_length]['cluster_raw'][percentile][namespace][repeat_keys[0]]
                metrics.update(raw_ns.keys())
            except KeyError:
                pass
            metrics = list(metrics)

            interpolated_means = {}

            for metric in metrics:
                interpolated_means[metric] = {}

                try:
                    stat_keys = list(
                        self.data[first_length]['cluster_stats'][percentile][namespace][repeat_keys[0]][metric].keys()
                    )
                except KeyError:
                    stat_keys = [
                        'mean', 'std', 'median', 'min', 'max',
                        'percentile_25', 'percentile_75',
                        'percentile_90', 'percentile_95', 'percentile_99', 'n_observations',
                    ]

                for stat_key in stat_keys:
                    mean_values = []
                    for sig_length in self.signature_lengths:
                        if sig_length not in self.data:
                            mean_values.append(np.nan)
                            continue
                        vals = [
                            v for rk in repeat_keys
                            if (v := self._collect_repeat_mean(
                                sig_length, percentile, namespace, rk, metric, stat_key
                            )) is not None
                        ]
                        mean_values.append(np.mean(vals) if vals else np.nan)

                    fn = self._make_interp_func(mean_values, method)
                    interpolated_means[metric][stat_key] = float(fn(target_length)) if fn else np.nan

            for repeat_key in repeat_keys:
                result[namespace][repeat_key] = interpolated_means

        return result

    def interpolate_enrichment_baseline(self, target_length: int, method: str = 'cubic') -> Dict:
        result = {}

        first_length = self.signature_lengths[0]
        if first_length not in self.data or 'enrichment' not in self.data[first_length]:
            return result

        namespaces = self.data[first_length]['enrichment'].keys()

        for namespace in tqdm(namespaces, desc='Interpolating enrichment', unit='namespace', leave=False):
            result[namespace] = {}

            go_terms = set()
            for sig_length in self.signature_lengths:
                if sig_length in self.data and 'enrichment' in self.data[sig_length]:
                    go_terms.update(self.data[sig_length]['enrichment'][namespace].keys())

            for go_term in go_terms:
                repeat_keys = set()
                for sig_length in self.signature_lengths:
                    try:
                        repeat_keys.update(self.data[sig_length]['enrichment'][namespace][go_term].keys())
                    except KeyError:
                        continue

                stat_keys = set()
                for sig_length in self.signature_lengths:
                    for rk in repeat_keys:
                        try:
                            stat_keys.update(self.data[sig_length]['enrichment'][namespace][go_term][rk].keys())
                        except KeyError:
                            continue
                stat_keys.discard('name')

                interpolated_means = {}
                for stat_key in stat_keys:
                    mean_values = []
                    for sig_length in self.signature_lengths:
                        if sig_length not in self.data or 'enrichment' not in self.data[sig_length]:
                            mean_values.append(np.nan)
                            continue
                        if go_term not in self.data[sig_length]['enrichment'][namespace]:
                            mean_values.append(np.nan)
                            continue
                        vals = []
                        for rk in repeat_keys:
                            try:
                                vals.append(self.data[sig_length]['enrichment'][namespace][go_term][rk][stat_key])
                            except KeyError:
                                continue
                        mean_values.append(np.mean(vals) if vals else np.nan)

                    fn = self._make_interp_func(mean_values, method)
                    interpolated_means[stat_key] = float(fn(target_length)) if fn else np.nan

                go_term_name = None
                for sig_length in self.signature_lengths:
                    for rk in repeat_keys:
                        try:
                            go_term_name = self.data[sig_length]['enrichment'][namespace][go_term][rk]['name']
                            break
                        except KeyError:
                            continue
                    if go_term_name:
                        break

                result[namespace][go_term] = {}
                for rk in repeat_keys:
                    result[namespace][go_term][rk] = dict(interpolated_means)
                    if go_term_name:
                        result[namespace][go_term][rk]['name'] = go_term_name

        return result

    def save_interpolated_data(self, target_length: int, output_dir: str, method: str = 'cubic'):
        output_path = Path(output_dir) / f"baseline_{target_length}"
        output_path.mkdir(parents=True, exist_ok=True)

        enrichment_data = self.interpolate_enrichment_baseline(target_length, method)
        if enrichment_data:
            with open(output_path / "enrichment_baseline_stats.json", 'w') as f:
                json.dump(enrichment_data, f, indent=2)

        percentiles = set()
        for sig_length in self.data:
            if 'cluster_stats' in self.data[sig_length]:
                percentiles.update(self.data[sig_length]['cluster_stats'].keys())

        for percentile in tqdm(sorted(percentiles), desc='Interpolating cluster stats', unit='percentile'):
            cluster_data = self.interpolate_cluster_statistics(target_length, percentile, method)
            if cluster_data:
                with open(output_path / f"cluster_statistics_{percentile}th.json", 'w') as f:
                    json.dump(cluster_data, f, indent=2)

    def visualize_interpolation(self, metrics: List[str], namespace: str = 'BP',
                                percentile: str = '90', output_dir: str = None, method: str = 'cubic'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        interp_range = np.linspace(min(self.signature_lengths), max(self.signature_lengths), 100)

        first_length = self.signature_lengths[0]
        repeat_keys = []
        try:
            repeat_keys = list(self.data[first_length]['cluster_stats'][percentile][namespace].keys())
        except KeyError:
            pass

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break

            ax = axes[idx]

            for repeat_idx, repeat_key in enumerate(repeat_keys):
                x_data, y_data = [], []
                for sig_length in self.signature_lengths:
                    try:
                        val = self.data[sig_length]['cluster_stats'][percentile][namespace][repeat_key][metric]['mean']
                        x_data.append(sig_length)
                        y_data.append(val)
                    except KeyError:
                        try:
                            raw = self.data[sig_length]['cluster_raw'][percentile][namespace][repeat_key][metric]
                            x_data.append(sig_length)
                            y_data.append(np.mean(raw))
                        except KeyError:
                            continue

                if len(x_data) >= 2:
                    ax.scatter(x_data, y_data, s=100, alpha=0.6,
                               label=repeat_key, color=colors[repeat_idx % len(colors)],
                               marker='o', edgecolors='black', linewidths=1.5, zorder=3)

            mean_values_per_length = []
            for sig_length in self.signature_lengths:
                vals = []
                for repeat_key in repeat_keys:
                    try:
                        vals.append(
                            self.data[sig_length]['cluster_stats'][percentile][namespace][repeat_key][metric]['mean']
                        )
                    except KeyError:
                        try:
                            raw = self.data[sig_length]['cluster_raw'][percentile][namespace][repeat_key][metric]
                            vals.append(np.mean(raw))
                        except KeyError:
                            continue
                if vals:
                    mean_values_per_length.append({
                        'length': sig_length,
                        'mean':   np.mean(vals),
                        'std':    np.std(vals),
                    })

            if mean_values_per_length:
                lengths = np.array([d['length'] for d in mean_values_per_length])
                means   = np.array([d['mean']   for d in mean_values_per_length])
                stds    = np.array([d['std']    for d in mean_values_per_length])

                ax.fill_between(lengths, means - stds, means + stds,
                                alpha=0.25, color='gray', label='+/-1 std (confidence)', zorder=1)

                fn = self._make_interp_func(means.tolist(), method)
                if fn is not None:
                    ax.plot(interp_range, fn(interp_range), '-', alpha=0.8,
                            label=f'Interpolation ({method})', color='red', linewidth=3, zorder=2)
                    ax.scatter(lengths, means, s=150, alpha=1.0,
                               label='Mean across repeats', color='red',
                               marker='D', edgecolors='darkred', linewidths=2, zorder=4)

            ax.set_xlabel('Signature Length', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric} (mean)', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()}\n{namespace}, {percentile}th percentile',
                         fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_file = output_path / f"interpolation_{namespace}_{percentile}th_{method}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Interpolate baseline statistics for arbitrary signature lengths.'
    )
    parser.add_argument('--base_dir',          type=str, required=True)
    parser.add_argument('--signature_lengths', type=str, default='50 100 150 200 300')
    parser.add_argument('--target_length',     type=int, required=True)
    parser.add_argument('--output_dir',        type=str, required=True)
    parser.add_argument('--method',            type=str, default='linear',
                        choices=['linear', 'quadratic', 'cubic'])
    parser.add_argument('--visualize',         action='store_true')
    parser.add_argument('--vis_namespace',     type=str, default='BP', choices=['BP', 'MF', 'CC'])
    parser.add_argument('--vis_percentile',    type=str, default='90')
    args = parser.parse_args()

    signature_lengths = [int(x) for x in args.signature_lengths.split()]

    interpolator = BaselineInterpolator(args.base_dir, signature_lengths)
    interpolator.load_all_data()
    interpolator.save_interpolated_data(args.target_length, args.output_dir, args.method)

    if args.visualize:
        metrics = ['extreme_fraction_genes', 'SR_max', 'related_genes_fraction', 'relationship_genes_fraction']
        interpolator.visualize_interpolation(
            metrics=metrics,
            namespace=args.vis_namespace,
            percentile=args.vis_percentile,
            output_dir=args.output_dir,
            method=args.method,
        )


if __name__ == '__main__':
    main()