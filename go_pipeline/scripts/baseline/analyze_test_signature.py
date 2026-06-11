#!/usr/bin/env python3

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import baseline


def load_test_signature(signature_file):
    genes = []
    with open(signature_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith('#'):
                genes.append(gene)
    return genes


def load_baseline_data(baseline_dir, signature_length, percentile):
    baseline_dir_path = Path(baseline_dir)
    baseline_path = baseline_dir_path / f"baseline_{signature_length}"

    if not baseline_path.exists():
        baseline_path = baseline_dir_path / "interpolated" / f"baseline_{signature_length}"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline for signature length {signature_length} not found in:\n"
                f"  {baseline_dir_path / f'baseline_{signature_length}'}\n"
                f"  {baseline_dir_path / 'interpolated' / f'baseline_{signature_length}'}"
            )

    enrichment_file = baseline_path / "enrichment_baseline_stats.json"
    if not enrichment_file.exists():
        raise FileNotFoundError(f"Enrichment baseline file not found: {enrichment_file}")

    with open(enrichment_file, 'r') as f:
        enrichment_stats_raw = json.load(f)

    enrichment_stats = {}
    for namespace, terms in enrichment_stats_raw.items():
        enrichment_stats[namespace] = {}
        for go_term, term_data in terms.items():
            if 'repeat_0' in term_data:
                repeats = [v for k, v in term_data.items() if k.startswith('repeat_')]
                aggregated = {}
                for key in repeats[0].keys():
                    if key == 'name':
                        aggregated[key] = repeats[0][key]
                    else:
                        aggregated[key] = np.mean([r[key] for r in repeats])
                enrichment_stats[namespace][go_term] = aggregated
            else:
                enrichment_stats[namespace][go_term] = term_data

    cluster_file = baseline_path / f"cluster_statistics_{percentile}th.json"
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster statistics file not found: {cluster_file}")
    with open(cluster_file, 'r') as f:
        cluster_stats = json.load(f)

    cluster_raw_file = baseline_path / f"cluster_raw_values_{percentile}th.json"
    cluster_raw_values = None
    if cluster_raw_file.exists():
        with open(cluster_raw_file, 'r') as f:
            cluster_raw_values = json.load(f)

    return enrichment_stats, cluster_stats, cluster_raw_values


def analyze_signature_for_namespace(test_signature, namespace, gene_annotations,
                                    go_dag, ic_data, baseline_stats, percentile):
    namespace_map = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component',
    }

    filtered_annotations = baseline.get_filtered_annotations_for_signature(
        test_signature, gene_annotations, namespace, go_dag
    )

    enrichments = baseline.calculate_enrichment_for_signature(
        filtered_annotations, len(test_signature), ic_data, go_dag, namespace_map[namespace]
    )

    augmented_baseline_stats = dict(baseline_stats)
    for term in enrichments:
        if term not in augmented_baseline_stats:
            augmented_baseline_stats[term] = {
                'mean': 1.0, 'std': 0.0, 'median': 1.0, 'min': 1.0, 'max': 1.0,
                'enrichment_75th': 1.0, 'enrichment_90th': 1.0,
                'enrichment_95th': 1.0, 'enrichment_99th': 1.0,
                'name': go_dag[term].name if term in go_dag else 'Unknown',
            }

    extreme_terms = baseline.get_extreme_terms(enrichments, augmented_baseline_stats, percentile)
    n_extreme = len(extreme_terms)
    extreme_fraction = n_extreme / len(test_signature)

    if n_extreme > 0:
        n_grouped = baseline.group_terms_by_ancestry(extreme_terms, go_dag)
        n_relationships = baseline.count_terms_with_relationships(extreme_terms, go_dag)
        sr_max = baseline.calculate_sr_max(extreme_terms, enrichments, augmented_baseline_stats, percentile)
        n_genes_related = baseline.count_genes_with_related_extreme_terms(
            filtered_annotations, extreme_terms, go_dag
        )
        n_genes_with_relationship = baseline.count_genes_with_relationship_between_extreme_terms(
            filtered_annotations, extreme_terms, go_dag
        )
        genes_with_extreme = {
            gene for gene, terms in filtered_annotations.items()
            if any(t in extreme_terms for t in terms)
        }
        n_genes_extreme = len(genes_with_extreme)
        extreme_fraction_genes = n_genes_extreme / len(test_signature)
    else:
        n_grouped = n_relationships = n_genes_related = n_genes_with_relationship = n_genes_extreme = 0
        sr_max = extreme_fraction_genes = 0.0

    sig_len = len(test_signature)
    return {
        'extreme_terms':              n_extreme,
        'extreme_fraction':           extreme_fraction,
        'extreme_genes_count':        n_genes_extreme,
        'extreme_fraction_genes':     extreme_fraction_genes,
        'grouped_terms':              n_grouped,
        'grouped_fraction':           n_grouped / sig_len,
        'relationship_terms':         n_relationships,
        'relationship_fraction':      n_relationships / sig_len,
        'SR_max':                     sr_max,
        'related_genes_count':        n_genes_related,
        'related_genes_fraction':     n_genes_related / sig_len,
        'relationship_genes_count':   n_genes_with_relationship,
        'relationship_genes_fraction': n_genes_with_relationship / sig_len,
    }


def calculate_baseline_quantile_stats(cluster_stats, cluster_raw_values, namespace, quantile, metrics):
    quantile_key = f'percentile_{quantile}'
    results = {}

    for metric in metrics:
        values = []
        if namespace not in cluster_stats:
            continue

        for repeat_key, repeat_data in cluster_stats[namespace].items():
            if not repeat_key.startswith('repeat_'):
                continue
            if metric in repeat_data and quantile_key in repeat_data[metric]:
                values.append(repeat_data[metric][quantile_key])
            elif metric not in repeat_data and cluster_raw_values is not None:
                ns_raw = cluster_raw_values.get(namespace, {})
                if repeat_key in ns_raw and metric in ns_raw[repeat_key]:
                    values.append(np.percentile(ns_raw[repeat_key][metric], quantile))

        if values:
            results[metric] = {'mean': np.mean(values), 'std': np.std(values), 'values': values}
        else:
            results[metric] = {'mean': np.nan, 'std': np.nan, 'values': []}

    return results


def calculate_snr(test_metrics, baseline_quantile_stats, metrics):
    snr_results = {}
    for metric in metrics:
        test_value = test_metrics.get(metric, np.nan)
        baseline_mean = baseline_quantile_stats[metric]['mean']
        baseline_std = baseline_quantile_stats[metric]['std']
        snr = test_value / baseline_mean if (baseline_mean > 0 and not np.isnan(test_value)) else np.nan
        snr_results[metric] = {
            'snr': snr,
            'test_value': test_value,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
        }
    return snr_results


def _snr_color(snr):
    if snr >= 1.5:
        return '#27ae60'
    if snr >= 1.0:
        return '#90ee90'
    ratio = max(0.0, snr)
    r = int(196 + (255 - 196) * ratio)
    g = int(57  + (165 - 57)  * ratio)
    b = int(43  + (0   - 43)  * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'


def _propagated_snr_error(test_val, baseline_mean, baseline_std):
    return (test_val / baseline_mean ** 2) * baseline_std if baseline_mean > 0 else 0.0


def plot_snr_analysis_combined(all_snr_results, all_metrics, current_namespace, output_file, quantile):
    output_path = Path(output_file)
    output_file_with_ns = output_path.parent / f"{current_namespace}_{output_path.name}"

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })

    snr_results = all_snr_results[current_namespace]
    metrics = list(all_metrics[current_namespace])

    if current_namespace == 'MF':
        metrics = [m for m in metrics if m != 'relationship_genes_fraction']

    metrics = [
        m for m in metrics
        if m in snr_results and snr_results[m].get('baseline_mean', 0) > 0
    ]

    if not metrics:
        return

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Panel A: SNR bar plot
    ax1 = fig.add_subplot(gs[0, 0])
    snr_values = [snr_results[m]['snr'] for m in metrics]
    colors = [_snr_color(s) for s in snr_values]
    x_pos = np.arange(len(metrics))

    bars = ax1.bar(x_pos, snr_values, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=2, width=0.7)

    baseline_errors = [
        _propagated_snr_error(snr_results[m]['test_value'],
                               snr_results[m]['baseline_mean'],
                               snr_results[m]['baseline_std'])
        for m in metrics
    ]
    ax1.errorbar(x_pos, snr_values, yerr=baseline_errors, fmt='none',
                 ecolor='#34495e', capsize=6, linewidth=2.5, capthick=2.5,
                 elinewidth=2.5, alpha=0.8, zorder=10)

    ax1.axhline(y=1.0, color='#90ee90', linestyle='--', linewidth=3, alpha=0.7,
                label='Moderate Threshold (SNR=1.0)', zorder=0)
    ax1.axhline(y=1.5, color='#27ae60', linestyle=':', linewidth=3, alpha=0.7,
                label='Strong Threshold (SNR=1.5)', zorder=0)

    metric_labels = [m.replace('_fraction', '').replace('_', ' ').title() for m in metrics]
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metric_labels, fontsize=11, rotation=0)
    ax1.set_xlabel('Cluster Metrics', fontsize=15, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Signal-to-Noise Ratio (SNR)', fontsize=15, fontweight='bold', labelpad=10)
    ax1.set_title(f'A   SNR Analysis - {current_namespace} Ontology',
                  fontsize=17, fontweight='bold', pad=15, loc='left')
    ax1.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='black',
               fancybox=False, shadow=False)
    ax1.grid(True, alpha=0.25, axis='y', linestyle='-', linewidth=0.8, color='#bdc3c7')
    ax1.set_axisbelow(True)

    valid_snr = [v for v in snr_values if not np.isnan(v) and not np.isinf(v)]
    ax1.set_ylim([0, max(valid_snr) * 1.25 if valid_snr else 3.0])

    for bar, snr, err in zip(bars, snr_values, baseline_errors):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + err + 0.05,
                 f'{snr:.2f}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='#2c3e50')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=11)

    # Panel B: Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    heatmap_data = np.array([
        [snr_results[m]['test_value'], snr_results[m]['baseline_mean'], snr_results[m]['snr']]
        for m in metrics
    ])

    def _metric_label(m):
        if 'relationship_genes' in m: return 'Relationship\nGenes'
        if 'related_genes' in m:      return 'Related\nGenes'
        if 'extreme' in m:            return 'Extreme'
        if 'SR_max' in m.lower() or 'sr_max' in m.lower(): return 'SR Max'
        return m.replace('_fraction', '').replace('_', ' ').title()

    sns.heatmap(
        heatmap_data, annot=True, fmt='.3f',
        cmap='RdYlGn', center=1.0, vmin=0, vmax=max(3.0, heatmap_data.max()),
        xticklabels=['Test\nSignature', f'Baseline\n(Q{quantile}%)', 'SNR'],
        yticklabels=[_metric_label(m) for m in metrics],
        cbar_kws={'label': 'Value', 'shrink': 0.85, 'aspect': 20, 'pad': 0.02},
        linewidths=3, linecolor='white', square=False, ax=ax2,
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    ax2.set_title(f'B   Detailed Metrics - {current_namespace} Ontology',
                  fontsize=17, fontweight='bold', pad=15, loc='left')
    ax2.set_ylabel('Cluster Metrics', fontsize=15, fontweight='bold', labelpad=10)
    ax2.tick_params(labelsize=11, width=1.5, length=0)
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, width=1.5)
    cbar.set_label('Value', fontsize=12, fontweight='bold', labelpad=10)

    # Panel C: All ontologies
    ax3 = fig.add_subplot(gs[1, :])
    ontology_scores = {}
    ontology_errors = {}

    for ont in ['BP', 'MF', 'CC']:
        if ont not in all_snr_results:
            continue
        ont_snr = all_snr_results[ont]
        valid = [
            m for m in metrics
            if not (ont == 'MF' and m == 'relationship_genes_fraction')
            and m in ont_snr
            and ont_snr[m].get('baseline_mean', 0) > 0
        ]
        if valid:
            ontology_scores[ont] = np.mean([ont_snr[m]['snr'] for m in valid])
            errors = [_propagated_snr_error(ont_snr[m]['test_value'],
                                             ont_snr[m]['baseline_mean'],
                                             ont_snr[m]['baseline_std']) for m in valid]
            ontology_errors[ont] = np.sqrt(np.sum(np.array(errors) ** 2)) / len(errors)
        else:
            ontology_scores[ont] = 0.0
            ontology_errors[ont] = 0.0

    for idx, ont in enumerate(['BP', 'MF', 'CC']):
        score = ontology_scores.get(ont, 0)
        color = _snr_color(score)
        classification = 'STRONG' if score >= 1.5 else ('MODERATE' if score >= 1.0 else 'NOISE/WEAK')
        y_pos = 2 - idx

        ax3.barh(y_pos, score, height=0.6, color=color, alpha=0.85,
                 edgecolor='black', linewidth=2.5)

        error = ontology_errors.get(ont, 0)
        if error > 0:
            ax3.errorbar(score, y_pos, xerr=error, fmt='none',
                         ecolor='#34495e', capsize=5, linewidth=2.5, capthick=2.5,
                         elinewidth=2.5, alpha=0.8, zorder=10)

        ax3.text(score * 0.5, y_pos, f'{score:.2f}',
                 va='center', ha='center', fontsize=13, fontweight='bold', color='black')
        ax3.text(-0.05, y_pos, ont, va='center', ha='right', fontsize=14, fontweight='bold')
        ax3.text(0.02, y_pos, classification, va='center', ha='left', fontsize=10,
                 style='italic', color=color, alpha=0.8)

    ax3.axvline(x=1.0, color='#90ee90', linestyle='--', linewidth=3, alpha=0.6, zorder=0)
    ax3.axvline(x=1.5, color='#27ae60', linestyle='--', linewidth=3, alpha=0.6, zorder=0)
    ax3.text(0.5,  3.2, '<- Noise/Weak', ha='center', fontsize=10, style='italic',
             alpha=0.7, fontweight='bold', color='#c0392b')
    ax3.text(1.25, 3.2, 'Moderate',      ha='center', fontsize=10, style='italic',
             alpha=0.7, fontweight='bold', color='#90ee90')
    ax3.text(2.0,  3.2, 'Strong ->',     ha='center', fontsize=10, style='italic',
             alpha=0.7, fontweight='bold', color='#27ae60')

    max_val = max(2.5, max(ontology_scores.values()) + 0.5) if ontology_scores else 2.5
    ax3.set_xlim(-0.1, max_val)
    ax3.set_ylim(-0.5, 3.5)
    ax3.set_xlabel(f'Signal Quality Score (Mean SNR @ Q{quantile}%)',
                   fontsize=15, fontweight='bold', labelpad=10)
    ax3.set_title('C   Ontology-Specific Signal Quality Assessment',
                  fontsize=17, fontweight='bold', pad=15, loc='left')
    ax3.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.8, color='#bdc3c7')
    ax3.set_axisbelow(True)
    ax3.set_yticks([])

    info_text = (f'Based on {len(metrics)} core metrics:\n'
                 f'{", ".join(metrics[:2])},\n{", ".join(metrics[2:])}')
    ax3.text(0.98, 0.05, info_text, transform=ax3.transAxes, fontsize=9,
             style='italic', alpha=0.7, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='#bdc3c7', alpha=0.8, linewidth=1.5))

    for spine in ['top', 'right', 'left']:
        ax3.spines[spine].set_visible(False)
    ax3.tick_params(labelsize=11, width=1.5, left=False)

    Path(output_file_with_ns).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_with_ns, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def _build_json_output(signature_filename, signature_path, signature_length,
                       baseline_dir, percentile, quantile, metrics,
                       all_snr_results):
    json_output = {
        'metadata': {
            'signature_file': signature_filename,
            'signature_path': str(signature_path),
            'signature_length': signature_length,
            'baseline_dir': baseline_dir,
            'percentile': percentile,
            'quantile': quantile,
            'cluster_metrics': metrics,
        },
        'ontologies': {},
    }

    for namespace in ['BP', 'MF', 'CC']:
        ont_snr = all_snr_results[namespace]
        ont_entry = {'metrics': {}, 'panel_c_score': None, 'panel_c_error': None}

        for metric in metrics:
            if metric not in ont_snr:
                continue
            r = ont_snr[metric]
            ont_entry['metrics'][metric] = {
                'test_value':     float(r['test_value']),
                'baseline_mean':  float(r['baseline_mean']),
                'baseline_std':   float(r['baseline_std']),
                'snr':            float(r['snr']),
                'snr_error':      float(_propagated_snr_error(
                    r['test_value'], r['baseline_mean'], r['baseline_std'])),
            }

        valid = [
            m for m in metrics
            if not (namespace == 'MF' and m == 'relationship_genes_fraction')
            and m in ont_snr
            and ont_snr[m].get('baseline_mean', 0) > 0
        ]
        if valid:
            mean_score = np.mean([ont_snr[m]['snr'] for m in valid])
            errors = [_propagated_snr_error(ont_snr[m]['test_value'],
                                             ont_snr[m]['baseline_mean'],
                                             ont_snr[m]['baseline_std']) for m in valid]
            mean_error = np.sqrt(np.sum(np.array(errors) ** 2)) / len(errors)
            ont_entry['panel_c_score'] = float(mean_score)
            ont_entry['panel_c_error'] = float(mean_error)
            ont_entry['panel_c_n_metrics'] = len(valid)
        else:
            ont_entry['panel_c_score'] = 0.0
            ont_entry['panel_c_error'] = 0.0
            ont_entry['panel_c_n_metrics'] = 0

        json_output['ontologies'][namespace] = ont_entry

    return json_output


def main():
    parser = argparse.ArgumentParser(
        description='Analyze test signatures against baseline and calculate SNR.'
    )
    parser.add_argument('--input_path',       type=str, required=True,
                        help='Directory containing the signature files')
    parser.add_argument('--output_path',      type=str, required=True,
                        help='Directory where results will be saved')
    parser.add_argument('--test_signatures',  type=str, nargs='+', required=True)
    parser.add_argument('--baseline_dir',     type=str, required=True)
    parser.add_argument('--percentile',       type=int, required=True)
    parser.add_argument('--quantile',         type=int, required=True)
    parser.add_argument(
        '--cluster_metrics',
        type=str,
        default='extreme_fraction_genes SR_max related_genes_fraction relationship_genes_fraction')
    parser.add_argument('--gene_list',   type=str, default='data/all_human_genes/all_genes.txt')
    parser.add_argument('--gaf_file',    type=str, default='data/goa_human.gaf')
    parser.add_argument('--obo_file',    type=str, default='data/go-basic.obo')
    parser.add_argument('--bp_ic_file',  type=str, default='data/GO_IC/bp_ic.json')
    parser.add_argument('--mf_ic_file',  type=str, default='data/GO_IC/mf_ic.json')
    parser.add_argument('--cc_ic_file',  type=str, default='data/GO_IC/cc_ic.json')
    args = parser.parse_args()

    metrics = args.cluster_metrics.split()

    valid_gene_set = baseline.load_gene_list(args.gene_list)
    go_dag = baseline._load_godag_silent(args.obo_file)
    baseline._GLOBAL_GO_DAG = go_dag
    gene_annotations, _ = baseline.load_gene_annotations(args.gaf_file, go_dag, valid_gene_set)
    ic_data = {
        'BP': baseline.load_ic_data(args.bp_ic_file),
        'MF': baseline.load_ic_data(args.mf_ic_file),
        'CC': baseline.load_ic_data(args.cc_ic_file),
    }

    for signature_filename in tqdm(args.test_signatures, desc='Analyzing signatures', unit='sig'):
        signature_path = Path(args.input_path) / signature_filename
        sig_base_name = signature_filename.replace('_genes.txt', '').replace('.txt', '')
        output_dir = Path(args.output_path) / sig_base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        test_genes = load_test_signature(str(signature_path))
        signature_length = len(test_genes)

        enrichment_stats, cluster_stats, cluster_raw_values = load_baseline_data(
            args.baseline_dir, signature_length, args.percentile
        )

        all_snr_results = {}
        for namespace in ['BP', 'MF', 'CC']:
            test_metrics = analyze_signature_for_namespace(
                test_genes, namespace, gene_annotations, go_dag,
                ic_data[namespace], enrichment_stats[namespace], args.percentile
            )
            baseline_quantile_stats = calculate_baseline_quantile_stats(
                cluster_stats, cluster_raw_values, namespace, args.quantile, metrics
            )
            all_snr_results[namespace] = calculate_snr(test_metrics, baseline_quantile_stats, metrics)

        all_metrics = {ns: metrics for ns in ['BP', 'MF', 'CC']}

        for namespace in ['BP', 'MF', 'CC']:
            output_file = str(output_dir / f"{namespace}_analysis_output.png")
            plot_snr_analysis_combined(all_snr_results, all_metrics, namespace, output_file, args.quantile)

        json_output = _build_json_output(
            signature_filename, signature_path, signature_length,
            args.baseline_dir, args.percentile, args.quantile,
            metrics, all_snr_results
        )
        with open(output_dir / 'baseline_analysis.json', 'w') as f:
            json.dump(json_output, f, indent=2)


if __name__ == '__main__':
    main()