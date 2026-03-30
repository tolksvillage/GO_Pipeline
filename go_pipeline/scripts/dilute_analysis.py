"""
GO Terms Extractor
Extracts GO terms from multiple signatures with explicit selection between
Cumulative and Fixed Mode. Outputs are written to separate directories.
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

def print(*args, **kwargs):
    pass

def extract_base_name(signature_name):
    """Extract base signature name from various naming patterns (cumulative and fixed mode)."""
    name = signature_name

    if name.startswith('diluted_'):
        name = name[8:]

    patterns = [
        r'(.+)_step\d+_cumulative\d+_total\d+',
        r'(.+)_step\d+_fixed\d+_total\d+',
    ]

    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            return match.group(1)

    return name

def resolve_signatures_argument(signatures_arg):
    """
    Resolve --signatures argument.
    """
    if len(signatures_arg) == 1:
        candidate_path = Path(signatures_arg[0])

        if candidate_path.exists() and candidate_path.is_dir():
            print(f"Signature directory detected: {candidate_path}")

            base_signatures = set()

            for item in candidate_path.iterdir():
                if not item.is_dir():
                    continue

                folder_name = item.name

                if folder_name.startswith("diluted_"):
                    base_name = extract_base_name(folder_name)

                    original_json = candidate_path / base_name / "parameter_analysis" / "BP" / "manifold_analysis_BP.json"
                    if original_json.exists():
                        base_signatures.add(base_name)

                else:
                    manifold_json = item / "parameter_analysis" / "BP" / "manifold_analysis_BP.json"
                    if manifold_json.exists():
                        base_signatures.add(folder_name)

            resolved = sorted(base_signatures)

            if not resolved:
                print(f"No valid base signatures found in {candidate_path}.")
            else:
                print(f"Automatically detected base signatures: {', '.join(resolved)}")

            return resolved

    print(f"Manually specified signatures: {', '.join(signatures_arg)}")
    return signatures_arg

def get_mode_from_signature(signature_name):
    if 'totalrandom' in signature_name or 'cumulative' in signature_name:
        return 'cumulative'
    elif 'fixed' in signature_name:
        return 'fixed'
    return None


def load_manifold_jsons(base_path, mode):
    """
    Load all manifold_analysis.json files and group by base name.
    Only include files matching the specified mode (cumulative or fixed).
    Also loads the original (non-diluted) signature for each base name.
    """
    groups = defaultdict(list)
    base_names_found = set()


    for json_file in Path(base_path).rglob("parameter_analysis/BP/manifold_analysis_BP.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            signature_name = data['metadata']['signature_name']
            file_mode = get_mode_from_signature(signature_name)

            if file_mode is not None and file_mode != mode:
                continue

            base_name = extract_base_name(signature_name)
            if file_mode is not None:
                base_names_found.add(base_name)

            groups[base_name].append({
                'signature_name': signature_name,
                'file_path': json_file,
                'data': data
            })

        except Exception as e:
            print(f"Error loading {json_file}: {e}")


    for base_name in base_names_found:
        original_path = Path(base_path) / base_name / "parameter_analysis" / "BP" / "manifold_analysis_BP.json"

        if original_path.exists():
            try:
                with open(original_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                signature_name = data['metadata']['signature_name']

                already_loaded = any(
                    item['signature_name'] == signature_name
                    for item in groups[base_name]
                )

                if not already_loaded:
                    groups[base_name].append({
                        'signature_name': signature_name,
                        'file_path': original_path,
                        'data': data
                    })
                    print(f"Original loaded: {signature_name}")

            except Exception as e:
                print(f"Error loading original {original_path}: {e}")
        else:
            print(f"Original not found: {original_path}")

    return dict(groups)


def extract_unique_go_terms_from_variant(variant_data, representatives_only=False, target_go_ids=None):
    """
    Extract GO terms from variant.
    """
    unique_terms = {}

    for group in variant_data['data']['groups']:
        if representatives_only and target_go_ids is None:
            if 'representative' in group:
                rep_go_id = group['representative']['go_id']
                for term in group['terms']:
                    if term['go_id'] == rep_go_id:
                        unique_terms[rep_go_id] = {
                            'name': term['name'],
                            'frequency_percentage': term['frequency_percentage'],
                            'ic': term['ic'],
                            'genes_direct': term['genes_direct'],
                            'genes_inherited': term['genes_inherited'],
                            'genes_total': term['genes_total'],
                            'alpha_beta_preference': term['alpha_beta_preference'],
                            'group_id': group['group_id']
                        }
                        break

        elif representatives_only and target_go_ids is not None:
            for term in group['terms']:
                if term['go_id'] in target_go_ids:
                    unique_terms[term['go_id']] = {
                        'name': term['name'],
                        'frequency_percentage': term['frequency_percentage'],
                        'ic': term['ic'],
                        'genes_direct': term['genes_direct'],
                        'genes_inherited': term['genes_inherited'],
                        'genes_total': term['genes_total'],
                        'alpha_beta_preference': term['alpha_beta_preference']
                    }

        else:
            for term in group['terms']:
                go_id = term['go_id']
                if go_id not in unique_terms:
                    unique_terms[go_id] = {
                        'name': term['name'],
                        'frequency_percentage': term['frequency_percentage'],
                        'ic': term['ic'],
                        'genes_direct': term['genes_direct'],
                        'genes_inherited': term['genes_inherited'],
                        'genes_total': term['genes_total'],
                        'alpha_beta_preference': term['alpha_beta_preference']
                    }

    return unique_terms


def extract_step_info(signature_name):
    """Extract step and random information from signature name for sorting (cumulative and fixed mode)."""
    step_match = re.search(r'step(\d+)', signature_name)
    step_num = int(step_match.group(1)) if step_match else 0

    random_match = re.search(r'totalrandom(\d+)', signature_name)
    if random_match:
        random_num = int(random_match.group(1))
    else:
        fixed_match = re.search(r'fixed(\d+)', signature_name)
        random_num = int(fixed_match.group(1)) if fixed_match else 0

    return (step_num, random_num)


def process_signature_variants(grouped_data, signature_name, representatives_only=False):
    """Process all variants of a signature and create GO-terms dictionary for each."""
    if signature_name not in grouped_data:
        print(f"Signature '{signature_name}' not found!")
        return None

    variants_dict = {}
    signature_variants = grouped_data[signature_name]

    signature_variants.sort(key=lambda x: extract_step_info(x['signature_name']))

    mode = "REPRESENTATIVES" if representatives_only else "ALL TERMS"
    print(f"Processing signature: {signature_name} ({mode})")
    print(f"Variants found: {len(signature_variants)}")
    print("-" * 60)

    target_go_ids = None
    if representatives_only:
        original_variant = None
        for variant in signature_variants:
            if extract_step_info(variant['signature_name'])[0] == 0:
                original_variant = variant
                break

        if original_variant:
            original_terms = extract_unique_go_terms_from_variant(
                original_variant,
                representatives_only=True,
                target_go_ids=None
            )
            target_go_ids = set(original_terms.keys())
            print(f"Extracted {len(target_go_ids)} representatives from original")

    for i, variant in enumerate(signature_variants, 1):
        variant_name = variant['signature_name']
        step_num = extract_step_info(variant_name)[0]

        if representatives_only and step_num == 0:
            unique_terms = original_terms
        elif representatives_only and step_num > 0:
            unique_terms = extract_unique_go_terms_from_variant(
                variant,
                representatives_only=True,
                target_go_ids=target_go_ids
            )
        else:
            unique_terms = extract_unique_go_terms_from_variant(variant, representatives_only=False)

        variants_dict[variant_name] = unique_terms

        step_info = f"Step {step_num:02d}" if step_num > 0 else "Original"
        print(f"{i:2d}. {step_info} | {len(unique_terms):2d} GO terms | {variant_name}")

    return variants_dict



def create_absolute_frequency_heatmap(frequency_data, original_terms, sorted_variants, signature_name, output_dir,
                                      threshold=0.0, representatives_only=False, cutoff_rank=None):
    """Create a heatmap of absolute frequencies with optional cutoff line."""
    go_ids = list(frequency_data.keys())
    step_labels = []

    for variant_name, _ in sorted_variants:
        step_num = extract_step_info(variant_name)[0]
        step_label = "Original" if step_num == 0 else f"Repetition {step_num}"
        step_labels.append(step_label)

    matrix_data = []
    go_labels = []

    for go_id in go_ids:
        frequencies = frequency_data[go_id]
        matrix_data.append(frequencies)

        term_name = original_terms[go_id]['name']
        short_name = term_name[:40] + "..." if len(term_name) > 40 else term_name
        go_labels.append(short_name)

    df = pd.DataFrame(matrix_data,
                      index=go_labels,
                      columns=step_labels)

    fig_width = max(12, len(step_labels) * 0.8)
    fig_height = max(8, len(go_ids) * 0.35)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if len(go_ids) <= 10:
        cbar_shrink = 0.6
    elif len(go_ids) <= 20:
        cbar_shrink = 0.5
    elif len(go_ids) <= 30:
        cbar_shrink = 0.4
    else:
        cbar_shrink = 0.3

    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 0.5  # Thinner axes
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5

    heatmap = sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100, cbar_kws={
                              'label': 'Robustness Score (%)',
                              'shrink': 1.0,
                              'aspect': 30
                          },
                          linewidths=0.5, linecolor='#E0E0E0', square=False, annot_kws={'size': 9, 'weight': 'normal'}, ax=ax)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9, width=0.5, length=3)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label('Robustness Score (%)', fontsize=10, weight='normal', labelpad=8)

    if cutoff_rank is not None and cutoff_rank > 0 and cutoff_rank < len(go_ids):
        ax.axhline(y=cutoff_rank, color='#B22222', linewidth=3.5, linestyle='--',
                   zorder=10, alpha=0.7)

    ax.set_xlabel('Dilution Repetition', fontsize=11, weight='normal', labelpad=8)
    ax.set_ylabel('GO Terms', fontsize=11, weight='normal', labelpad=8)

    ax.tick_params(axis='both', which='major', labelsize=9, width=0.5, length=3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#666666')

    ax.set_axisbelow(True)

    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename_parts = ["robustness_heatmap", signature_name]

    all_values = df.values.flatten()
    global_mean = np.mean(all_values)
    global_median = np.median(all_values)

    cbar_ax = cbar.ax
    cbar_pos = cbar_ax.get_position()

    stats_text = f'Mean: {global_mean:.1f}%\nMedian: {global_median:.1f}%'
    fig.text(cbar_pos.x0 + (cbar_pos.x1 - cbar_pos.x0) / 2,
             cbar_pos.y0 - 0.08,
             stats_text,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='#666666', linewidth=0.5))

    if representatives_only:
        filename_parts.append("representatives")
    if threshold > 0.0:
        filename_parts.append(f"threshold{threshold}")

    base_filename = '_'.join(filename_parts)
    output_png = Path(output_dir) / f"{base_filename}.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved: {output_png}")

    plt.close()


def calculate_robustness_score(frequencies):
    """
    Calculate robustness score based on initial frequency, retention, and stability.
    """
    if not frequencies or len(frequencies) < 2:
        return 0.0

    initial_freq = frequencies[0]

    if initial_freq == 0:
        return 0.0

    mean_freq = sum(frequencies) / len(frequencies)
    mean_retention = mean_freq / initial_freq

    deviations = [abs(f - initial_freq) for f in frequencies[1:]]
    avg_deviation = sum(deviations) / len(deviations) if deviations else 0

    relative_deviation = avg_deviation / initial_freq if initial_freq > 0 else 0

    stability = max(0, 1 - relative_deviation)

    score = initial_freq * mean_retention * stability

    return score


def find_natural_cutoff(scores, go_ids, go_names):
    """
    Finds the natural cut-off based on the maximum absolute gap.
    """
    if len(scores) < 2:
        print("Not enough terms for cut-off calculation")
        return len(scores)

    print("\n" + "=" * 80)
    print("AUTOMATIC CUT-OFF DETECTION (Method: Maximum absolute gap)")
    print("=" * 80)

    absolute_gaps = []
    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]
        absolute_gaps.append(gap)

    max_gap = max(absolute_gaps)
    max_gap_index = absolute_gaps.index(max_gap)

    cutoff_rank = max_gap_index + 1

    print(f"\n Score distribution:")
    print(f"   Number of terms: {len(scores)}")
    print(f"   Score range: {scores[-1]:.2f} - {scores[0]:.2f}")

    print(f"\n Gap analysis (Top 5 largest jumps):")
    gap_info = [(i, gap, scores[i], scores[i + 1]) for i, gap in enumerate(absolute_gaps)]
    gap_info_sorted = sorted(gap_info, key=lambda x: x[1], reverse=True)

    for rank, (idx, gap, score_before, score_after) in enumerate(gap_info_sorted[:5], 1):
        marker = " MAXIMUM" if idx == max_gap_index else ""
        print(f"   {rank}. Gap {idx + 1}→{idx + 2}: Δ={gap:6.2f} | {score_before:6.2f} → {score_after:6.2f} {marker}")

    print(f"\n RESULT:")
    print(f"   Maximum gap: {max_gap:.2f}")
    print(f"   Position: Between rank {max_gap_index + 1} and rank {max_gap_index + 2}")
    print(f"   {go_ids[max_gap_index]} ({scores[max_gap_index]:.2f})")
    print(f"        GAP = {max_gap:.2f}")
    print(f"   {go_ids[max_gap_index + 1]} ({scores[max_gap_index + 1]:.2f})")
    print(f"\n   CUT-OFF: Keep top {cutoff_rank} terms")
    print(f"   Discard: {len(scores) - cutoff_rank} terms")

    print(f"\n ROBUST TERMS (Top {cutoff_rank}):")
    for i in range(cutoff_rank):
        print(f"   {i + 1:2d}. Score: {scores[i]:6.2f} | {go_ids[i]} | {go_names[i][:60]}")

    print(f"\n UNSTABLE TERMS (Rank {cutoff_rank + 1}-{len(scores)}):")
    for i in range(cutoff_rank, min(cutoff_rank + 3, len(scores))):
        print(f"   {i + 1:2d}. Score: {scores[i]:6.2f} | {go_ids[i]} | {go_names[i][:60]}")
    if len(scores) - cutoff_rank > 3:
        print(f"   ... and {len(scores) - cutoff_rank - 3} more")

    print("=" * 80 + "\n")

    return cutoff_rank


def analyze_all_original_terms(variants_dict, signature_name, base_path, threshold=0.0,
                               representatives_only=False, auto_cutoff=False, mode='cumulative'):
    """Analyze all GO terms from original signature and create heatmap, including IC and gene info."""
    sorted_variants = sorted(variants_dict.items(),
                             key=lambda x: extract_step_info(x[0]))

    original_variant = None
    for variant_name, terms in sorted_variants:
        if extract_step_info(variant_name)[0] == 0:
            original_variant = (variant_name, terms)
            break

    if not original_variant:
        print("No original variant found!")
        return

    original_name, original_terms = original_variant

    if threshold > 0.0:
        filtered_go_ids = []
        for go_id, term_info in original_terms.items():
            robustness_score = term_info.get('frequency_percentage', 0.0)
            if robustness_score >= threshold:
                filtered_go_ids.append(go_id)

        mode_text = "REPRESENTATIVES" if representatives_only else "TERMS"
        print(f"\nFILTERING {mode_text} WITH MINIMUM ROBUSTNESS {threshold}%")
        print(f"Original terms: {len(original_terms)} → Filtered: {len(filtered_go_ids)}")

        if not filtered_go_ids:
            print(f"No terms meet the threshold of {threshold}%!")
            return

        original_go_ids = filtered_go_ids
    else:
        original_go_ids = list(original_terms.keys())

    mode_text = "REPRESENTATIVES" if representatives_only else "GO TERMS"
    print(f"\nANALYZING {len(original_go_ids)} {mode_text}")
    if threshold > 0.0:
        print(f"    (Threshold: ≥{threshold}% Robustness Score)")

    frequency_data = {}

    for go_id in original_go_ids:
        frequencies = []

        for variant_name, terms in sorted_variants:
            if go_id in terms:
                freq = terms[go_id]['frequency_percentage']
            else:
                freq = 0.0
            frequencies.append(freq)

        frequency_data[go_id] = frequencies

    robustness_scores = {}
    robustness_details = {}

    for go_id, frequencies in frequency_data.items():
        score = calculate_robustness_score(frequencies)
        robustness_scores[go_id] = score

        initial_freq = frequencies[0] if frequencies else 0
        final_freq = frequencies[-1] if frequencies else 0
        mean_freq = sum(frequencies) / len(frequencies) if frequencies else 0
        mean_retention = mean_freq / initial_freq if initial_freq > 0 else 0

        if len(frequencies) > 1 and initial_freq > 0:
            deviations = [abs(f - initial_freq) for f in frequencies[1:]]
            avg_deviation = sum(deviations) / len(deviations)
            stability = max(0, 1 - (avg_deviation / initial_freq))
        else:
            avg_deviation = 0
            stability = 1.0

        robustness_details[go_id] = {
            'initial_frequency': initial_freq,
            'final_frequency': final_freq,
            'mean_frequency': mean_freq,
            'mean_retention': mean_retention,
            'avg_deviation': avg_deviation,
            'stability': stability,
            'robustness_score': score
        }

    sorted_go_ids = sorted(robustness_scores.keys(),
                           key=lambda x: robustness_scores[x],
                           reverse=True)

    sorted_frequency_data = {go_id: frequency_data[go_id] for go_id in sorted_go_ids}

    cutoff_rank = None
    if auto_cutoff:
        scores_list = [robustness_scores[go_id] for go_id in sorted_go_ids]
        go_names_list = [original_terms[go_id]['name'] for go_id in sorted_go_ids]
        cutoff_rank = find_natural_cutoff(scores_list, sorted_go_ids, go_names_list)

    json_data = {
        'metadata': {
            'signature_name': signature_name,
            'dilution_mode': mode,
            'analysis_type': 'representatives' if representatives_only else 'all_terms',
            'threshold': threshold,
            'n_terms': len(sorted_go_ids),
            'n_steps': len(frequency_data[sorted_go_ids[0]]) if sorted_go_ids else 0,
            'auto_cutoff_enabled': auto_cutoff,
            'auto_cutoff_rank': cutoff_rank if auto_cutoff else None
        },
        'score_calculation_formula': {
            'formula': 'score = initial_freq * mean_retention * stability',
            'components': {
                'mean_retention': 'mean(all_frequencies) / initial_freq',
                'stability': 'max(0, 1 - (avg_deviation / initial_freq))',
                'avg_deviation': 'mean(|freq_i - initial_freq|) for i > 0'
            }
        },
        'ranking': []
    }

    step_labels = []
    for variant_name, _ in sorted_variants:
        step_num = extract_step_info(variant_name)[0]
        step_label = "Original" if step_num == 0 else f"Step_{step_num:02d}"
        step_labels.append(step_label)

    for rank, go_id in enumerate(sorted_go_ids, 1):
        term_info = original_terms[go_id]
        term_entry = {
            'rank': rank,
            'go_id': go_id,
            'go_name': term_info['name'],
            'ic': term_info['ic'],
            'genes_direct': term_info['genes_direct'],
            'genes_inherited': term_info['genes_inherited'],
            'genes_total': term_info['genes_total'],
            'is_robust': (cutoff_rank is not None and rank <= cutoff_rank) if auto_cutoff else None,
            'frequencies_per_step': {
                step_labels[i]: frequency_data[go_id][i]
                for i in range(len(step_labels))
            },
            'score_components': robustness_details[go_id]
        }
        json_data['ranking'].append(term_entry)

    output_dir = Path(base_path) / signature_name / "Dilutions" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    json_filename_parts = [signature_name, "ranking"]
    if representatives_only:
        json_filename_parts.append("representatives")
    if threshold > 0.0:
        json_filename_parts.append(f"threshold{threshold}")

    json_output_path = output_dir / f"{'_'.join(json_filename_parts)}.json"

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"JSON saved: {json_output_path}")

    if not auto_cutoff:
        print(f"\nRobustness ranking (all {len(sorted_go_ids)} GO terms):")
        for i, go_id in enumerate(sorted_go_ids, 1):
            score = robustness_scores[go_id]
            name = original_terms[go_id]['name'][:50]
            initial = frequency_data[go_id][0]
            final = frequency_data[go_id][-1]
            print(f"   {i:2d}. Score: {score:6.2f} | {go_id} | {name:50s} | {initial:5.1f}% → {final:5.1f}%")

    create_absolute_frequency_heatmap(
        sorted_frequency_data,
        original_terms,
        sorted_variants,
        signature_name,
        output_dir,
        threshold,
        representatives_only,
        cutoff_rank=cutoff_rank if auto_cutoff else None
    )

    print(f"Analysis completed for {signature_name}")



def process_multiple_signatures(grouped_data, signature_names, base_path, analyze_all=False,
                                threshold=0.0, auto_cutoff=False, mode='cumulative'):
    """Process multiple signatures sequentially and create both versions (all terms + representatives)."""
    results = {}

    for signature_name in tqdm(signature_names, desc=f"Signatures ({mode})", unit="sig"):
        for representatives_only in [False]:
            variants_dict = process_signature_variants(grouped_data, signature_name, representatives_only)

            if variants_dict is None:
                continue

            key = f"{signature_name}_representatives" if representatives_only else signature_name
            results[key] = variants_dict

            if analyze_all:
                analyze_all_original_terms(
                    variants_dict,
                    signature_name,
                    base_path,
                    threshold,
                    representatives_only,
                    auto_cutoff,
                    mode
                )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extracts GO terms from multiple signatures with mode selection",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input_path", required=True,
                        help="Base directory with signature subdirectories")
    parser.add_argument("--signatures", nargs='+', required=True,
                        help="Names of the signatures to analyze (multiple possible)")
    parser.add_argument("--mode", required=True, choices=['cumulative', 'fixed'],
                        help="Dilution mode: 'cumulative' (totalrandom) or 'fixed'")
    parser.add_argument("--analyze_all", action="store_true",
                        help="Analyze all GO terms and create heatmaps")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum robustness score (%%) for original terms (default: 0.0)")
    parser.add_argument("--auto_cutoff", action="store_true",
                        help="Enable automatic cut-off detection (method: maximum gap)")

    args = parser.parse_args()

    resolved_signatures = resolve_signatures_argument(args.signatures)

    print("GO Terms Dilution Analyzer")
    print("=" * 80)
    print(f"Selected mode: {args.mode.upper()}")

    if args.auto_cutoff:
        print("Auto cut-off detection: ENABLED")
        print("   Method: Maximum absolute gap")

    print(f"Loading manifold analyses (only {args.mode} mode)...")
    grouped_data = load_manifold_jsons(args.input_path, args.mode)

    if not grouped_data:
        print(f"No {args.mode.upper()} manifold analyses found!")
        return

    print(f"{len(grouped_data)} base signatures found ({args.mode} mode)")

    print(f"\nAvailable signatures ({args.mode} mode):")
    for base_name in sorted(grouped_data.keys()):
        variant_count = len(grouped_data[base_name])
        print(f"   {base_name} ({variant_count} variants)")

    results = process_multiple_signatures(
        grouped_data,
        resolved_signatures,
        args.input_path,
        analyze_all=args.analyze_all,
        threshold=args.threshold,
        auto_cutoff=args.auto_cutoff,
        mode=args.mode
    )

    if not results:
        print("\nNo signatures were successfully processed!")
        return

    print(f"\nSuccessfully processed:")
    for sig_name in results.keys():
        print(f"   {sig_name}")

    print(f"\nOutputs saved in: {{signature}}/Dilutions/{args.mode}/")


if __name__ == "__main__":
    main()