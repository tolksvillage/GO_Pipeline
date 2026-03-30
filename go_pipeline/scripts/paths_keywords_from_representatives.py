"""
GO Parameter Analysis

Runs a parameter sweep over alpha/beta combinations to rank GO terms
by an enrichment-weighted IC metric, then extracts terms that are
robust across the full parameter space.
"""

import json
import math
import os
import glob
import itertools
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import gc
import argparse


# Data loading

def load_ic_data(ic_dir="data/GO_IC"):
    """Loads precomputed IC values for all three GO ontologies from JSON files."""
    ic_data = {}
    ontology_files = {
        'BP': 'bp_ic.json',
        'MF': 'mf_ic.json',
        'CC': 'cc_ic.json'
    }
    for ontology, filename in ontology_files.items():
        filepath = os.path.join(ic_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    ic_data[ontology] = json.load(f)
            except Exception:
                ic_data[ontology] = {}
        else:
            ic_data[ontology] = {}
    return ic_data


def load_signature_genes(signature_file):
    """Loads gene symbols line by line from a plain-text signature file."""
    genes = []
    try:
        with open(signature_file, 'r', encoding='utf-8') as f:
            for line in f:
                gene = line.strip()
                if gene:
                    genes.append(gene)
    except Exception:
        return []
    return genes


def get_signature_name_from_directory(directory_name):
    """Strips dilution suffixes from a directory name to recover the original signature name."""
    name = directory_name
    if name.startswith("diluted_"):
        name = name[len("diluted_"):]
    cut_markers = ["_step", "_totalrandom", "_total"]
    cut_positions = [name.find(m) for m in cut_markers if m in name]
    if cut_positions:
        name = name[:min(cut_positions)]
    return name


def find_signature_file(signatures_dir, directory_name):
    """Locates the original signature .txt file that corresponds to a given directory."""
    signature_name = get_signature_name_from_directory(directory_name)
    expected_file = os.path.join(signatures_dir, f"{signature_name}.txt")
    if os.path.exists(expected_file):
        return expected_file
    matches = glob.glob(os.path.join(signatures_dir, f"*{signature_name}*.txt"))
    return matches[0] if matches else None


# Directory and node loading

def load_single_directory_data(base_path, directory_name):
    """Reads reduced_terms_*.json for all ontologies of one directory and returns node lists."""
    bp_data, mf_data, cc_data = {}, {}, {}
    reps_dir = os.path.join(base_path, directory_name, "representatives_analysis")
    if not os.path.exists(reps_dir):
        return bp_data, mf_data, cc_data

    ontology_files = {
        "BP": "reduced_terms_bp.json",
        "MF": "reduced_terms_mf.json",
        "CC": "reduced_terms_cc.json"
    }

    for ontology, filename in ontology_files.items():
        file_path = os.path.join(reps_dir, filename)
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                terms_data = json.load(f)

            nodes = []
            for go_id, info in terms_data.items():
                node_info = {
                    'go_annotation': info.get('go_annotation', 'UNKNOWN_TERM'),
                    'genes_direct_count': info.get('genes_direct_count', 0),
                    'genes_inherited_count': info.get('genes_inherited_count', 0),
                    'genes_total_count': info.get(
                        'genes_total_count',
                        info.get('genes_direct_count', 0) + info.get('genes_inherited_count', 0)
                    ),
                    'ic': info.get('ic_normalized', info.get('ic', 0.0))
                }
                nodes.append({'go_id': go_id, 'info': node_info})

            if ontology == "BP":
                bp_data[directory_name] = nodes
            elif ontology == "MF":
                mf_data[directory_name] = nodes
            elif ontology == "CC":
                cc_data[directory_name] = nodes

        except Exception:
            pass

    return bp_data, mf_data, cc_data


def load_reduced_go_ids(base_path, directory_name, ontology):
    """Returns the set of GO IDs present in the reduced terms file for one ontology."""
    reduced_file = os.path.join(
        base_path, directory_name, "representatives_analysis",
        f"reduced_terms_{ontology.lower()}.json"
    )
    if not os.path.exists(reduced_file):
        return set()
    try:
        with open(reduced_file, "r", encoding="utf-8") as f:
            return set(json.load(f).keys())
    except Exception:
        return set()


def get_all_directories(base_path):
    """Returns a sorted list of all subdirectories that contain at least one reduced_terms_*.json."""
    directories = []
    if not os.path.isdir(base_path):
        return directories
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue
        reps_dir = os.path.join(item_path, "representatives_analysis")
        if not os.path.isdir(reps_dir):
            continue
        has_files = any(
            os.path.exists(os.path.join(reps_dir, f))
            for f in ["reduced_terms_bp.json", "reduced_terms_mf.json", "reduced_terms_cc.json"]
        )
        if has_files:
            directories.append(item)
    return sorted(directories)


# Metric and scoring

def metric_sum_enrichment(info, alpha, beta, gamma=0, is_leaf=False, n_signature=None, ic_raw=None):
    """Computes the enrichment-weighted IC score for a single GO node given alpha/beta parameters."""
    if n_signature is None:
        raise ValueError("n_signature must be provided!")
    if ic_raw is None:
        raise ValueError("ic_raw must be provided!")

    genes_direct_count = info.get('genes_direct_count', 0)
    genes_inherited_count = info.get('genes_inherited_count', 0)
    genes_total_count = genes_direct_count + genes_inherited_count

    ln_direct = math.log(genes_direct_count) if genes_direct_count > 0 else 0
    ln_inherited = math.log(genes_inherited_count) if genes_inherited_count > 0 else 0

    p_term = math.exp(-ic_raw)
    expected_genes = max(1.0, n_signature * p_term)
    enrichment = genes_total_count / expected_genes

    return ic_raw * (alpha * ln_direct + beta * ln_inherited) * enrichment


def calculate_node_metrics_optimized(directory_data, alpha, beta, gamma=0,
                                     metric_function=metric_sum_enrichment,
                                     n_signature=None, ic_data_ontology=None):
    """Scores all nodes in a directory for one alpha/beta pair and returns them sorted by metric."""
    if n_signature is None:
        raise ValueError("n_signature must be provided!")
    if ic_data_ontology is None:
        raise ValueError("ic_data_ontology must be provided!")

    go_id_groups = defaultdict(list)

    for node in directory_data:
        go_id = node['go_id']
        info = node['info']
        ic_raw = ic_data_ontology[go_id]['ic_raw'] if go_id in ic_data_ontology else info.get('ic', 1.0) * 10.0

        metric = metric_function(
            info, alpha, beta, gamma,
            is_leaf=True,
            n_signature=n_signature,
            ic_raw=ic_raw
        )
        go_id_groups[go_id].append((metric, info, True, None))

    unique_nodes = []
    for go_id, node_list in go_id_groups.items():
        node_list.sort(key=lambda x: x[0], reverse=True)
        best_metric, best_info, best_is_leaf, best_path_id = node_list[0]
        other_path_ids = [n[3] for n in node_list[1:]]
        unique_nodes.append((go_id, best_metric, best_info, best_is_leaf, best_path_id, other_path_ids))

    unique_nodes.sort(key=lambda x: x[1], reverse=True)
    return unique_nodes


def create_parameter_analysis_optimized(directory_data, parameter_combinations, max_keywords=10,
                                        n_signature=None, ic_data_ontology=None):
    """Runs the full parameter sweep and returns top-ranked GO terms for each alpha/beta combination."""
    if n_signature is None:
        raise ValueError("n_signature must be provided!")
    if ic_data_ontology is None:
        raise ValueError("ic_data_ontology must be provided!")

    results = {}

    for i, (alpha, beta) in enumerate(parameter_combinations):
        param_key = f"alpha_{alpha}_beta_{beta}"

        unique_nodes = calculate_node_metrics_optimized(
            directory_data, alpha=alpha, beta=beta, gamma=0,
            metric_function=metric_sum_enrichment,
            n_signature=n_signature,
            ic_data_ontology=ic_data_ontology
        )

        top_nodes = unique_nodes[:max_keywords]
        del unique_nodes

        keyword_list = []
        for rank, node_data in enumerate(top_nodes, 1):
            go_id, metric, info, is_leaf, path_id, other_path_ids = node_data
            keyword_data = {
                'rank': rank,
                'go_id': go_id,
                'go_annotation': info.get('go_annotation', 'UNKNOWN_TERM'),
                'metric': metric,
                'genes_direct_count': info.get('genes_direct_count', 0),
                'genes_inherited_count': info.get('genes_inherited_count', 0),
                'genes_total_count': info.get('genes_total_count', 0),
                'ic': info.get('ic', 0),
                'is_leaf': is_leaf,
                'path_id': path_id
            }
            if other_path_ids:
                keyword_data['other_path_ids'] = other_path_ids
            keyword_list.append(keyword_data)

        del top_nodes

        results[param_key] = {
            'parameters': {'alpha': alpha, 'beta': beta},
            'keywords': keyword_list
        }

        if (i + 1) % 20 == 0:
            gc.collect()

    return results


# Per-directory processing

def process_single_directory_optimized(base_path, directory_name, parameter_combinations,
                                       max_keywords=20, output_dir="results/", skip_existing=True,
                                       signatures_dir=None, ic_data=None, pbar=None):
    """Runs the parameter analysis for all ontologies of a single directory and writes results to disk."""
    n_signature = None
    if signatures_dir:
        signature_file = find_signature_file(signatures_dir, directory_name)
        if signature_file:
            n_signature = len(load_signature_genes(signature_file))
        elif pbar:
            pbar.write(f"  No signature found for: {directory_name}")

    bp_data, mf_data, cc_data = load_single_directory_data(base_path, directory_name)

    for ontology_name, ontology_data in [('BP', bp_data), ('MF', mf_data), ('CC', cc_data)]:
        if not ontology_data or directory_name not in ontology_data:
            continue

        dir_output_path = os.path.join(output_dir, directory_name, ontology_name, "keyword_analysis")
        os.makedirs(dir_output_path, exist_ok=True)

        output_path = os.path.join(dir_output_path, f"{ontology_name}_parameter_analysis_sum.json")
        if skip_existing and os.path.exists(output_path):
            continue

        ic_data_ontology = ic_data.get(ontology_name, {}) if ic_data else {}

        if n_signature and ic_data_ontology:
            results = create_parameter_analysis_optimized(
                ontology_data[directory_name],
                parameter_combinations,
                max_keywords=max_keywords,
                n_signature=n_signature,
                ic_data_ontology=ic_data_ontology
            )
        else:
            raise ValueError("Legacy metric not implemented - please provide signatures_dir and ic_data.")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        del results
        gc.collect()

    del bp_data, mf_data, cc_data
    gc.collect()


def process_all_directories_optimized(base_path, parameter_combinations, max_keywords=20,
                                      output_dir="results/", skip_existing=True,
                                      signatures_dir=None, ic_data=None, pbar=None):
    """Iterates over all valid directories under base_path and calls process_single_directory_optimized for each."""
    directories = get_all_directories(base_path)
    if not directories:
        print("No valid directories found!")
        return []

    for directory_name in directories:
        if pbar:
            pbar.set_description(f"Parameter analysis: {directory_name}")
        try:
            process_single_directory_optimized(
                base_path, directory_name, parameter_combinations,
                max_keywords=max_keywords, output_dir=output_dir,
                skip_existing=skip_existing, signatures_dir=signatures_dir,
                ic_data=ic_data, pbar=pbar
            )
        except Exception as e:
            if pbar:
                pbar.write(f"  Error in {directory_name}: {e}")
            import traceback
            traceback.print_exc()
        if pbar:
            pbar.update(1)

    return directories


# Validation

def load_path_data_for_validation(base_directory, target_ontology, target_directory):
    """Loads the collected_paths JSON for one ontology and maps path IDs to per-node gene counts."""
    ontology_files = {
        'BP': 'bp_complete_paths.json',
        'MF': 'mf_complete_paths.json',
        'CC': 'cc_complete_paths.json'
    }
    if target_ontology not in ontology_files:
        return {}

    json_file = os.path.join(base_directory, target_directory, "collected_paths",
                             ontology_files[target_ontology])
    if not os.path.exists(json_file):
        return {}

    with open(json_file, 'r', encoding='utf-8') as f:
        paths_data = json.load(f)

    path_mapping = {}
    for path_info in paths_data:
        path_id = path_info['path_id']
        path_mapping[path_id] = {}
        for go_step in path_info['path']:
            if isinstance(go_step, dict):
                for go_id, go_data in go_step.items():
                    path_mapping[path_id][go_id] = {
                        'genes_direct_count': go_data.get('genes_direct_count', 0),
                        'genes_inherited_count': go_data.get('genes_inherited_count', 0),
                        'genes_total_count': go_data.get('genes_total_count', 0)
                    }
            elif isinstance(go_step, list):
                for leaf_dict in go_step:
                    if isinstance(leaf_dict, dict):
                        for go_id, go_data in leaf_dict.items():
                            path_mapping[path_id][go_id] = {
                                'genes_direct_count': go_data.get('genes_direct_count', 0),
                                'genes_inherited_count': go_data.get('genes_inherited_count', 0),
                                'genes_total_count': go_data.get('genes_total_count', 0)
                            }
    return path_mapping


def load_parameter_analysis(json_file):
    """Reads and returns a parameter analysis JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_robust_terms(data, top_n=10, min_appearances=None, path_mapping=None):
    """Identifies GO terms that appear consistently in the top-n across many parameter combinations."""
    total_combinations = len(data)
    if min_appearances is None:
        min_appearances = max(1, total_combinations // 10)

    term_appearances = defaultdict(list)
    for param_key, param_data in data.items():
        params = param_data['parameters']
        for kw in param_data['keywords'][:top_n]:
            go_id = kw['go_id']
            term_appearances[go_id].append({
                'rank': kw['rank'],
                'metric': kw['metric'],
                'params': params,
                'param_key': param_key,
                'go_annotation': kw['go_annotation'],
                'genes_direct_count': kw.get('genes_direct_count', 0),
                'genes_inherited_count': kw.get('genes_inherited_count', 0),
                'genes_total_count': kw.get('genes_total_count', 0),
                'ic': kw['ic'],
                'path_id': kw.get('path_id'),
                'other_path_ids': kw.get('other_path_ids', [])
            })

    robust_terms = []
    for go_id, appearances in term_appearances.items():
        if len(appearances) < min_appearances:
            continue

        ranks = [a['rank'] for a in appearances]
        metrics = [a['metric'] for a in appearances]

        path_ids_info = {}
        if path_mapping:
            all_path_ids = [appearances[0]['path_id']] + appearances[0]['other_path_ids']
            for path_id in all_path_ids:
                if path_id and path_id in path_mapping and go_id in path_mapping[path_id]:
                    path_ids_info[path_id] = path_mapping[path_id][go_id]

        robust_terms.append({
            'go_id': go_id,
            'go_annotation': appearances[0]['go_annotation'],
            'robustness_score': len(appearances) / total_combinations,
            'appearances': len(appearances),
            'avg_rank': sum(ranks) / len(ranks),
            'avg_metric': sum(metrics) / len(metrics),
            'best_rank': min(ranks),
            'rank_std': pd.Series(ranks).std(),
            'metric_std': pd.Series(metrics).std(),
            'ic': appearances[0]['ic'],
            'path_ids': path_ids_info
        })

    robust_terms.sort(key=lambda x: (-x['robustness_score'], x['avg_rank']))
    return robust_terms


def find_all_parameter_analysis_files(base_directory):
    """Scans the result directory tree and returns an indexed dict of all parameter analysis JSON files."""
    organized_files = {}
    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for ontology in ['BP', 'MF', 'CC']:
            keyword_analysis_path = os.path.join(dir_path, ontology, "keyword_analysis")
            if not os.path.exists(keyword_analysis_path):
                continue
            for file in glob.glob(os.path.join(keyword_analysis_path, "*_parameter_analysis_*.json")):
                filename = os.path.basename(file)
                parts = filename.replace('.json', '').split('_')
                try:
                    param_idx = parts.index('parameter')
                    analysis_idx = parts.index('analysis')
                    if param_idx + 1 == analysis_idx and analysis_idx + 1 < len(parts):
                        file_ontology = parts[0]
                        metric = parts[analysis_idx + 1]
                        if file_ontology == ontology:
                            organized_files[(ontology, dir_name, metric)] = file
                except ValueError:
                    continue
    return organized_files


def process_all_validations(input_directory="results/", output_directory="results/",
                            top_n=10, min_appearances_ratio=0.0, final_count=10, pbar=None):
    """Processes all parameter analysis files and writes robust_terms_validation.json per directory."""
    organized_files = find_all_parameter_analysis_files(input_directory)
    if not organized_files:
        if pbar:
            pbar.write("No parameter analysis files found!")
        return

    directory_data = {}
    for (ontology, directory, metric), filepath in organized_files.items():
        directory_data.setdefault(directory, {}).setdefault(ontology, {})[metric] = filepath

    for directory, ontology_metrics in directory_data.items():
        if pbar:
            pbar.set_description(f"Validation: {directory}")
        combined_results = {}

        for ontology, metrics in ontology_metrics.items():
            combined_results[ontology] = {}
            for metric, filepath in metrics.items():
                try:
                    data = load_parameter_analysis(filepath)
                    total_combinations = len(data)
                    min_appearances = max(1, int(total_combinations * min_appearances_ratio))
                    path_mapping = load_path_data_for_validation(output_directory, ontology, directory)
                    robust_terms = extract_robust_terms(
                        data, top_n=top_n,
                        min_appearances=min_appearances,
                        path_mapping=path_mapping
                    )
                    combined_results[ontology][metric] = [{k: v for k, v in term.items()} for term in robust_terms]
                except Exception as e:
                    if pbar:
                        pbar.write(f"  Error in {directory}/{ontology}: {e}")
                    combined_results[ontology][metric] = []

        output_path = os.path.join(output_directory, directory, "robust_terms_validation.json")
        os.makedirs(os.path.join(output_directory, directory), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)

        if pbar:
            pbar.update(1)


# Workflow

def run_optimized_parameter_analysis_workflow(file_path, output_dir, max_keywords=20, skip_existing=True,
                                              signatures_dir=None, ic_dir="data/GO_IC"):
    """Loads IC data and builds the alpha/beta parameter grid, returns both for use in the full workflow."""
    ic_data = load_ic_data(ic_dir)
    if not any(ic_data.values()):
        print("Error: No IC data loaded!")
        return None, None

    param_values = [round(x * 0.1, 1) for x in range(11)]
    parameter_combinations = [
        (a, b) for a, b in itertools.product(param_values, repeat=2)
        if not (a == 0 and b == 0)
    ]

    if not signatures_dir:
        print("Error: No signatures directory provided!")
        return None, None

    return ic_data, parameter_combinations


def run_optimized_complete_workflow(file_path, output_dir, max_keywords=20, top_n=10,
                                    min_appearances_ratio=0.1, final_count=10, skip_existing=True,
                                    signatures_dir=None, ic_dir="data/GO_IC"):
    """Runs the full two-phase pipeline: parameter analysis followed by robust term validation."""
    ic_data, parameter_combinations = run_optimized_parameter_analysis_workflow(
        file_path, output_dir, max_keywords, skip_existing,
        signatures_dir=signatures_dir, ic_dir=ic_dir
    )
    if ic_data is None:
        return

    directories = get_all_directories(file_path)
    n = len(directories)

    with tqdm(total=n * 2, unit="sig") as pbar:
        process_all_directories_optimized(
            file_path, parameter_combinations,
            max_keywords=max_keywords, output_dir=output_dir,
            skip_existing=skip_existing, signatures_dir=signatures_dir,
            ic_data=ic_data, pbar=pbar
        )
        process_all_validations(
            input_directory=output_dir, output_directory=output_dir,
            top_n=top_n, min_appearances_ratio=min_appearances_ratio,
            final_count=final_count, pbar=pbar
        )


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GO analysis with enrichment metric',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', required=True, help='Input directory containing collected_paths subdirectories')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--signatures', required=True, help='Directory with signature .txt files')
    parser.add_argument('--ic-dir', default='data/GO_IC', help='Directory with IC data (default: data/GO_IC)')
    parser.add_argument('--max-keywords', type=int, default=10, help='Max keywords per parameter combination (default: 10)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already processed files')

    args = parser.parse_args()

    if not os.path.exists(args.signatures):
        print(f"Error: Signatures directory not found: {args.signatures}")
        exit(1)

    if not os.path.exists(args.ic_dir):
        print(f"Error: IC directory not found: {args.ic_dir}")
        exit(1)

    run_optimized_complete_workflow(
        file_path=args.input,
        output_dir=args.output,
        max_keywords=args.max_keywords,
        skip_existing=args.skip_existing,
        signatures_dir=args.signatures,
        ic_dir=args.ic_dir
    )