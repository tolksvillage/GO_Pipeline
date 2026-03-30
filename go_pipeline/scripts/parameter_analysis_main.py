import json
from collections import defaultdict
import math
import os
from itertools import combinations
from contextlib import redirect_stdout
from io import StringIO
import argparse
from tqdm import tqdm


def get_all_signature_directories(base_path):
    """Find all directories containing the required JSON files for all ontologies"""
    signature_dirs = []

    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            ontology_files = {}
            for ontology in ['BP', 'MF', 'CC']:
                param_file = os.path.join(subdir_path, ontology, "keyword_analysis",
                                          f"{ontology}_parameter_analysis_sum.json")
                if os.path.exists(param_file):
                    ontology_files[ontology] = param_file

            robust_file = os.path.join(subdir_path, "robust_terms_validation.json")

            if ontology_files and os.path.exists(robust_file):
                signature_dirs.append({
                    'name': subdir,
                    'path': subdir_path,
                    'ontology_files': ontology_files,
                    'robust_file': robust_file
                })

    return signature_dirs


def load_data_from_paths(param_file, robust_file):
    """Load data from specific file paths"""
    with open(param_file, 'r') as f:
        param_data = json.load(f)
    with open(robust_file, 'r') as f:
        robust_data = json.load(f)

    return param_data, robust_data, param_file


def get_top_robust_terms(robust_data, ontology, top_n=3):
    """Extract the top N terms from robust validation data for specified ontology"""
    if isinstance(robust_data, dict) and ontology in robust_data:
        ontology_data = robust_data[ontology]
        if 'sum' in ontology_data and isinstance(ontology_data['sum'], list):
            top_terms = ontology_data['sum'][:top_n]
            return [term['go_id'] for term in top_terms]
        else:
            raise ValueError(f"Expected 'sum' list in {ontology} data")
    else:
        raise ValueError(f"Expected robust_data to be a dict with '{ontology}' key")

def get_annotation_from_robust_data(robust_data, ontology, go_id):
    """Get GO annotation from robust data for specified ontology"""
    if isinstance(robust_data, dict) and ontology in robust_data:
        ontology_data = robust_data[ontology]
        if 'sum' in ontology_data and isinstance(ontology_data['sum'], list):
            for term in ontology_data['sum']:
                if term.get('go_id') == go_id:
                    return term.get('go_annotation', go_id)
    return go_id


def ensure_output_directory(param_file_path, ontology):
    """Create output directory based on input file path and ontology"""
    normalized_path = os.path.normpath(param_file_path)
    parts = normalized_path.split(os.sep)
    try:
        ontology_index = parts.index(ontology)
        base_dir = os.path.join(*parts[:ontology_index])
    except ValueError:
        base_dir = "."

    output_dir = os.path.join(base_dir, "parameter_analysis", ontology)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory ({ontology}): {output_dir}")
    return output_dir


def calculate_term_frequencies(param_data):
    """Calculate how often each term appears across all parameter configurations (Top-10 only)
    and track alpha vs beta preferences"""
    term_frequencies = defaultdict(int)
    term_alpha_beta_counts = defaultdict(lambda: {'alpha_dominant': 0, 'beta_dominant': 0, 'equal': 0})
    total_configs = len(param_data)

    for config_name, config_data in param_data.items():
        top_10_keywords = config_data['keywords'][:10]
        config_terms = set(kw['go_id'] for kw in top_10_keywords)

        alpha = config_data['parameters']['alpha']
        beta = config_data['parameters']['beta']

        for term_id in config_terms:
            term_frequencies[term_id] += 1

            if alpha > beta:
                term_alpha_beta_counts[term_id]['alpha_dominant'] += 1
            elif beta > alpha:
                term_alpha_beta_counts[term_id]['beta_dominant'] += 1
            else:
                term_alpha_beta_counts[term_id]['equal'] += 1

    term_percentages = {term_id: (count / total_configs) * 100
                        for term_id, count in term_frequencies.items()}

    MAX_POSSIBLE_DIFFERENCE = 55

    term_alpha_beta_differences = {}
    for term_id, counts in term_alpha_beta_counts.items():
        alpha_count = counts['alpha_dominant']
        beta_count = counts['beta_dominant']
        equal_count = counts['equal']

        difference = alpha_count - beta_count

        normalized_difference = difference / MAX_POSSIBLE_DIFFERENCE

        term_alpha_beta_differences[term_id] = {
            'absolute': difference,
            'normalized': normalized_difference
        }

    return term_percentages, term_alpha_beta_counts, term_alpha_beta_differences


def filter_terms_by_frequency(all_terms, term_percentages, min_percentage=10):
    """Filter terms that appear in less than min_percentage of configurations"""
    filtered_terms = {term_id for term_id in all_terms
                      if term_percentages.get(term_id, 0) > min_percentage}

    print(f"Original terms: {len(all_terms)}")
    print(f"Filtered terms (>{min_percentage}%): {len(filtered_terms)}")
    print(f"Removed terms: {len(all_terms) - len(filtered_terms)}")

    return filtered_terms


def parse_relevant_go_terms(target_terms, obo_file="data/go-basic.obo"):
    """Parse only GO terms relevant to our manifold + their relationships"""
    relevant_terms = {}
    current_term = {}

    with open(obo_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                if current_term and 'id' in current_term and current_term['id'] in target_terms:
                    relevant_terms[current_term['id']] = current_term
                current_term = {}

            elif line.startswith("id: GO:"):
                term_id = line[4:]
                if term_id in target_terms:
                    current_term['id'] = term_id

            elif line.startswith("name: ") and 'id' in current_term:
                current_term['name'] = line[6:]

            elif line.startswith("is_a: GO:") and 'id' in current_term:
                if 'parents' not in current_term:
                    current_term['parents'] = []
                parent_id = line[6:].split(' ! ')[0]
                current_term['parents'].append(parent_id)

            elif line.startswith("relationship: ") and 'id' in current_term:
                if 'relationships' not in current_term:
                    current_term['relationships'] = []

                parts = line[14:].split(' ')
                if len(parts) >= 2 and parts[1].startswith('GO:'):
                    rel_type = parts[0]
                    target_go = parts[1]
                    current_term['relationships'].append({
                        'type': rel_type,
                        'target': target_go
                    })

    if current_term and 'id' in current_term and current_term['id'] in target_terms:
        relevant_terms[current_term['id']] = current_term

    return relevant_terms


def find_related_groups(manifold_terms, go_terms):
    """Find groups of related terms using simple parent-child relationships"""

    groups = []
    ungrouped = set(manifold_terms)

    while ungrouped:
        seed = next(iter(ungrouped))
        current_group = {seed}
        ungrouped.remove(seed)

        changed = True
        while changed:
            changed = False
            to_add = set()

            for term in current_group:
                if term in go_terms and 'parents' in go_terms[term]:
                    for parent in go_terms[term]['parents']:
                        if parent in ungrouped:
                            to_add.add(parent)
                            changed = True

                for other_term in ungrouped:
                    if other_term in go_terms and 'parents' in go_terms[other_term]:
                        if term in go_terms[other_term]['parents']:
                            to_add.add(other_term)
                            changed = True

            current_group.update(to_add)
            ungrouped -= to_add

        groups.append(current_group)

    return groups


def find_group_regulations(term_groups, go_terms, all_terms):
    """Find part_of, regulates, positively_regulates, and negatively_regulates relationships between groups"""

    term_to_group = {}
    for group_idx, group in enumerate(term_groups):
        for term_id in group:
            term_to_group[term_id] = group_idx

    group_relationships = defaultdict(lambda: defaultdict(set))

    for term_id in all_terms:
        if term_id in go_terms and 'relationships' in go_terms[term_id]:
            source_group = term_to_group.get(term_id)

            for rel in go_terms[term_id]['relationships']:
                target_term = rel['target']
                rel_type = rel['type']

                if target_term in term_to_group:
                    target_group = term_to_group[target_term]

                    if source_group != target_group and rel_type in ['part_of', 'regulates', 'positively_regulates',
                                                                     'negatively_regulates']:
                        group_relationships[source_group][target_group].add(rel_type)

    return group_relationships


def find_group_representative(group, go_terms, all_terms, param_data, term_percentages):
    """Find the term with the highest frequency percentage in a group"""

    best_representative = None
    max_percentage = -1

    for term_id in group:
        frequency = term_percentages.get(term_id, 0)
        if frequency > max_percentage:
            max_percentage = frequency
            best_representative = term_id

    if best_representative is None:
        best_representative = next(iter(group))

    return best_representative


def get_term_connections(term_id, go_terms, term_to_group, current_group):
    """Get connections from a term to terms in other groups"""
    connections = []

    if term_id in go_terms:
        if 'parents' in go_terms[term_id]:
            for parent_id in go_terms[term_id]['parents']:
                if parent_id in term_to_group and term_to_group[parent_id] != current_group:
                    connections.append((parent_id, 'is_a', term_to_group[parent_id]))

        if 'relationships' in go_terms[term_id]:
            for rel in go_terms[term_id]['relationships']:
                target_term = rel['target']
                if target_term in term_to_group and term_to_group[target_term] != current_group:
                    connections.append((target_term, rel['type'], term_to_group[target_term]))

    return connections


def get_term_data_from_config(param_data, go_id):
    """Extract direct, inherited, and ic data for a specific GO term from any configuration"""
    for config_name, config_data in param_data.items():
        for keyword in config_data['keywords']:
            if keyword['go_id'] == go_id:
                return {
                    'direct': keyword['genes_direct_count'],
                    'inherited': keyword['genes_inherited_count'],
                    'ic': keyword['ic']
                }
    return None


def compare_go_term_rankings(param_data, go_id1, go_id2):
    """Compare rankings of two GO terms across all parameter configurations using pure metric comparison"""

    comparison_results = {
        'term1_better': {'count': 0, 'alpha_dominant': 0, 'beta_dominant': 0, 'equal': 0},
        'term2_better': {'count': 0, 'alpha_dominant': 0, 'beta_dominant': 0, 'equal': 0},
        'both_missing': 0,
        'configurations_analyzed': 0,
        'term1_in_top10': 0,
        'term2_in_top10': 0,
        'both_in_top10': 0,
        'ties': 0
    }

    detailed_results = []

    term1_data = get_term_data_from_config(param_data, go_id1)
    term2_data = get_term_data_from_config(param_data, go_id2)

    if term1_data is None and term2_data is None:
        print(f"Error: Both GO terms {go_id1} and {go_id2} not found in data")
        return comparison_results, detailed_results
    elif term1_data is None:
        print(f"Error: GO term {go_id1} not found in data")
        return comparison_results, detailed_results
    elif term2_data is None:
        print(f"Error: GO term {go_id2} not found in data")
        return comparison_results, detailed_results

    for config_name, config_data in param_data.items():
        alpha = config_data['parameters']['alpha']
        beta = config_data['parameters']['beta']

        # Determine alpha/beta relationship
        if alpha > beta:
            alpha_beta_status = 'alpha_dominant'
        elif beta > alpha:
            alpha_beta_status = 'beta_dominant'
        else:
            alpha_beta_status = 'equal'

        ln_direct1 = math.log(term1_data['direct']) if term1_data['direct'] > 0 else 0
        ln_inherited1 = math.log(term1_data['inherited']) if term1_data['inherited'] > 0 else 0
        metric1 = (alpha * ln_direct1 + beta * ln_inherited1) * term1_data['ic']

        ln_direct2 = math.log(term2_data['direct']) if term2_data['direct'] > 0 else 0
        ln_inherited2 = math.log(term2_data['inherited']) if term2_data['inherited'] > 0 else 0
        metric2 = (alpha * ln_direct2 + beta * ln_inherited2) * term2_data['ic']

        term1_rank_top10 = None
        term2_rank_top10 = None
        term1_rank_all = None
        term2_rank_all = None

        top_10_keywords = config_data['keywords'][:10]
        for idx, keyword in enumerate(top_10_keywords):
            if keyword['go_id'] == go_id1:
                term1_rank_top10 = idx + 1
            elif keyword['go_id'] == go_id2:
                term2_rank_top10 = idx + 1

        for idx, keyword in enumerate(config_data['keywords']):
            if keyword['go_id'] == go_id1:
                term1_rank_all = idx + 1
            elif keyword['go_id'] == go_id2:
                term2_rank_all = idx + 1

        comparison_results['configurations_analyzed'] += 1
        if term1_rank_top10 is not None:
            comparison_results['term1_in_top10'] += 1
        if term2_rank_top10 is not None:
            comparison_results['term2_in_top10'] += 1
        if term1_rank_top10 is not None and term2_rank_top10 is not None:
            comparison_results['both_in_top10'] += 1

        if metric1 > metric2:
            comparison_results['term1_better']['count'] += 1
            comparison_results['term1_better'][alpha_beta_status] += 1
        elif metric2 > metric1:
            comparison_results['term2_better']['count'] += 1
            comparison_results['term2_better'][alpha_beta_status] += 1
        else:
            comparison_results['ties'] += 1

        detailed_results.append({
            'config': config_name,
            'alpha': alpha,
            'beta': beta,
            'alpha_beta_status': alpha_beta_status,
            'metric1': metric1,
            'metric2': metric2,
            'metric_difference': metric1 - metric2,
            'term1_rank_top10': term1_rank_top10,
            'term2_rank_top10': term2_rank_top10,
            'term1_rank_all': term1_rank_all,
            'term2_rank_all': term2_rank_all
        })

    return comparison_results, detailed_results


def perform_automated_comparisons(param_data, robust_data, go_terms, output_dir, ontology):
    """Perform automated comparisons for the first 3 terms from robust validation for specified ontology"""

    os.makedirs(output_dir, exist_ok=True)

    top_terms = get_top_robust_terms(robust_data, ontology, top_n=3)

    if len(top_terms) < 2:
        print(f"Error: Need at least 2 terms for comparison in {ontology}")
        return

    print(f"Found {len(top_terms)} {ontology} terms for comparison:")
    for i, go_id in enumerate(top_terms):
        term_name = go_terms.get(go_id, {}).get('name', 'Unknown')
        robust_name = get_annotation_from_robust_data(robust_data, ontology, go_id)
        if robust_name != go_id:
            term_name = robust_name
        print(f"  {i + 1}. {go_id}: {term_name}")

    comparisons = list(combinations(range(len(top_terms)), 2))

    print(f"\nPerforming {len(comparisons)} pairwise comparisons...")

    for i, (idx1, idx2) in enumerate(comparisons):
        go_id1 = top_terms[idx1]
        go_id2 = top_terms[idx2]

        print(f"\n" + "=" * 60)
        print(f"COMPARISON {i + 1}/{len(comparisons)}: {go_id1} vs {go_id2} ({ontology})")
        print("=" * 60)

        comparison_results, detailed_results = compare_go_term_rankings(param_data, go_id1, go_id2)

        print("Comparison completed (no PNG output).")


def analyze_solution_manifold(param_data):
    """Analyze manifold using only TOP-10 keywords per configuration"""
    all_terms = set()
    config_terms = {}

    for config_name, config_data in param_data.items():
        top_10_keywords = config_data['keywords'][:10]
        terms_in_config = set(kw['go_id'] for kw in top_10_keywords)
        all_terms.update(terms_in_config)
        config_terms[config_name] = terms_in_config

    term_percentages, term_alpha_beta_counts, term_alpha_beta_differences = calculate_term_frequencies(param_data)

    go_terms = parse_relevant_go_terms(all_terms)
    term_groups = find_related_groups(all_terms, go_terms)

    group_regulations = find_group_regulations(term_groups, go_terms, all_terms)

    named_groups = []
    for group in term_groups:
        named_group = []
        for term_id in group:
            name = go_terms.get(term_id, {}).get('name', 'Unknown')
            named_group.append({'id': term_id, 'name': name})
        named_groups.append(named_group)

    return {
        'total_unique_terms': len(all_terms),
        'total_configs': len(param_data),
        'config_terms': config_terms,
        'diversity_index': len(all_terms) / (len(param_data) * 10),
        'term_groups': named_groups,
        'num_groups': len(named_groups),
        'group_regulations': group_regulations,
        'go_terms': go_terms,
        'term_percentages': term_percentages,
        'term_alpha_beta_counts': term_alpha_beta_counts,
        'term_alpha_beta_differences': term_alpha_beta_differences
    }


def save_manifold_analysis_json(manifold, param_data, robust_data, output_dir, signature_name, ontology):
    """Save complete manifold analysis as structured JSON"""

    term_to_group = {}
    for group_idx, group in enumerate(manifold['term_groups']):
        for term in group:
            term_to_group[term['id']] = group_idx

    term_info_lookup = {}
    for config_name, config_data in param_data.items():
        for keyword in config_data['keywords'][:10]:
            term_id = keyword['go_id']
            if term_id not in term_info_lookup:
                all_path_ids = []
                if 'path_id' in keyword and keyword['path_id']:
                    all_path_ids.append(keyword['path_id'])
                if 'other_path_ids' in keyword and keyword['other_path_ids']:
                    all_path_ids.extend(keyword['other_path_ids'])

                term_info_lookup[term_id] = {
                    'ic': keyword['ic'],
                    'direct': keyword['genes_direct_count'],
                    'inherited': keyword['genes_inherited_count'],
                    'total': keyword['genes_total_count'],
                    'path_ids': all_path_ids
                }
            else:
                existing_path_ids = set(term_info_lookup[term_id].get('path_ids', []))
                if 'path_id' in keyword and keyword['path_id']:
                    existing_path_ids.add(keyword['path_id'])
                if 'other_path_ids' in keyword and keyword['other_path_ids']:
                    existing_path_ids.update(keyword['other_path_ids'])
                term_info_lookup[term_id]['path_ids'] = sorted(list(existing_path_ids))

    manifold_data = {
        "metadata": {
            "signature_name": signature_name,
            "ontology": ontology,
            "analysis_type": "manifold_analysis"
        },
        "summary_statistics": {
            "unique_go_terms_top10": manifold['total_unique_terms'],
            "parameter_configurations": manifold['total_configs'],
            "diversity_index": manifold['diversity_index'],
            "number_of_groups": manifold['num_groups']
        },
        "groups": []
    }

    groups_with_frequencies = []
    for i, group in enumerate(manifold['term_groups']):
        group_ids = set(term['id'] for term in group)
        representative_id = find_group_representative(group_ids, manifold['go_terms'],
                                                      group_ids, param_data, manifold['term_percentages'])
        rep_percentage = manifold['term_percentages'].get(representative_id, 0)
        representative_name = manifold['go_terms'].get(representative_id, {}).get('name', 'Unknown')
        groups_with_frequencies.append((i, group, rep_percentage, representative_id, representative_name))

    sorted_groups = sorted(groups_with_frequencies, key=lambda x: x[2], reverse=True)

    for display_num, (original_idx, group, rep_percentage, representative_id, representative_name) in enumerate(
            sorted_groups, 1):

        group_data = {
            "group_id": display_num,
            "original_group_index": original_idx,
            "size": len(group),
            "representative": {
                "go_id": representative_id,
                "name": representative_name,
                "frequency_percentage": rep_percentage
            },
            "terms": [],
            "connections_to_other_groups": []
        }

        sorted_terms = sorted(group, key=lambda term: manifold['term_percentages'].get(term['id'], 0), reverse=True)

        for term in sorted_terms:
            term_id = term['id']
            term_name = term['name']
            term_percentage = manifold['term_percentages'].get(term_id, 0)

            alpha_beta_info = manifold['term_alpha_beta_counts'].get(
                term_id,
                {'alpha_dominant': 0, 'beta_dominant': 0, 'equal': 0}
            )

            alpha_beta_diff = manifold['term_alpha_beta_differences'].get(
                term_id,
                {'absolute': 0, 'normalized': 0.0}
            )

            term_data = {
                "go_id": term_id,
                "name": term_name,
                "frequency_percentage": term_percentage,
                "alpha_beta_preference": {
                    "alpha_dominant": alpha_beta_info['alpha_dominant'],
                    "beta_dominant": alpha_beta_info['beta_dominant'],
                    "equal": alpha_beta_info['equal'],
                    "difference_absolute": alpha_beta_diff['absolute'],
                    "difference_normalized": alpha_beta_diff['normalized']
                }
            }

            if term_id in term_info_lookup:
                info = term_info_lookup[term_id]
                term_data.update({
                    "ic": info['ic'],
                    "genes_direct": info['direct'],
                    "genes_inherited": info['inherited'],
                    "genes_total": info['total'],
                    "path_ids": info.get('path_ids', [])
                })
            else:
                term_data.update({
                    "ic": None,
                    "genes_direct": None,
                    "genes_inherited": None,
                    "genes_total": None,
                    "path_ids": []
                })

            group_data["terms"].append(term_data)

        for term in sorted_terms:
            term_id = term['id']
            connections = get_term_connections(term_id, manifold['go_terms'], term_to_group, original_idx)

            for target_term, rel_type, target_group in connections:
                target_name = manifold['go_terms'].get(target_term, {}).get('name', 'Unknown')
                target_display = next((i + 1 for i, (orig_idx, _, _, _, _) in enumerate(sorted_groups)
                                       if orig_idx == target_group), target_group + 1)

                connection_data = {
                    "source_group_id": display_num,
                    "target_group_id": target_display,
                    "relationship_type": rel_type,
                    "source_term": {
                        "go_id": term_id,
                        "name": term['name']
                    },
                    "target_term": {
                        "go_id": target_term,
                        "name": target_name
                    }
                }

                group_data["connections_to_other_groups"].append(connection_data)

        manifold_data["groups"].append(group_data)

    json_filename = f"manifold_analysis_{ontology}.json"
    json_filepath = os.path.join(output_dir, json_filename)

    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(manifold_data, f, indent=2, ensure_ascii=False)

    print(f"Manifold analysis JSON saved: {json_filepath}")
    return json_filepath


def process_single_ontology(signature_info, ontology):
    """Process a single ontology for a signature"""
    signature_name = signature_info['name']

    if ontology not in signature_info['ontology_files']:
        return False

    param_file = signature_info['ontology_files'][ontology]
    robust_file = signature_info['robust_file']

    try:
        console_output = StringIO()

        with redirect_stdout(console_output):
            param_data, robust_data, param_file_path = load_data_from_paths(param_file, robust_file)

            output_dir = ensure_output_directory(param_file_path, ontology)

            manifold = analyze_solution_manifold(param_data)

            save_manifold_analysis_json(manifold, param_data, robust_data, output_dir, signature_name, ontology)

            perform_automated_comparisons(param_data, robust_data, manifold['go_terms'], output_dir, ontology)

            term_info_lookup = {}
            for config_name, config_data in param_data.items():
                for keyword in config_data['keywords'][:10]:
                    term_id = keyword['go_id']
                    if term_id not in term_info_lookup:
                        term_info_lookup[term_id] = {
                            'ic': keyword['ic'],
                            'direct': keyword['genes_direct_count'],
                            'inherited': keyword['genes_inherited_count'],
                            'total': keyword['genes_total_count']
                        }

            term_to_group = {}
            for group_idx, group in enumerate(manifold['term_groups']):
                for term in group:
                    term_to_group[term['id']] = group_idx

            groups_with_frequencies = []
            for i, group in enumerate(manifold['term_groups']):
                group_ids = set(term['id'] for term in group)
                representative_id = find_group_representative(
                    group_ids,
                    manifold['go_terms'],
                    group_ids,
                    param_data,
                    manifold['term_percentages']
                )
                rep_percentage = manifold['term_percentages'].get(representative_id, 0)
                representative_name = manifold['go_terms'].get(representative_id, {}).get('name', 'Unknown')
                groups_with_frequencies.append((i, group, rep_percentage, representative_id, representative_name))

            sorted_groups = sorted(groups_with_frequencies, key=lambda x: x[2], reverse=True)

            for display_num, (original_idx, group, rep_percentage, representative_id, representative_name) in enumerate(
                    sorted_groups, 1):

                sorted_terms = sorted(group, key=lambda term: manifold['term_percentages'].get(term['id'], 0),
                                      reverse=True)

                for term in sorted_terms:
                    term_id = term['id']
                    term_name = term['name']
                    term_percentage = manifold['term_percentages'].get(term_id, 0)

                    alpha_beta_info = manifold['term_alpha_beta_counts'].get(
                        term_id,
                        {'alpha_dominant': 0, 'beta_dominant': 0, 'equal': 0}
                    )
                    alpha_count = alpha_beta_info['alpha_dominant']
                    beta_count = alpha_beta_info['beta_dominant']
                    equal_count = alpha_beta_info['equal']

                    if term_id in term_info_lookup:
                        info = term_info_lookup[term_id]
                    else:
                        info = None

                found_connections = False
                connection_output = []
                for term in sorted_terms:
                    term_id = term['id']
                    connections = get_term_connections(term_id, manifold['go_terms'], term_to_group, original_idx)
                    if connections:
                        found_connections = True
                        term_name = term['name']
                        connection_output.append(f"\t\t{term_name} ({term_id}):")
                        for target_term, rel_type, target_group in connections:
                            target_name = manifold['go_terms'].get(target_term, {}).get('name', 'Unknown')
                            target_display = next((i + 1 for i, (orig_idx, _, _, _, _) in enumerate(sorted_groups)
                                                   if orig_idx == target_group), target_group + 1)
                            connection_output.append(
                                f"\t\t\t{rel_type} → Group {target_display}: {target_name} ({target_term})"
                            )

        captured_output = console_output.getvalue()
        return True

    except Exception:
        return False


def process_single_signature(signature_info, selected_ontologies=None):
    """Process a single signature directory for specified ontologies"""
    if selected_ontologies is None:
        selected_ontologies = ['BP', 'MF', 'CC']

    available_ontologies = list(signature_info['ontology_files'].keys())
    ontologies_to_process = [ont for ont in selected_ontologies if ont in available_ontologies]

    if not ontologies_to_process:
        return False

    success_count = 0
    total_count = len(ontologies_to_process)

    for ontology in ontologies_to_process:
        if process_single_ontology(signature_info, ontology):
            success_count += 1

    if success_count == total_count:
        return True
    else:
        return success_count > 0


def main():
    """Main function to process all signatures in a base directory"""

    parser = argparse.ArgumentParser(
        description="Batch process GO term parameter analysis for multiple signatures and ontologies",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--input_path', type=str, default='results',
                        help='Path to results directory')

    parser.add_argument(
        "--filter",
        help="Only process signatures containing this substring in their name"
    )

    parser.add_argument(
        "--ontologies", "-o",
        nargs="*",
        choices=["BP", "MF", "CC"],
        default=["BP", "MF", "CC"],
        help="Specify which ontologies to process (default: all three)"
    )

    args = parser.parse_args()

    input_path = args.input_path.strip()

    if not os.path.exists(input_path):
        print(f"Error: Base path '{input_path}' does not exist!")
        return 1

    signature_dirs = get_all_signature_directories(input_path)

    if not signature_dirs:
        print(f"No valid signature directories found in '{input_path}'")
        print("Looking for directories containing:")
        print("  - BP/keyword_analysis/BP_parameter_analysis_sum.json")
        print("  - MF/keyword_analysis/MF_parameter_analysis_sum.json")
        print("  - CC/keyword_analysis/CC_parameter_analysis_sum.json")
        print("  - robust_terms_validation.json")
        return 1

    if args.filter:
        filtered_dirs = [sig for sig in signature_dirs if args.filter.lower() in sig['name'].lower()]
        if not filtered_dirs:
            print(f"No signatures found matching filter '{args.filter}'")
            return 1
        signature_dirs = filtered_dirs

    successful_count = 0
    failed_count = 0
    partial_count = 0

    progress_bar = tqdm(signature_dirs, desc="Signatures", unit="sig", dynamic_ncols=True)

    for sig_info in progress_bar:
        result = process_single_signature(sig_info, args.ontologies)
        if result is True:
            successful_count += 1
        elif result is False:
            failed_count += 1
        else:
            partial_count += 1

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    main()