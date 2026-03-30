"""
Complete GO Term Reduction Pipeline
=============================================================

Processes BP, MF, and CC ontologies separately and outputs:
- Reduced terms in standardized JSON format for each ontology
- Comprehensive statistics and hierarchy analysis
"""

import json
from collections import defaultdict
from goatools.obo_parser import GODag
import math
import os
import argparse
import glob
from tqdm import tqdm


_descendants_cache = {}

# Helper Functions
def load_my_terms(filepath):
    """Load the GO terms we're interested in."""
    terms = []
    with open(filepath, 'r') as f:
        for line in f:
            term = line.strip()
            if term:
                terms.append(term)
    return terms


def load_gene_annotations(filepath):
    """Load gene to GO term annotations."""
    gene_to_terms = {}
    current_gene = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.endswith(':'):
                current_gene = line[:-1]
                gene_to_terms[current_gene] = []
            elif line.startswith('- ') and '(GO:' in line:
                go_id = line.split('(')[-1].rstrip(')')
                if current_gene:
                    gene_to_terms[current_gene].append(go_id)

    return gene_to_terms


def get_all_descendants(go_dag, term_id):
    """Get all descendants (children) of a GO term."""
    if term_id in _descendants_cache:
        return _descendants_cache[term_id]

    # Ab hier: alles exakt wie vorher
    if term_id not in go_dag:
        return set()

    descendants = set()
    to_visit = [term_id]
    visited = set()

    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)

        if current in go_dag:
            children = go_dag[current].children
            for child in children:
                descendants.add(child.id)
                to_visit.append(child.id)

    _descendants_cache[term_id] = descendants
    return descendants


def get_all_ancestors(go_dag, term_id):
    """Get all ancestors (parents) of a GO term."""
    if term_id not in go_dag:
        return set()

    term_obj = go_dag[term_id]
    ancestors = set(term_obj.get_all_parents())
    return ancestors


def get_direct_children(go_dag, term_id):
    """Get direct children (one is_a edge away) of a GO term."""
    if term_id not in go_dag:
        return set()

    term_obj = go_dag[term_id]
    children = set()
    for child in term_obj.children:
        children.add(child.id)

    return children


def load_ic_values(filepath):
    """Load information content values."""
    with open(filepath, 'r') as f:
        ic_data = json.load(f)

    ic_normalized = {}
    frequency_data = {}
    gene_counts = {}

    for go_id, data in ic_data.items():
        ic_normalized[go_id] = data.get('ic_normalized', None)
        frequency_data[go_id] = data.get('frequency', None)
        gene_counts[go_id] = data.get('gene_count', 0)

    return ic_normalized, frequency_data, gene_counts


def calculate_term_stats(term, go_dag, gene_to_terms):
    """Calculate direct and inherited gene counts for a term."""
    direct_count = 0
    inherited_count = 0
    descendants = get_all_descendants(go_dag, term)

    for gene, annotations in gene_to_terms.items():
        if term in annotations:
            direct_count += 1
        elif any(annot in descendants for annot in annotations):
            inherited_count += 1

    return {
        'direct': direct_count,
        'inherited': inherited_count,
        'total_genes': direct_count + inherited_count
    }


def calculate_tiebreaker_metric(term, ic_normalized, frequency_data, term_stats, n_signature):
    """Calculate tiebreaker metric for gene assignment."""
    ic_norm = ic_normalized.get(term, 0)
    if ic_norm is None:
        ic_norm = 0

    frequency = frequency_data.get(term, 0)
    if frequency is None:
        frequency = 0

    total_genes = term_stats.get('total_genes', 0)
    if total_genes == 0:
        return 0

    expectation = max(1, n_signature * math.exp(frequency))
    metric = math.log(total_genes) * ic_norm / expectation

    return metric


# Phase 1: Representative selection

def select_representatives_unified(my_terms, go_dag, gene_to_terms, ic_normalized, term_stats,
                                   coverage_threshold=0.95, ic_threshold=0.2):
    """Select representatives using unified greedy algorithm."""

    my_terms_set = set(my_terms)
    pool = set(my_terms)

    for term in my_terms:
        ancestors = get_all_ancestors(go_dag, term)
        pool.update(ancestors)

    n_genes = len(gene_to_terms)
    n_my_terms = len(my_terms)

    representatives = []
    covered_terms = set()
    excluded_terms = set()

    while len(covered_terms) < len(my_terms):
        coverage_ratio = len(covered_terms) / len(my_terms)

        if coverage_ratio >= coverage_threshold:
            break

        uncovered_terms = my_terms_set - covered_terms

        if not uncovered_terms:
            break

        candidates = pool - excluded_terms

        if not candidates:
            break

        candidates_with_metrics = []

        for candidate in candidates:
            ic_norm = ic_normalized.get(candidate, 0)
            if ic_norm is None or ic_norm < ic_threshold:
                continue

            descendants_of_candidate = get_all_descendants(go_dag, candidate)
            coverage_of_uncovered = uncovered_terms & (descendants_of_candidate | {candidate})

            if not coverage_of_uncovered:
                continue

            total_genes = term_stats[candidate]['total_genes']
            coverage_count = len(coverage_of_uncovered)

            metric = (total_genes / n_genes) * ic_norm * (coverage_count / n_my_terms)

            candidates_with_metrics.append({
                'term': candidate,
                'metric': metric,
                'coverage': coverage_count,
                'ic': ic_norm,
                'total_genes': total_genes,
                'coverage_terms': coverage_of_uncovered
            })

        if not candidates_with_metrics:
            break

        candidates_with_metrics.sort(key=lambda x: x['metric'], reverse=True)
        best = candidates_with_metrics[0]

        representatives.append({
            'term': best['term'],
            'metric': best['metric'],
            'coverage': best['coverage'],
            'ic': best['ic'],
            'total_genes': best['total_genes'],
            'source': 'FROM MY_TERMS' if best['term'] in my_terms_set else 'ANCESTOR'
        })

        covered_terms.update(best['coverage_terms'])

        descendants_of_selected = get_all_descendants(go_dag, best['term'])
        excluded_terms.add(best['term'])
        excluded_terms.update(descendants_of_selected)

    return {
        'representatives': representatives,
        'covered_terms': covered_terms,
        'uncovered_terms': my_terms_set - covered_terms,
        'final_coverage': len(covered_terms) / len(my_terms),
        'term_stats': term_stats
    }


# Phase 2: Hierarchical gene-base reduction

def map_genes_to_representatives(representatives, go_dag, gene_to_terms,
                                 ic_normalized, frequency_data, term_stats, n_signature):
    """Map each gene to the representative that covers most of its terms."""

    rep_to_genes = {r['term']: [] for r in representatives}
    gene_assignments = {}

    for gene, annotations in gene_to_terms.items():
        best_rep = None
        candidates = []

        for rep_info in representatives:
            rep_term = rep_info['term']
            descendants = get_all_descendants(go_dag, rep_term)

            coverage = 0
            for annot in annotations:
                if annot == rep_term or annot in descendants:
                    coverage += 1

            if coverage > 0:
                candidates.append((rep_term, coverage))

        if not candidates:
            continue

        max_coverage = max(c[1] for c in candidates)
        tied_reps = [c[0] for c in candidates if c[1] == max_coverage]

        if len(tied_reps) == 1:
            best_rep = tied_reps[0]
        else:
            best_metric = -1
            for rep_term in tied_reps:
                metric = calculate_tiebreaker_metric(
                    rep_term, ic_normalized, frequency_data,
                    term_stats[rep_term], n_signature
                )
                if metric > best_metric:
                    best_metric = metric
                    best_rep = rep_term

        if best_rep:
            rep_to_genes[best_rep].append(gene)
            gene_assignments[gene] = best_rep

    rep_to_genes = {rep: genes for rep, genes in rep_to_genes.items() if len(genes) > 0}

    return rep_to_genes, gene_assignments


def recursive_hierarchy_expansion(node_term, node_genes, go_dag, gene_to_terms,
                                  ic_normalized, frequency_data, term_stats,
                                  n_signature, hierarchy_map, level=0, max_level=10):
    """Recursively expand hierarchy by assigning genes to children."""
    if level >= max_level:
        return

    children = get_direct_children(go_dag, node_term)

    if not children or not node_genes:
        return

    child_to_genes = {child: [] for child in children}

    for gene in node_genes:
        annotations = gene_to_terms[gene]
        best_child = None
        candidates = []

        for child in children:
            descendants = get_all_descendants(go_dag, child)

            coverage = 0
            for annot in annotations:
                if annot == child or annot in descendants:
                    coverage += 1

            if coverage > 0:
                candidates.append((child, coverage))

        if not candidates:
            continue

        max_coverage = max(c[1] for c in candidates)
        tied_children = [c[0] for c in candidates if c[1] == max_coverage]

        if len(tied_children) == 1:
            best_child = tied_children[0]
        else:
            best_metric = -1
            for child in tied_children:
                metric = calculate_tiebreaker_metric(
                    child, ic_normalized, frequency_data,
                    term_stats.get(child, {'total_genes': 0}), n_signature
                )
                if metric > best_metric:
                    best_metric = metric
                    best_child = child

        if best_child:
            child_to_genes[best_child].append(gene)

    for child, child_genes in child_to_genes.items():
        if child_genes:
            hierarchy_map[child] = child_genes
            recursive_hierarchy_expansion(
                child, child_genes, go_dag, gene_to_terms,
                ic_normalized, frequency_data, term_stats,
                n_signature, hierarchy_map, level + 1, max_level
            )


def build_hierarchical_reduction(rep_to_genes, go_dag, gene_to_terms,
                                 ic_normalized, frequency_data, term_stats, n_signature):
    """Build complete hierarchical structure."""

    hierarchy_map = {}

    for rep_term, rep_genes in rep_to_genes.items():
        hierarchy_map[rep_term] = rep_genes

        recursive_hierarchy_expansion(
            rep_term, rep_genes, go_dag, gene_to_terms,
            ic_normalized, frequency_data, term_stats,
            n_signature, hierarchy_map, level=0
        )

    return hierarchy_map


def analyze_hierarchy(hierarchy_map, my_terms, go_dag, gene_to_terms, representatives):
    """Analyze the hierarchical structure and create reduced my_terms list."""

    my_terms_set = set(my_terms)
    reduced_my_terms = set(hierarchy_map.keys()) & my_terms_set

    genes_in_hierarchy = set()
    for genes in hierarchy_map.values():
        genes_in_hierarchy.update(genes)

    return {
        'reduced_my_terms': reduced_my_terms,
        'hierarchy_map': hierarchy_map,
        'total_hierarchy_nodes': len(hierarchy_map),
        'genes_in_hierarchy': genes_in_hierarchy
    }


# Output

def create_reduced_terms_json(reduced_my_terms, hierarchy_map, term_stats,
                              ic_normalized, frequency_data, go_dag):
    output = {}

    for term in sorted(reduced_my_terms):
        term_name = go_dag[term].name if term in go_dag else "Unknown"
        stats = term_stats.get(term, {'direct': 0, 'inherited': 0, 'total_genes': 0})
        ic_value = ic_normalized.get(term, None)
        freq_value = frequency_data.get(term, None)

        output[term] = {
            "go_annotation": term_name,
            "genes_direct_count": stats['direct'],
            "genes_inherited_count": stats['inherited'],
            "genes_total_count": stats['total_genes'],
            "ic_normalized": ic_value,
            "frequency": freq_value
        }

    return output




# Main Pipeline
def process_ontology(ontology, base_path, output_dir, go_dag):
    """Process a single ontology (BP, MF, or CC)."""

    ontology_lower = ontology.lower()

    my_terms_path = os.path.join(base_path, ontology.upper(), f"my_terms_{ontology_lower}.txt")
    gene_mapping_path = os.path.join(
        base_path,
        ontology.upper(),
        f"mapping_genes_to_{ontology_lower}",
        f"map_genes_to_{ontology_lower}.txt"
    )
    ic_path = os.path.join("data", "GO_IC", f"{ontology_lower}_ic.json")

    if not os.path.exists(my_terms_path):
        return None

    if not os.path.exists(gene_mapping_path):
        return None

    if not os.path.exists(ic_path):
        return None

    my_terms = load_my_terms(my_terms_path)
    gene_to_terms = load_gene_annotations(gene_mapping_path)
    ic_normalized, frequency_data, gene_counts = load_ic_values(ic_path)

    if len(my_terms) == 0:
        return None

    all_terms = set(my_terms)
    for term in my_terms:
        all_terms.update(get_all_ancestors(go_dag, term))

    term_stats = {}
    for term in all_terms:
        term_stats[term] = calculate_term_stats(term, go_dag, gene_to_terms)

    representatives_result = select_representatives_unified(
        my_terms, go_dag, gene_to_terms, ic_normalized, term_stats,
        coverage_threshold=0.95, ic_threshold=0.2
    )

    representatives_result['term_stats'] = term_stats

    n_signature = len(gene_to_terms)

    rep_to_genes, gene_assignments = map_genes_to_representatives(
        representatives_result['representatives'], go_dag, gene_to_terms,
        ic_normalized, frequency_data, term_stats, n_signature
    )

    hierarchy_map = build_hierarchical_reduction(
        rep_to_genes, go_dag, gene_to_terms,
        ic_normalized, frequency_data, term_stats, n_signature
    )

    reduction_result = analyze_hierarchy(
        hierarchy_map, my_terms, go_dag, gene_to_terms,
        representatives_result['representatives']
    )

    reduced_terms_json = create_reduced_terms_json(
        reduction_result['reduced_my_terms'],
        hierarchy_map,
        term_stats,
        ic_normalized,
        frequency_data,
        go_dag
    )

    output_file = os.path.join(output_dir, f"reduced_terms_{ontology_lower}.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(reduced_terms_json, f, indent=2)

    return {
        'ontology': ontology,
        'reduced_count': len(reduction_result['reduced_my_terms']),
        'original_count': len(my_terms),
        'reduction_percentage': (1 - len(reduction_result['reduced_my_terms']) / len(my_terms)) * 100,
        'output_file': output_file
    }


def main():
    """Main execution pipeline for all signatures and all ontologies."""

    parser = argparse.ArgumentParser(
        description="GO Term Reduction Pipeline"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Base directory containing signature folders"
    )
    args = parser.parse_args()

    base_input_dir = args.input_dir

    signatures = [
        d for d in glob.glob(os.path.join(base_input_dir, "*"))
        if os.path.isdir(d)
    ]

    if not signatures:
        return

    go_dag = GODag("data/go-basic.obo")

    ontologies = ["BP", "MF", "CC"]

    for signature_dir in tqdm(signatures, desc="Signatures", unit="sig"):
        output_dir = os.path.join(signature_dir, "representatives_analysis")

        for ontology in ontologies:
            process_ontology(
                ontology,
                base_path=signature_dir,
                output_dir=output_dir,
                go_dag=go_dag
            )


if __name__ == "__main__":
    main()