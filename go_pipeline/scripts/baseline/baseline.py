#!/usr/bin/env python3

import argparse
import json
import random
import sys
from collections import defaultdict
from functools import lru_cache
from io import StringIO
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from goatools.obo_parser import GODag
from tqdm import tqdm


class _NullWriter:
    def write(self, *a, **kw): pass
    def flush(self, *a, **kw): pass


def _load_godag_silent(obo_file):
    null = _NullWriter()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = null
    try:
        return GODag(obo_file, optional_attrs='relationship', prt=null)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


_GLOBAL_GO_DAG = None
_GLOBAL_GENE_ANNOTATIONS = None
_GLOBAL_ALL_GENES = None
_GLOBAL_OBO_FILE = None


def _init_worker(obo_file, gene_annotations, all_genes):
    global _GLOBAL_GO_DAG, _GLOBAL_GENE_ANNOTATIONS, _GLOBAL_ALL_GENES, _GLOBAL_OBO_FILE

    _GLOBAL_GO_DAG = _load_godag_silent(obo_file)

    _GLOBAL_OBO_FILE = obo_file
    _GLOBAL_GENE_ANNOTATIONS = gene_annotations
    _GLOBAL_ALL_GENES = all_genes


def load_gene_list(gene_list_file):
    valid_genes = set()
    with open(gene_list_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith('#'):
                valid_genes.add(gene)
    return valid_genes


def load_gene_annotations(gaf_file, go_dag, valid_gene_set):
    gene_annotations = {
        'BP': defaultdict(list),
        'MF': defaultdict(list),
        'CC': defaultdict(list),
    }
    genes_with_annotations = set()

    aspect_map = {'P': 'BP', 'F': 'MF', 'C': 'CC'}

    with open(gaf_file, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 13:
                continue

            gene_symbol = parts[2]
            go_id = parts[4]
            aspect = parts[8]

            ns_key = aspect_map.get(aspect)
            if ns_key is None or gene_symbol not in valid_gene_set:
                continue

            genes_with_annotations.add(gene_symbol)
            if go_id in go_dag:
                gene_annotations[ns_key][gene_symbol].append(go_id)

    return gene_annotations, list(genes_with_annotations)


def load_ic_data(ic_file):
    with open(ic_file, 'r') as f:
        return json.load(f)


def create_random_signature(all_genes, signature_length):
    return random.sample(all_genes, min(signature_length, len(all_genes)))


def get_all_parents(go_id, go_dag):
    return _get_all_parents_cached(go_id, id(go_dag))


@lru_cache(maxsize=50000)
def _get_all_parents_cached(go_id, go_dag_id):
    go_dag = _GLOBAL_GO_DAG
    if go_id not in go_dag:
        return frozenset()
    parents = set()
    for parent in go_dag[go_id].parents:
        parents.add(parent.id)
        parents.update(_get_all_parents_cached(parent.id, go_dag_id))
    return frozenset(parents)


def get_all_children(go_id, go_dag):
    return _get_all_children_cached(go_id, id(go_dag))


@lru_cache(maxsize=50000)
def _get_all_children_cached(go_id, go_dag_id):
    go_dag = _GLOBAL_GO_DAG
    if go_id not in go_dag:
        return frozenset()
    children = set()
    for child in go_dag[go_id].children:
        children.add(child.id)
        children.update(_get_all_children_cached(child.id, go_dag_id))
    return frozenset(children)


def get_term_relationships(go_id, go_dag):
    return _get_term_relationships_cached(go_id, id(go_dag))


@lru_cache(maxsize=50000)
def _get_term_relationships_cached(go_id, go_dag_id):
    go_dag = _GLOBAL_GO_DAG
    if go_id not in go_dag:
        return frozenset()
    relationships = set()
    term = go_dag[go_id]
    if hasattr(term, 'relationship'):
        for related_terms in term.relationship.values():
            for related_term in related_terms:
                relationships.add(related_term.id)
    return frozenset(relationships)


def filter_most_specific_terms(term_list, go_dag):
    if not term_list:
        return []
    term_set = set(term_list)
    term_parents = {term: get_all_parents(term, go_dag) for term in term_list}
    return [
        term for term in term_list
        if not any(term in term_parents[other] for other in term_set if other != term)
    ]


def get_filtered_annotations_for_signature(signature, gene_annotations, namespace, go_dag):
    result = {}
    for gene in signature:
        if gene in gene_annotations[namespace]:
            filtered = filter_most_specific_terms(gene_annotations[namespace][gene], go_dag)
            result[gene] = filtered
    return result


def calculate_enrichment_for_signature(filtered_annotations, signature_length, ic_data, go_dag, namespace):
    all_terms = {
        term
        for terms in filtered_annotations.values()
        for term in terms
        if term in go_dag and go_dag[term].namespace == namespace
    }

    enrichments = {}
    for term in all_terms:
        direct = sum(1 for gene_terms in filtered_annotations.values() if term in gene_terms)

        inherited_genes = set()
        for gene, gene_terms in filtered_annotations.items():
            for gene_term in gene_terms:
                if gene_term != term and gene_term in go_dag and go_dag[gene_term].namespace == namespace:
                    if term in get_all_parents(gene_term, go_dag):
                        inherited_genes.add(gene)
                        break

        frequency = ic_data.get(term, {}).get('frequency', 0.0)
        expectation = max(1, signature_length * frequency)
        enrichments[term] = (direct + len(inherited_genes)) / expectation

    return enrichments


def get_extreme_terms(enrichments, baseline_stats, percentile):
    percentile_key = f'enrichment_{percentile}th'
    return {
        term for term, enrichment in enrichments.items()
        if term in baseline_stats
        and (bp := baseline_stats[term].get(percentile_key, float('inf'))) > 0
        and enrichment / bp > 1
    }


def group_terms_by_ancestry(extreme_terms, go_dag):
    if not extreme_terms:
        return 0

    related = defaultdict(set)
    for term1 in extreme_terms:
        parents1 = get_all_parents(term1, go_dag)
        children1 = get_all_children(term1, go_dag)
        for term2 in extreme_terms:
            if term1 != term2 and (term2 in parents1 or term2 in children1):
                related[term1].add(term2)
                related[term2].add(term1)

    visited = set()
    total = 0
    for term in extreme_terms:
        if term not in visited:
            group, queue = set(), [term]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                group.add(current)
                queue.extend(n for n in related[current] if n not in visited)
            if len(group) >= 2:
                total += len(group)
    return total


def count_terms_with_relationships(extreme_terms, go_dag):
    if not extreme_terms:
        return 0
    result = set()
    for term1 in extreme_terms:
        rels = get_term_relationships(term1, go_dag)
        if any(term2 in rels for term2 in extreme_terms if term2 != term1):
            result.add(term1)
    return len(result)


def calculate_sr_max(extreme_terms, enrichments, baseline_stats, percentile):
    if not extreme_terms:
        return 0.0
    percentile_key = f'enrichment_{percentile}th'
    sr_values = [
        enrichments[term] / baseline_stats[term][percentile_key]
        for term in extreme_terms
        if term in baseline_stats and term in enrichments
        and baseline_stats[term].get(percentile_key, 0) > 0
    ]
    return max(sr_values) if sr_values else 0.0


def terms_are_related(term1, term2, go_dag):
    if term1 == term2:
        return True
    parents2 = get_all_parents(term2, go_dag)
    if term1 in parents2:
        return True
    parents1 = get_all_parents(term1, go_dag)
    return term2 in parents1


def terms_have_relationship_connection(term1, term2, go_dag):
    if term1 == term2:
        return False
    ancestors1 = get_all_parents(term1, go_dag)
    descendants2 = get_all_children(term2, go_dag)
    terms1 = [term1] + list(ancestors1)
    terms2 = [term2] + list(descendants2)
    for t1 in terms1:
        rels = get_term_relationships(t1, go_dag)
        if any(t2 in rels for t2 in terms2):
            return True
    return False


def _connected_components(genes, adjacency):
    visited = set()
    total = 0
    for gene in genes:
        if gene not in visited:
            component, queue = set(), [gene]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                queue.extend(n for n in adjacency[current] if n not in visited)
            if len(component) >= 2:
                total += len(component)
    return total


def count_genes_with_related_extreme_terms(filtered_annotations, extreme_terms, go_dag):
    if not extreme_terms or not filtered_annotations:
        return 0

    gene_extreme = {
        gene: [t for t in terms if t in extreme_terms]
        for gene, terms in filtered_annotations.items()
        if any(t in extreme_terms for t in terms)
    }
    if len(gene_extreme) < 2:
        return len(gene_extreme)

    genes = list(gene_extreme.keys())
    adjacency = defaultdict(set)
    for i, gene1 in enumerate(genes):
        for gene2 in genes[i + 1:]:
            if any(
                terms_are_related(t1, t2, go_dag)
                for t1 in gene_extreme[gene1]
                for t2 in gene_extreme[gene2]
            ):
                adjacency[gene1].add(gene2)
                adjacency[gene2].add(gene1)

    return _connected_components(genes, adjacency)


def count_genes_with_relationship_between_extreme_terms(filtered_annotations, extreme_terms, go_dag):
    if not extreme_terms or not filtered_annotations:
        return 0

    gene_extreme = {
        gene: [t for t in terms if t in extreme_terms]
        for gene, terms in filtered_annotations.items()
        if any(t in extreme_terms for t in terms)
    }
    if len(gene_extreme) < 2:
        return len(gene_extreme)

    genes = list(gene_extreme.keys())
    adjacency = defaultdict(set)
    for i, gene1 in enumerate(genes):
        for gene2 in genes[i + 1:]:
            if any(
                terms_have_relationship_connection(t1, t2, go_dag)
                for t1 in gene_extreme[gene1]
                for t2 in gene_extreme[gene2]
            ):
                adjacency[gene1].add(gene2)
                adjacency[gene2].add(gene1)

    return _connected_components(genes, adjacency)


def calculate_statistics(values):
    if not values:
        return None
    arr = np.array(values)
    return {
        'mean':           float(np.mean(arr)),
        'std':            float(np.std(arr)),
        'median':         float(np.median(arr)),
        'min':            float(np.min(arr)),
        'max':            float(np.max(arr)),
        'percentile_25':  float(np.percentile(arr, 25)),
        'percentile_75':  float(np.percentile(arr, 75)),
        'percentile_90':  float(np.percentile(arr, 90)),
        'percentile_95':  float(np.percentile(arr, 95)),
        'percentile_99':  float(np.percentile(arr, 99)),
        'n_observations': len(values),
    }


def calculate_baseline_statistics(values):
    if not values:
        return None
    arr = np.array(values)
    return {
        'mean':             float(np.mean(arr)),
        'std':              float(np.std(arr)),
        'median':           float(np.median(arr)),
        'min':              float(np.min(arr)),
        'max':              float(np.max(arr)),
        'enrichment_75th':  float(np.percentile(arr, 75)),
        'enrichment_90th':  float(np.percentile(arr, 90)),
        'enrichment_95th':  float(np.percentile(arr, 95)),
        'enrichment_99th':  float(np.percentile(arr, 99)),
    }


def _baseline_worker(args):
    signature, namespace, ic_data_ns, namespace_full = args
    filtered = get_filtered_annotations_for_signature(
        signature, _GLOBAL_GENE_ANNOTATIONS, namespace, _GLOBAL_GO_DAG
    )
    return calculate_enrichment_for_signature(
        filtered, len(signature), ic_data_ns, _GLOBAL_GO_DAG, namespace_full
    )


def _cluster_worker(args):
    signature, namespace, ic_data_ns, namespace_full, baseline_stats, percentile = args

    filtered = get_filtered_annotations_for_signature(
        signature, _GLOBAL_GENE_ANNOTATIONS, namespace, _GLOBAL_GO_DAG
    )
    enrichments = calculate_enrichment_for_signature(
        filtered, len(signature), ic_data_ns, _GLOBAL_GO_DAG, namespace_full
    )
    extreme_terms = get_extreme_terms(enrichments, baseline_stats, percentile)

    n_extreme = len(extreme_terms)
    sig_len = len(signature)

    if n_extreme > 0:
        n_grouped = group_terms_by_ancestry(extreme_terms, _GLOBAL_GO_DAG)
        n_relationships = count_terms_with_relationships(extreme_terms, _GLOBAL_GO_DAG)
        sr_max = calculate_sr_max(extreme_terms, enrichments, baseline_stats, percentile)
        n_genes_related = count_genes_with_related_extreme_terms(filtered, extreme_terms, _GLOBAL_GO_DAG)
        n_genes_relationship = count_genes_with_relationship_between_extreme_terms(
            filtered, extreme_terms, _GLOBAL_GO_DAG
        )
        genes_extreme = {gene for gene, terms in filtered.items() if any(t in extreme_terms for t in terms)}
        n_genes_extreme = len(genes_extreme)
        extreme_fraction_genes = n_genes_extreme / sig_len
    else:
        n_grouped = n_relationships = n_genes_related = n_genes_relationship = n_genes_extreme = 0
        sr_max = extreme_fraction_genes = 0.0

    return {
        'extreme_terms':              n_extreme,
        'extreme_fraction':           n_extreme / sig_len,
        'extreme_genes_count':        n_genes_extreme,
        'extreme_fraction_genes':     extreme_fraction_genes,
        'grouped_terms':              n_grouped,
        'grouped_fraction':           n_grouped / sig_len,
        'relationship_terms':         n_relationships,
        'relationship_fraction':      n_relationships / sig_len,
        'SR_max':                     sr_max,
        'related_genes_count':        n_genes_related,
        'related_genes_fraction':     n_genes_related / sig_len,
        'relationship_genes_count':   n_genes_relationship,
        'relationship_genes_fraction': n_genes_relationship / sig_len,
    }


def process_namespace(namespace, signatures, gene_annotations, go_dag, ic_data, obo_file, n_cores=None):
    namespace_map = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component',
    }
    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)

    worker_args = [(sig, namespace, ic_data, namespace_map[namespace]) for sig in signatures]

    with Pool(n_cores, initializer=_init_worker, initargs=(obo_file, gene_annotations, None)) as pool:
        results = list(tqdm(
            pool.imap(_baseline_worker, worker_args),
            total=len(signatures),
            desc=f"Baseline {namespace}",
            unit="sig",
            ncols=100,
        ))

    term_enrichments = defaultdict(list)
    for enrichments in results:
        for term, enrichment in enrichments.items():
            term_enrichments[term].append(enrichment)

    baseline_stats = {}
    for term, enrichment_list in term_enrichments.items():
        stats = calculate_baseline_statistics(enrichment_list)
        if stats:
            stats['name'] = go_dag[term].name if term in go_dag else 'Unknown'
            baseline_stats[term] = stats

    return baseline_stats


def cluster_analysis_for_namespace(namespace, signatures, gene_annotations, go_dag, ic_data,
                                   baseline_stats, percentile, obo_file, n_cores=None):
    namespace_map = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component',
    }
    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)

    worker_args = [
        (sig, namespace, ic_data, namespace_map[namespace], baseline_stats, percentile)
        for sig in signatures
    ]

    with Pool(n_cores, initializer=_init_worker, initargs=(obo_file, gene_annotations, None)) as pool:
        results = list(tqdm(
            pool.imap(_cluster_worker, worker_args),
            total=len(signatures),
            desc=f"Cluster {namespace} p{percentile}",
            unit="sig",
            ncols=100,
        ))

    keys = [
        'extreme_terms', 'extreme_fraction', 'extreme_genes_count', 'extreme_fraction_genes',
        'grouped_terms', 'grouped_fraction', 'relationship_terms', 'relationship_fraction',
        'SR_max', 'related_genes_count', 'related_genes_fraction',
        'relationship_genes_count', 'relationship_genes_fraction',
    ]

    collected = {k: [r[k] for r in results] for k in keys}

    stats = {k: calculate_statistics(v) for k, v in collected.items()}
    raw_values = dict(collected)

    return stats, raw_values


def main():
    global _GLOBAL_GO_DAG

    parser = argparse.ArgumentParser(
        description='Create enrichment baseline statistics from random gene signatures.'
    )
    parser.add_argument('--num_of_iterations', type=int, required=True)
    parser.add_argument('--signature_lengths',  type=str, required=True,
                        help='Space-separated signature lengths (e.g., "50 100 200")')
    parser.add_argument('--percentile',    type=str, default='90',
                        help='Percentile threshold(s) (e.g., "90" or "90 95 99")')
    parser.add_argument('--n_cores',       type=int, default=None)
    parser.add_argument('--gene_list',     type=str, default='data/all_human_genes/all_genes.txt')
    parser.add_argument('--gaf_file',      type=str, default='data/goa_human.gaf')
    parser.add_argument('--obo_file',      type=str, default='data/go-basic.obo')
    parser.add_argument('--bp_ic_file',    type=str, default='data/GO_IC/bp_ic.json')
    parser.add_argument('--mf_ic_file',    type=str, default='data/GO_IC/mf_ic.json')
    parser.add_argument('--cc_ic_file',    type=str, default='data/GO_IC/cc_ic.json')
    parser.add_argument('--output_dir',    type=str, default='.')
    parser.add_argument('--num_repeats',   type=int, default=1)
    args = parser.parse_args()

    signature_lengths = [int(x) for x in args.signature_lengths.split()]
    max_length = max(signature_lengths)
    percentiles = [int(x) for x in args.percentile.split()]
    n_cores = args.n_cores if args.n_cores else max(1, cpu_count() - 1)

    valid_gene_set = load_gene_list(args.gene_list)

    go_dag = _load_godag_silent(args.obo_file)

    _GLOBAL_GO_DAG = go_dag

    gene_annotations, all_genes = load_gene_annotations(args.gaf_file, go_dag, valid_gene_set)

    ic_data = {
        'BP': load_ic_data(args.bp_ic_file),
        'MF': load_ic_data(args.mf_ic_file),
        'CC': load_ic_data(args.cc_ic_file),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sig_length in signature_lengths:
        num_iterations = int(args.num_of_iterations * max_length / sig_length)

        all_baseline_stats = {ns: {} for ns in ['BP', 'MF', 'CC']}
        all_cluster_stats = {p: {ns: {} for ns in ['BP', 'MF', 'CC']} for p in percentiles}
        all_cluster_raw_values = {p: {ns: {} for ns in ['BP', 'MF', 'CC']} for p in percentiles}

        for repeat_idx in tqdm(range(args.num_repeats),
                               desc=f"Length {sig_length}", unit="repeat", ncols=100):

            baseline_signatures = [
                create_random_signature(all_genes, sig_length)
                for _ in range(num_iterations)
            ]

            baseline_stats = {}
            for namespace in ['BP', 'MF', 'CC']:
                baseline_stats[namespace] = process_namespace(
                    namespace, baseline_signatures, gene_annotations,
                    go_dag, ic_data[namespace], args.obo_file, n_cores
                )

            cluster_signatures = [
                create_random_signature(all_genes, sig_length)
                for _ in range(num_iterations)
            ]

            for percentile in percentiles:
                for namespace in ['BP', 'MF', 'CC']:
                    stats, raw_vals = cluster_analysis_for_namespace(
                        namespace, cluster_signatures, gene_annotations, go_dag,
                        ic_data[namespace], baseline_stats[namespace], percentile,
                        args.obo_file, n_cores
                    )
                    all_cluster_stats[percentile][namespace][f'repeat_{repeat_idx}'] = stats
                    all_cluster_raw_values[percentile][namespace][f'repeat_{repeat_idx}'] = raw_vals

            for namespace in ['BP', 'MF', 'CC']:
                for go_term, term_stats in baseline_stats[namespace].items():
                    if go_term not in all_baseline_stats[namespace]:
                        all_baseline_stats[namespace][go_term] = {}
                    all_baseline_stats[namespace][go_term][f'repeat_{repeat_idx}'] = term_stats

        base_out = output_dir / f"baseline_{sig_length}"
        base_out.mkdir(parents=True, exist_ok=True)

        with open(base_out / 'enrichment_baseline_stats.json', 'w') as f:
            json.dump(all_baseline_stats, f, indent=2)

        for percentile in percentiles:
            with open(base_out / f'cluster_statistics_{percentile}th.json', 'w') as f:
                json.dump(all_cluster_stats[percentile], f, indent=2)
            with open(base_out / f'cluster_raw_values_{percentile}th.json', 'w') as f:
                json.dump(all_cluster_raw_values[percentile], f, indent=2)


if __name__ == '__main__':
    main()