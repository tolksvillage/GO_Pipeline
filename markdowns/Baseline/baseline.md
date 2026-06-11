# baseline.py

## Purpose

This script constructs an **empirical null distribution** for Gene Ontology (GO) enrichment analysis. It answers the question: *what do enrichment scores and cluster structure metrics look like in purely random gene lists?* The resulting distributions define the statistical baseline against which real gene signatures are later evaluated.

---

## Biological and mathematical background

### The Gene Ontology as a directed acyclic graph (DAG)

GO terms are organized in a hierarchy: specific terms (e.g. "mitotic spindle assembly") are children of more general terms (e.g. "cell division"), which are in turn children of even more general ones. This forms a **directed acyclic graph (DAG)** — a tree-like structure but where a node can have multiple parents. The three independent sub-graphs are:

- **BP** — Biological Process
- **MF** — Molecular Function
- **CC** — Cellular Component

Each gene is annotated to one or more GO terms per namespace. A gene annotated to a specific child term is also implicitly annotated to all its ancestors up to the root.

### Information Content (IC)

Each GO term has a **frequency** in the genome: the fraction of all genes annotated to it (directly or via inheritance). A very general term like "biological process" applies to nearly all genes (high frequency), while a very specific term like "kinetochore assembly" applies to few genes (low frequency). The IC of a term is defined as:

```
IC(t) = -log2(frequency(t))
```

High IC = specific, informative term. Low IC = general, less informative term. The IC values are precomputed and loaded from the `bp_ic.json`, `mf_ic.json`, and `cc_ic.json` files.

---

## Step 1 — Filtering to most-specific terms

For each gene in a signature, the full list of annotated GO terms is retrieved. However, using all terms including ancestors would introduce redundancy: if a gene is annotated to "kinetochore assembly", it is implicitly also annotated to "cell division" — but adding "cell division" explicitly carries no additional information.

To avoid this, only the **most specific terms** per gene are kept. A term is removed if any other term in the same gene's annotation list is its descendant in the DAG. This is checked by computing the full transitive ancestor set of each term recursively:

```
ancestors(t) = parents(t) ∪ ancestors(parents(t))
```

A term `t` is removed if there exists another term `t'` in the gene's annotation list such that `t ∈ ancestors(t')`.

---

## Step 2 — Enrichment score per GO term

For every GO term `t` that appears in any gene's filtered annotation set, an enrichment score is computed:

```
E(t) = (direct(t) + inherited(t)) / expectation(t)
```

Where:

- **direct(t)** — number of genes in the signature that have `t` as one of their filtered (most-specific) annotations
- **inherited(t)** — number of genes whose filtered terms are descendants of `t` (i.e. `t` is an ancestor of their annotated terms), but which do not have `t` directly. This captures genes that are "implicitly" associated with `t` through specificity.
- **expectation(t)** — the expected number of genes one would find annotated to `t` in a random signature of the same size:

```
expectation(t) = max(1, signature_length × frequency(t))
```

##### frequency(t) — Definition

`frequency(t)` is the fraction of all human genes annotated to GO term `t`, either directly or via inheritance through a more specific child term:

**Example:** if 1,000 out of 19,273 human genes are associated with "cell division", then:

The expectation then states: in a random signature of length `n`, one would expect to find approximately `n × frequency(t)` genes carrying term `t`. For a signature of length 200:

The `max(1, ...)` prevents extremely rare terms (e.g. `frequency = 0.0001`) from producing an expectation close to zero (e.g. 0.02), which would result in astronomically large enrichment scores that are mathematically correct but biologically misleading. The frequency values are precomputed genome-wide and loaded from the IC files (`bp_ic.json`, `mf_ic.json`, `cc_ic.json`).s

---

## Step 3 — Building the baseline distribution

The script generates `N` random gene signatures (by sampling without replacement from the full human gene pool) for each target signature length. Each random signature is processed in parallel using multiprocessing, and for every GO term encountered across all random signatures, its enrichment scores are collected into a list.

After all `N` signatures are processed, the distribution of enrichment scores per term is summarized:

```
baseline_stats[term] = {
    mean, std, median, min, max,
    enrichment_75th, enrichment_90th, enrichment_95th, enrichment_99th
}
```

These percentile values define the **null distribution thresholds**: the enrichment score a term would reach in the top X% of random signatures.

---

## Step 4 — Defining extreme terms

A GO term `t` in a signature is called **extreme** if its enrichment score exceeds the baseline percentile threshold:

```
E(t) > baseline_stats[t]['enrichment_{p}th']
```

Where `p` is the chosen percentile (e.g. 90 or 99). This means the term is enriched more than it would be in the top `(100-p)%` of random signatures.

---

## Step 5 — Cluster metrics

Once extreme terms are identified, a set of structural metrics is computed for each random signature. These describe the **topology** of the extreme terms within the GO DAG:

### `extreme_fraction_genes`
```
extreme_fraction_genes = |genes with ≥1 extreme term| / signature_length
```
The fraction of genes in the signature that carry at least one extreme GO term.

### `SR_max` (Signal Ratio maximum)
For each extreme term `t`, the signal ratio is:
```
SR(t) = E(t) / baseline_stats[t]['enrichment_{p}th']
```
`SR_max` is the maximum SR over all extreme terms in the signature. It captures the single most outstanding enrichment signal.

### `related_genes_fraction`
Two genes are considered **related** if their extreme terms are connected in the GO hierarchy (one is an ancestor or descendant of the other, transitively). The script builds an undirected graph where genes are nodes and an edge is drawn between two genes if any of their extreme terms are related. Connected components of size ≥ 2 are identified via BFS. The metric is:
```
related_genes_fraction = |genes in components of size ≥ 2| / signature_length
```

### `relationship_genes_fraction`
Similar to above, but instead of is-a/part-of hierarchy, it uses **non-hierarchical GO relationships** (e.g. `regulates`, `positively_regulates`, `occurs_in`). Two terms are considered connected if:
- Term A has a direct relationship to term B, or
- Any ancestor of A has a relationship to B or any descendant of B

The same BFS connected-component logic is applied to count genes in such relationship-connected components.

---

## Step 6 — Output

For each signature length and each repeat, all metric values across random signatures are saved in two files per percentile `p`:

- **`cluster_statistics_{p}th.json`** — summary statistics (mean, std, percentiles) of each cluster metric across all random signatures
- **`cluster_raw_values_{p}th.json`** — the raw per-signature values of each metric (used for visualization and fallback quantile calculation)
- **`enrichment_baseline_stats.json`** — per GO term enrichment statistics across all random signatures

---

## Usage

```bash
python baseline.py \
  --num_of_iterations 1000 \
  --signature_lengths "50 100 200" \
  --percentile "90 99" \
  --output_dir baselines \
  --num_repeats 3
```

| Argument | Required | Description |
|---|---|---|
| `--num_of_iterations` | Yes | Number of random signatures for the longest length; shorter lengths scale proportionally |
| `--signature_lengths` | Yes | Space-separated list of signature lengths to compute baselines for |
| `--percentile` | No | One or more percentile thresholds for extreme term definition (default: 90) |
| `--output_dir` | No | Root output directory (default: current directory) |
| `--num_repeats` | No | Number of independent repetitions for stability estimation (default: 1) |
| `--n_cores` | No | CPU cores to use; defaults to all minus one |
