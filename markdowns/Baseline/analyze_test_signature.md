# analyze_test_signature.py

## Purpose

This script takes one or more real gene signatures and evaluates how biologically structured they are, by measuring each signature's cluster metrics and comparing them to the null distribution from `baseline.py`. The comparison yields a **Signal-to-Noise Ratio (SNR)** per metric and per GO namespace, expressing how far above random chance the signature sits.

---

## Input

A **gene signature** is a plain text file with one gene symbol per line. It represents a set of biologically meaningful genes — for example, genes upregulated in a cancer subtype, or genes belonging to a known pathway. The script processes one or more such files in a single run.

---

## Step 1 — Loading and aggregating baseline data

For each signature, the script determines the signature length and loads the corresponding baseline from disk. If the exact length does not exist, it falls back to an interpolated baseline (see `interpolate_baseline.py`).

If the baseline was computed with multiple repeats, the enrichment statistics per GO term are **averaged across repeats**:

```
aggregated_stat = mean(repeat_0_stat, repeat_1_stat, ..., repeat_k_stat)
```

This produces a single stable set of per-term thresholds that represent the expected null behavior more reliably than any single repeat.

---

## Step 2 — Enrichment and extreme term identification

The same enrichment calculation used in `baseline.py` is applied to the test signature. For each GO term `t` appearing in the signature:

```
E(t) = (direct(t) + inherited(t)) / max(1, signature_length × frequency(t))
```

Terms with no baseline entry (i.e. terms so rare they never appeared in random signatures) are assigned a fallback baseline of `enrichment_{p}th = 1.0`. This is conservative: it means any enrichment above 1.0 counts as extreme for such terms, which is appropriate because they were too rare to characterize statistically.

A term is **extreme** if:
```
E(t) / baseline_stats[t]['enrichment_{p}th'] > 1
```

---

## Step 3 — Cluster metrics on the test signature

The same four cluster metrics computed during baseline construction are now computed for the actual test signature. Their values reflect the real biological structure in the data:

- **`extreme_fraction_genes`** — what fraction of the signature's genes carry a term that is enriched beyond the random threshold
- **`SR_max`** — how far the most outstanding GO term exceeds its own baseline threshold: `max over extreme terms of E(t) / baseline_p(t)`
- **`related_genes_fraction`** — fraction of genes in connected components (size ≥ 2) where connection means their extreme terms are hierarchically related in the GO DAG
- **`relationship_genes_fraction`** — same, but connection requires non-hierarchical GO relationships (e.g. `regulates`, `occurs_in`)

---

## Step 4 — SNR calculation

For each metric `m`, the baseline reference value is taken as the **mean of the chosen quantile** (e.g. 99th percentile) across all repeats:

```
baseline_ref(m) = mean over repeats of percentile_q(random_distribution_of_m)
```

The SNR is then:

```
SNR(m) = test_value(m) / baseline_ref(m)
```

- **SNR = 1.0** — the signature performs exactly at the chosen quantile of random signatures
- **SNR > 1.0** — the signature exceeds that quantile (signal above noise)
- **SNR < 1.0** — the signature falls below even the chosen quantile (indistinguishable from or weaker than noise)

### Error propagation

The baseline reference has uncertainty because it is estimated from a finite number of random signatures, quantified by `baseline_std`. The uncertainty on the SNR is propagated via the quotient rule:

```
σ_SNR = (test_value / baseline_ref²) × baseline_std
```

This gives the error bar shown on the SNR bar chart.

---

## Step 5 — Panel C score (overall signal quality)

For each namespace, a single **signal quality score** is computed as the mean SNR across all valid metrics:

```
score(namespace) = mean(SNR(m) for m in valid_metrics)
```

Metrics are excluded from this mean if their baseline is zero (no signal possible) or, specifically for MF, `relationship_genes_fraction` is excluded because GO relationship edges are extremely sparse in the Molecular Function namespace, making this metric unreliable there.

The combined error for the mean score uses propagated errors:

```
σ_score = sqrt(Σ σ²_SNR(m)) / n_metrics
```

### Classification thresholds

| Score | Classification |
|---|---|
| ≥ 1.5 | STRONG — clearly non-random biological structure |
| 1.0 – 1.5 | MODERATE — signal present but not overwhelming |
| < 1.0 | NOISE/WEAK — indistinguishable from random |

---

## Output

For each input signature, a subfolder is created in `--output_path` containing:

**Three PNG files** (one per namespace: BP, MF, CC), each with three panels:

- **Panel A** — SNR bar chart per metric, colored green (strong) through red (weak), with propagated error bars
- **Panel B** — Heatmap with three columns: test value, baseline reference, and SNR side by side for all metrics
- **Panel C** — Horizontal bar chart of the overall signal quality score per namespace

**One JSON file** (`baseline_analysis.json`) with all numeric results including test values, baseline means/stds, SNR values, propagated errors, and Panel C scores.

---

## Usage

```bash
python analyze_test_signature.py \
  --input_path analysis/signatures \
  --output_path analysis/results \
  --test_signatures hallmark_EMT.txt \
  --baseline_dir baselines \
  --percentile 90 \
  --quantile 99
```

| Argument | Required | Description |
|---|---|---|
| `--input_path` | Yes | Directory containing the signature `.txt` files |
| `--output_path` | Yes | Directory where all results are saved |
| `--test_signatures` | Yes | One or more signature filenames |
| `--baseline_dir` | Yes | Directory with precomputed `baseline_{length}` folders |
| `--percentile` | Yes | Percentile threshold `p` for defining extreme GO terms |
| `--quantile` | Yes | Quantile `q` of the baseline cluster metric distribution used as SNR denominator |
| `--cluster_metrics` | No | Metrics to evaluate (default: `extreme_fraction_genes SR_max related_genes_fraction relationship_genes_fraction`) |
