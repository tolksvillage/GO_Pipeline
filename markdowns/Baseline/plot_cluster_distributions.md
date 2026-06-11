# plot_cluster_distributions.py

## Purpose

This script visualizes the empirical distribution of each cluster metric across all random signatures in the baseline, separated by GO namespace. It serves as a diagnostic tool to verify that the baseline is well-behaved, stable across repeats, and that the chosen percentile thresholds fall at sensible positions in the distribution.

---

## What is being plotted

The input file `cluster_raw_values_{p}th.json` contains, for each namespace and each repeat, the raw metric value computed for every individual random signature. For example, for `extreme_fraction_genes` in `BP`, `repeat_0`, there is a list of N values — one per random signature — recording what fraction of each random signature's genes carried an extreme GO term.

For each metric and namespace, the script pools these raw values across all repeats to construct a histogram and computes summary statistics.

---

## Mathematical content of each plot

### Histogram (bars)

The raw values from each repeat are binned using a common set of bin edges derived from the combined data across all repeats. For each repeat independently, the histogram is normalized to percentage:

```
percentage_in_bin = (count_in_bin / total_signatures_in_repeat) × 100
```

This gives one percentage-histogram per repeat. The bars in the plot show the **mean percentage** across repeats at each bin:

```
bar_height(bin) = mean over repeats of percentage_in_bin(bin)
```

### Error bars

The error bars on each bar represent the **standard deviation** of the per-bin percentages across repeats:

```
error(bin) = std over repeats of percentage_in_bin(bin)
```

Small error bars indicate that the distribution is stable across repeats — the baseline is reproducible. Large error bars suggest that the metric is highly variable and more repeats may be needed for a reliable baseline.

### Vertical reference lines

Four vertical lines are drawn on each panel, computed from all raw values pooled across all repeats:

- **Mean** (red dashed) — the arithmetic average of the metric across all random signatures
- **Median** (dark green dashed) — the 50th percentile; if the distribution is symmetric, mean and median coincide; if they differ, the distribution is skewed
- **90th percentile** (purple dotted) — 90% of random signatures fall below this value; used as the extreme term threshold when `--percentile 90` is chosen in `baseline.py`
- **99th percentile** (orange dotted) — 99% of random signatures fall below this value; used as the threshold when `--percentile 99` is chosen

These lines allow you to visually inspect where the chosen threshold sits in the distribution and whether the chosen quantile is meaningfully above the bulk of the distribution.

---

## Percentile extraction from filename

The percentile value is not passed as an argument. Instead, it is parsed directly from the filename using a regular expression matching the pattern `_{digits}th.json`. For example:

```
cluster_raw_values_90th.json  →  percentile = 90
cluster_raw_values_99th.json  →  percentile = 99
```

This ensures the output filename and the reference lines always correspond to the correct threshold without requiring manual input.

---

## Output

One PNG file per namespace is saved:

```
{output_dir}/
    BP_cluster_statistics_{p}th.png
    MF_cluster_statistics_{p}th.png
    CC_cluster_statistics_{p}th.png
```

Each file contains one subplot per metric arranged in a 2-column grid (or single column for ≤ 3 metrics).

---

## Usage

```bash
python plot_cluster_distributions.py \
  --raw_values_file baselines/baseline_200/cluster_raw_values_90th.json \
  --plot_statistics "extreme_fraction_genes SR_max related_genes_fraction relationship_genes_fraction" \
  --output_dir baselines/plots
```

| Argument | Required | Description |
|---|---|---|
| `--raw_values_file` | Yes | Path to a `cluster_raw_values_{p}th.json` file; percentile is extracted from the filename automatically |
| `--plot_statistics` | Yes | Space-separated list of metric names to include in the plots |
| `--output_dir` | No | Directory where the PNG files are saved (default: current directory) |
