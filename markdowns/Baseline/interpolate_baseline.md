# interpolate_baseline.py

## Purpose

This script estimates baseline statistics for signature lengths that were never explicitly computed. Rather than running `baseline.py` for every possible gene count, baselines are computed at a few representative lengths (e.g. 50, 100, 150, 200) and the values in between are estimated by fitting a curve through the known points and evaluating it at the desired length.

---

## Why interpolation is necessary

The cluster metrics and enrichment thresholds depend on signature length. For example, a random signature of 200 genes will, on average, carry more extreme GO terms than one of 50 genes — simply because more genes means more GO term coverage. This means you cannot reuse a baseline computed for length 200 when analyzing a signature of length 175: the thresholds would be systematically too high, making the real signature appear weaker than it is.

Running `baseline.py` for every integer from 50 to 500 would require enormous computation time. Interpolation allows you to cover the full range from just a handful of anchor points.

---

## What gets interpolated

Every numeric statistic in every baseline file is interpolated independently:

### Enrichment baseline (`enrichment_baseline_stats.json`)
For each GO term `t` and each statistic `s` (mean, std, 90th percentile, etc.), the values at the known lengths `L = [l_1, l_2, ..., l_k]` are collected and a curve is fitted:

```
s(t, L) = [s(t, l_1), s(t, l_2), ..., s(t, l_k)]
```

The fitted function is then evaluated at the target length `l*` to produce `s(t, l*)`.

### Cluster statistics (`cluster_statistics_{p}th.json`)
The same process is applied to every statistic of every cluster metric (e.g. the 99th percentile of `extreme_fraction_genes` across random signatures). Since the baseline may have been computed with multiple repeats, values are first **averaged across repeats** at each length before fitting:

```
mean_across_repeats(s, metric, l_i) → single value to interpolate
```

The interpolated output maintains the repeat structure of the original files (all repeats receive the same interpolated value), so that downstream scripts can read it without modification.

---

## Interpolation methods

The curve fitting is done using `scipy.interpolate.interp1d` with `fill_value='extrapolate'`, meaning the function can also estimate values outside the range of known lengths if needed.

Three methods are available:

**Linear** — fits straight line segments between adjacent known points. Simple and numerically stable, but cannot capture curvature between points:
```
f(l) = y_i + (y_{i+1} - y_i) / (l_{i+1} - l_i) × (l - l_i)
```

**Quadratic** — fits a piecewise quadratic polynomial through the known points. Captures gentle curves.

**Cubic** — fits a piecewise cubic spline through the known points, ensuring smooth first and second derivatives at each known point. Best for most biological data because the true relationship between signature length and metric values tends to be smooth. Requires at least 4 known points; automatically falls back to linear if fewer are available.

---

## Handling multiple repeats

When baselines have been computed with multiple repeats (for stability estimation), the interpolation proceeds as follows:

1. At each known length `l_i`, compute the mean of the statistic across all repeats
2. Fit a curve through the per-length means
3. Evaluate the curve at the target length `l*`
4. Write the interpolated value into all repeat slots of the output file

This means the interpolated baseline does not carry repeat-level variance (all repeats are identical), but the mean is a stable estimate of the true expected value at that length.

---

## Optional visualization

With `--visualize`, a diagnostic plot is produced for the chosen namespace and percentile. For each metric, the plot shows:

- Individual repeat data points at each known length (colored dots)
- The mean across repeats at each length (red diamond)
- A shaded ±1 standard deviation band around the mean (gray), reflecting repeat-to-repeat variability
- The fitted interpolation curve (red line) across the full range

This lets you visually confirm that the curve follows the data sensibly and does not oscillate or diverge.

---

## Output

The interpolated data is saved with the same directory structure as a regular baseline:

```
{output_dir}/baseline_{target_length}/
    enrichment_baseline_stats.json
    cluster_statistics_90th.json
    cluster_statistics_99th.json
```

`analyze_test_signature.py` automatically searches for this folder under `{baseline_dir}/interpolated/baseline_{length}/` if the exact baseline is not found directly under `{baseline_dir}/baseline_{length}/`.

---

## Usage

```bash
python interpolate_baseline.py \
  --base_dir baselines \
  --signature_lengths "50 100 150 200" \
  --target_length 175 \
  --output_dir baselines/interpolated \
  --method cubic \
  --visualize \
  --vis_namespace BP \
  --vis_percentile 90
```

| Argument | Required | Description |
|---|---|---|
| `--base_dir` | Yes | Directory containing the `baseline_{length}` anchor folders |
| `--target_length` | Yes | The signature length to produce a baseline for |
| `--output_dir` | Yes | Where to save the interpolated baseline |
| `--signature_lengths` | No | Space-separated list of available anchor lengths (default: `50 100 150 200 300`) |
| `--method` | No | Interpolation method: `linear`, `quadratic`, or `cubic` (default: `cubic`) |
| `--visualize` | No | Flag — if set, saves diagnostic plots |
| `--vis_namespace` | No | Namespace to plot: `BP`, `MF`, or `CC` (default: `BP`) |
| `--vis_percentile` | No | Percentile to plot (default: `90`) |
